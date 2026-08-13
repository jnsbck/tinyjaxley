from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .mechanism import Mechanism
    from .model import Model

from .field import Field
from .mechanism import Mechanism, Synapse
from .utils import IDENTITY

T = TypeVar("T")


@jax.tree_util.register_pytree_node_class
class FrozenMap(Mapping[str, T], Generic[T]):
    """Small immutable, named pytree mapping."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, T] | None = None) -> None:
        self._data = dict(values or {})

    def __getitem__(self, key: str) -> T:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"

    def tree_flatten(self) -> tuple[tuple[T, ...], tuple[str, ...]]:
        return tuple(self._data.values()), tuple(self._data)

    @classmethod
    def tree_unflatten(
        cls, keys: tuple[str, ...], values: tuple[T, ...]
    ) -> "FrozenMap[T]":
        return cls(dict(zip(keys, values, strict=True)))

    __eq__ = object.__eq__
    __hash__ = object.__hash__


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True, eq=False)
class Access:
    """One resolved pool reference and its optional slot projection."""

    ref: str
    index: jax.Array | None = None

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], tuple[str, bool]]:
        if self.index is None:
            return (), (self.ref, True)
        return (self.index,), (self.ref, False)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[str, bool], children: tuple[jax.Array, ...]
    ) -> "Access":
        ref, identity = aux
        return cls(ref, None if identity else children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True, eq=False)
class BoundMechanism:
    """A mechanism equation paired with its resolved runtime accesses."""

    kernel: "Mechanism"
    sink: jax.Array | None
    ns: FrozenMap[FrozenMap[Access]]

    def tree_flatten(self) -> tuple[tuple[object, ...], "_KernelMeta"]:
        return (self.sink, self.ns), _KernelMeta(self.kernel)

    @classmethod
    def tree_unflatten(
        cls, kernel: "_KernelMeta", children: tuple[object, ...]
    ) -> "BoundMechanism":
        return cls(kernel.kernel, *children)


class _KernelMeta:
    __slots__ = ("kernel",)

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _KernelMeta) and self.kernel is other.kernel

    def __hash__(self) -> int:
        return id(self.kernel)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True, eq=False)
class Runtime:
    """Bound, executable model values and static equation metadata."""

    pool: FrozenMap[jax.Array]
    mechs: FrozenMap[BoundMechanism]
    syns: FrozenMap[BoundMechanism]
    morph: FrozenMap[jax.Array | None]
    dynamic: frozenset[str]

    @property
    def n(self) -> int:
        return self.pool["v"].shape[0]

    def split(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return dynamic state and static parameter views of ``pool``."""
        dynamic = {
            name: value for name, value in self.pool.items() if name in self.dynamic
        }
        static = {
            name: value for name, value in self.pool.items() if name not in self.dynamic
        }
        return dynamic, static

    def tree_flatten(self) -> tuple[tuple[Any, ...], frozenset[str]]:
        return (self.pool, self.mechs, self.syns, self.morph), self.dynamic

    @classmethod
    def tree_unflatten(
        cls,
        aux: frozenset[str],
        children: tuple[Any, ...],
    ) -> "Runtime":
        return cls(*children, dynamic=aux)


_MORPH_FIELDS = ("cm", "area", "Ra", "rin", "rout")


def _device(device: Any) -> Any:
    if device is None or not isinstance(device, str):
        return device
    try:
        return jax.devices(device)[0]
    except RuntimeError as error:
        raise RuntimeError(
            f"no JAX device available for platform {device!r}"
        ) from error


def _projection(field: Field, sites: np.ndarray, device: Any) -> object:
    """Resolve and specialize one validated Field projection."""
    raw = field.slot_index(sites)
    identity = np.arange(field.n_slots, dtype=np.int32)
    if np.array_equal(sites, field.index) and np.array_equal(raw, identity):
        return IDENTITY
    return jax.device_put(jnp.asarray(raw, dtype=jnp.int32), device=device)


def _access(field: Field, ref: str, sites: np.ndarray, device: Any) -> Access:
    projection = _projection(field, sites, device)
    return Access(ref, None if projection is IDENTITY else projection)


def _require_field(fields: Mapping[str, Field], name: str, owner: str) -> Field:
    try:
        return fields[name]
    except KeyError as error:
        raise ValueError(f"{owner} references unknown field {name!r}") from error


def _bound(
    model: "Model",
    mechanism: Mechanism,
    voltage: Field,
    device: Any,
) -> BoundMechanism:
    assert mechanism.index is not None
    fields = model._fields
    sites = mechanism.index
    namespaces: dict[str, dict[str, Access]] = {
        "s": {},
        "p": {},
    }

    for namespace, accessor, declaration in mechanism.declarations():
        ref = accessor if declaration is None else declaration.ref
        assert ref is not None
        field = _require_field(fields, ref, mechanism.name)
        namespaces.setdefault(namespace, {})[accessor] = _access(
            field, ref, sites, device
        )

    sink_sites = mechanism.post_index if isinstance(mechanism, Synapse) else sites
    if sink_sites is None:
        raise AssertionError("Synapse insertion requires post_index")
    sink = _projection(voltage, sink_sites, device)

    if isinstance(mechanism, Synapse):
        assert mechanism.pre_index is not None
        for namespace, names, endpoint in (
            ("pre", mechanism.pre, mechanism.pre_index),
            ("post", mechanism.post, mechanism.post_index),
        ):
            assert endpoint is not None
            namespace_access = namespaces.setdefault(namespace, {})
            for field_name in names:
                field = _require_field(fields, field_name, mechanism.name)
                namespace_access[field_name] = _access(
                    field, field_name, endpoint, device
                )

    return BoundMechanism(
        kernel=mechanism,
        sink=None if sink is IDENTITY else sink,
        ns=FrozenMap({name: FrozenMap(values) for name, values in namespaces.items()}),
    )


def _pool(model: "Model", *, device: Any, dtype: Any) -> FrozenMap[jax.Array]:
    target = None if dtype is None else np.dtype(dtype)
    if (
        target is not None
        and target == np.dtype(np.float64)
        and not jax.config.read("jax_enable_x64")
    ):
        raise RuntimeError("float64 requested, but jax_enable_x64 is disabled")

    values: dict[str, jax.Array] = {}
    for name, field in model._fields.items():
        source = field.slots
        cast = target if np.issubdtype(source.dtype, np.inexact) else None
        values[name] = jax.device_put(jnp.asarray(source, dtype=cast), device=device)
    return FrozenMap(values)


def _bind(model: "Model", *, device: Any = None, dtype: Any = None) -> Runtime:
    """Bind declarations into an inspectable, executable Runtime."""
    device = _device(device)
    voltage = _require_field(model._fields, "v", "model")
    if _projection(voltage, model.morph.index, device) is not IDENTITY:
        raise ValueError("voltage field must have one ordered slot per compartment")

    mechs = {
        name: _bound(model, mechanism, voltage, device)
        for name, mechanism in model._mechs.items()
    }
    syns = {
        name: _bound(model, synapse, voltage, device)
        for name, synapse in model._syns.items()
    }
    morph = {
        name: _projection(
            _require_field(model._fields, name, "morphology"),
            model.morph.index,
            device,
        )
        for name in _MORPH_FIELDS
    }
    return Runtime(
        pool=_pool(model, device=device, dtype=dtype),
        mechs=FrozenMap(mechs),
        syns=FrozenMap(syns),
        morph=FrozenMap(
            {
                name: None if projection is IDENTITY else projection
                for name, projection in morph.items()
            }
        ),
        dynamic=frozenset(
            name for name, field in model._fields.items() if field.dynamic
        ),
    )
