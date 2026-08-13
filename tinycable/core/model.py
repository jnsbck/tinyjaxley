import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt

from .field import Field
from .mechanism import Mechanism, Synapse
from .morphology import Morphology
from .utils import dict2mapping

if TYPE_CHECKING:
    from .runtime import Runtime


def _insert_field(fields: dict[str, Field], field: Field) -> None:
    assert field.ref is not None, "stored fields must have a pool reference"
    previous = fields.get(field.ref)
    fields[field.ref] = field if previous is None else previous.extend(field)


def _insert_declared_fields(fields: dict[str, Field], mech: Mechanism) -> None:
    for _, _, field in mech.declarations():
        if field is not None:
            _insert_field(fields, field)


def _assert_sites(index: np.ndarray, n: int, name: str) -> None:
    assert np.all((index >= 0) & (index < n)), (
        f"{name} must reference model compartments"
    )


def _validate_current_names(item: Mechanism) -> set[str]:
    names = item.currents
    if not all(isinstance(name, str) for name in names):
        raise TypeError("Mechanism currents must be strings")
    unique = set(names)
    if len(unique) != len(names):
        raise ValueError(f"{type(item).__name__} currents must be unique")
    return unique


def _validate_cross_kind_names(
    item: Mechanism,
    other: Mapping[str, Mechanism],
    other_kind: str,
) -> None:
    if item.name in other:
        raise ValueError(
            f"{item.name!r} is already used by a {other_kind}; "
            "Mechanism and Synapse names must be disjoint"
        )

    current_names = _validate_current_names(item)
    other_current_names = {
        current for declaration in other.values() for current in declaration.currents
    }
    collision = current_names & other_current_names
    if collision:
        raise ValueError(
            f"current names {sorted(collision)!r} are already used by a "
            f"{other_kind}; Mechanism and Synapse current names must be disjoint"
        )


@dataclass(frozen=True, init=False, eq=False)
class Model:
    """Immutable morphology, field, mechanism, and Synapse declarations."""

    morph: Morphology
    _fields: Mapping[str, Field]
    _mechs: Mapping[str, Mechanism]
    _syns: Mapping[str, Synapse]
    _num_syns: int

    def __init__(
        self,
        morph: Morphology,
        v: Any = -65.0,
        Ra: Any = 100.0,
        cm: Any = 1.0,
    ) -> None:
        index = morph.index
        fields = {
            "v": Field("v", v, dynamic=True, index=index),
            "Ra": Field("Ra", Ra, index=index),
            "cm": Field("cm", cm, index=index),
            "rad": Field("rad", morph.rad, index=index),
            "len": Field("len", morph.len, index=index),
            "area": Field("area", morph.area, index=index),
            "volume": Field("volume", morph.volume, index=index),
            "rin": Field("rin", morph.rin, index=index),
            "rout": Field("rout", morph.rout, index=index),
        }
        assert all(field.value_shape == () for field in fields.values()), (
            "model base fields must be scalar-valued"
        )
        object.__setattr__(self, "morph", morph)
        object.__setattr__(self, "_fields", dict2mapping(fields))
        object.__setattr__(self, "_mechs", dict2mapping({}))
        object.__setattr__(self, "_syns", dict2mapping({}))
        object.__setattr__(self, "_num_syns", 0)

    def _with(
        self,
        *,
        fields: dict[str, Field] | None = None,
        mechs: dict[str, Mechanism] | None = None,
        syns: dict[str, Synapse] | None = None,
        num_syns: int | None = None,
    ) -> Self:
        model = object.__new__(type(self))
        object.__setattr__(model, "morph", self.morph)
        object.__setattr__(
            model,
            "_fields",
            self._fields if fields is None else dict2mapping(fields),
        )
        object.__setattr__(
            model,
            "_mechs",
            self._mechs if mechs is None else dict2mapping(mechs),
        )
        object.__setattr__(
            model,
            "_syns",
            self._syns if syns is None else dict2mapping(syns),
        )
        object.__setattr__(
            model,
            "_num_syns",
            self._num_syns if num_syns is None else num_syns,
        )
        return model

    def insert(self, item: Field | Mechanism) -> Self:
        if isinstance(item, Field):
            if item.index is None and self._syns:
                raise ValueError(
                    "unplaced Fields require an explicit index after Synapses "
                    "have been inserted"
                )
            fields = dict(self._fields)
            if item.index is None:
                item = item._insert(self.morph.index)
            _insert_field(fields, item)
            return self._with(fields=fields)

        if isinstance(item, Synapse):
            assert item.index is None, (
                "Synapse edge indices are assigned by Model.insert"
            )
            _validate_cross_kind_names(item, self._mechs, "Mechanism")
            assert item.pre_index is not None and item.post_index is not None, (
                "Synapse insertion requires pre_index and post_index"
            )
            assert len(item.pre_index) > 0, (
                "Synapse insertion requires at least one edge"
            )
            _assert_sites(item.pre_index, self.morph.n, "Synapse pre_index")
            _assert_sites(item.post_index, self.morph.n, "Synapse post_index")
            n_new = len(item.pre_index)

            fields = dict(self._fields)
            syns = dict(self._syns)
            previous = syns.get(item.name)

            # Same-name insertions append edge instances and replace declarations.
            new_index = np.arange(
                self.morph.n + self._num_syns,
                self.morph.n + self._num_syns + n_new,
                dtype=np.int32,
            )
            if previous is not None:
                new_index = np.concatenate((previous.index, new_index))
                pre_index = np.concatenate((previous.pre_index, item.pre_index))
                post_index = np.concatenate((previous.post_index, item.post_index))
            else:
                pre_index = item.pre_index
                post_index = item.post_index

            item = replace(
                item,
                index=new_index,
                pre_index=pre_index,
                post_index=post_index,
            )
            syns[item.name] = item
            _insert_declared_fields(fields, item)
            return self._with(
                fields=fields,
                syns=syns,
                num_syns=self._num_syns + n_new,
            )

        fields = dict(self._fields)
        mechs = dict(self._mechs)
        _validate_cross_kind_names(item, self._syns, "Synapse")
        if item.index is None:
            item = replace(item, index=self.morph.index)
        _assert_sites(item.index, self.morph.n, "mechanism index")
        previous = mechs.get(item.name)
        # Name alone selects the instance set; a different type may overwrite it.
        mech_index = (
            np.union1d(previous.index, item.index).astype(np.int32)
            if previous is not None
            else item.index
        )
        item = replace(item, index=mech_index)
        mechs[item.name] = item
        _insert_declared_fields(fields, item)
        return self._with(fields=fields, mechs=mechs)

    def share(
        self,
        name: str,
        groups: npt.ArrayLike | None = None,
        *,
        reduce: Callable[[np.ndarray], Any] | None = None,
    ) -> Self:
        if name not in self._fields:
            raise ValueError(f"unknown field {name!r}")
        fields = dict(self._fields)
        field = fields[name]
        if groups is None:
            groups = np.zeros(len(field.index), dtype=np.int32)
        fields[name] = field.regroup(groups, reduce=reduce)
        return self._with(fields=fields)

    def unshare(self, name: str) -> Self:
        if name not in self._fields:
            raise ValueError(f"unknown field {name!r}")
        groups = np.arange(len(self._fields[name].index), dtype=np.int32)
        return self.share(name, groups)

    def set(
        self,
        name: str,
        value: Any,
        *,
        at: npt.ArrayLike | None = None,
    ) -> Self:
        """Return a copy with values replaced in one stored Field."""
        if name not in self._fields:
            raise ValueError(f"unknown field {name!r}")
        fields = dict(self._fields)
        fields[name] = fields[name].set(value, at=at)
        return self._with(fields=fields)

    def values(
        self,
        fmt: str = "str",
        *,
        role: bool = False,
        shape: bool = False,
        slot: bool = True,
        sites: bool = True,
        filter: Any = None,
        compress: bool = False,
    ) -> str | dict[str, dict[str, Any]]:
        """Return per-slot representations of all stored Fields."""
        if fmt not in {"str", "dict"}:
            raise ValueError(f"unknown values format {fmt!r}")
        fields = {}
        for name, field in self._fields.items():
            data = field.render(
                fmt="dict",
                role=role,
                shape=shape,
                slot=slot,
                sites=sites,
                filter=filter,
                compress=compress,
            )
            if data:
                fields[name] = data
        if fmt == "dict":
            return fields
        from tinycable.extra.render import fields_dict_to_str

        return fields_dict_to_str(fields)

    def remove(self, name: str) -> Self:
        if name not in self._mechs:
            return self

        warnings.warn(
            f"Removing mechanism {name!r} leaves its declared fields in the model.",
            UserWarning,
            stacklevel=2,
        )
        mechs = dict(self._mechs)
        del mechs[name]
        # TODO: Track field ownership so removing or replacing a mechanism can
        # remove stale fields and support contributed only by that mechanism.
        return self._with(mechs=mechs)

    def bind(self, *, device: Any = None, dtype: Any = None) -> "Runtime":
        """Return an executable Runtime with values placed on a JAX device."""
        from .runtime import _bind

        return _bind(self, device=device, dtype=dtype)
