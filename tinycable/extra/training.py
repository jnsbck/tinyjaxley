from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tinycable.core.runtime import FrozenMap, Runtime


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenPartition:
    runtime: Runtime
    selections: tuple[tuple[str, tuple[int, ...] | None], ...]


def _mask(value: Any, n_slots: int, name: str) -> bool | np.ndarray:
    if isinstance(value, jax.core.Tracer):
        raise TypeError(f"mask for {name!r} must be concrete")
    if isinstance(value, jax.Array):
        value = np.asarray(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != (n_slots,):
        raise ValueError(f"mask for {name!r} must be boolean with shape {(n_slots,)}")
    return value


def partition(
    runtime: Runtime,
    mask: Mapping[str, Any],
) -> tuple[dict[str, jax.Array], _FrozenPartition]:
    """Select complete physical Field slots for optimization."""
    if not isinstance(runtime, Runtime):
        raise TypeError("partition expects a Runtime")
    if not isinstance(mask, Mapping):
        raise TypeError("partition mask must be a mapping")
    unknown = set(mask) - set(runtime.pool)
    if unknown:
        raise KeyError(f"unknown pool fields in partition mask: {sorted(unknown)}")

    theta: dict[str, jax.Array] = {}
    selections: list[tuple[str, tuple[int, ...] | None]] = []
    for name, value in runtime.pool.items():
        if name not in mask:
            continue
        requested = _mask(mask[name], value.shape[0], name)
        if requested is False:
            continue
        if requested is True:
            slots = None
            selected = value
        else:
            indices = np.flatnonzero(requested).astype(np.int32)
            if not len(indices):
                continue
            slots = tuple(int(index) for index in indices)
            selected = value[indices]
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError(f"pool field {name!r} is not inexact")
        theta[name] = selected
        selections.append((name, slots))
    return theta, _FrozenPartition(runtime, tuple(selections))


def combine(theta: Mapping[str, Any], frozen: _FrozenPartition) -> Runtime:
    """Reconstruct a Runtime while keeping unselected slots frozen."""
    if not isinstance(frozen, _FrozenPartition):
        raise TypeError("combine expects the frozen result of partition")
    expected = {name for name, _ in frozen.selections}
    if set(theta) != expected:
        raise KeyError(f"combine expected theta keys {sorted(expected)}")

    pool = {
        name: jax.lax.stop_gradient(value)
        for name, value in frozen.runtime.pool.items()
    }
    for name, slots in frozen.selections:
        value = theta[name]
        if not isinstance(value, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"theta[{name!r}] must be a JAX array")
        base = pool[name]
        shape = base.shape if slots is None else (len(slots), *base.shape[1:])
        if value.shape != shape:
            raise ValueError(
                f"theta[{name!r}] has shape {value.shape}, expected {shape}"
            )
        if value.dtype != base.dtype:
            raise ValueError(
                f"theta[{name!r}] has dtype {value.dtype}, expected {base.dtype}"
            )
        pool[name] = (
            value
            if slots is None
            else base.at[jnp.asarray(slots, dtype=jnp.int32)].set(value)
        )
    return replace(frozen.runtime, pool=FrozenMap(pool))
