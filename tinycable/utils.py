from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


class Ns(dict[str, Any]):
    """Dictionary namespace with attribute access."""

    __getattr__ = dict.__getitem__


T = TypeVar("T")


def readonly(values: npt.ArrayLike, dtype: Any = None) -> np.ndarray:
    """Copy an array-like value and make the result read-only."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def dict2mapping(values: Mapping[str, T]) -> Mapping[str, T]:
    """Copy a dictionary into a read-only mapping."""
    return MappingProxyType(dict(values))


def assert_index(index: npt.ArrayLike, sorted: bool = True) -> np.ndarray:
    index = readonly(index, dtype=np.int32)
    assert index.ndim == 1, "indices must be one-dimensional"
    if sorted:
        assert np.all(index[1:] > index[:-1]), "index must be sorted and deduplicated"
    return index


IDENTITY: Final = object()


def _safe_exp(x: jax.Array) -> jax.Array:
    limit = -jnp.log(jnp.finfo(x.dtype).eps)
    return jnp.exp(jnp.minimum(x, limit))


def gather(x: jax.Array, idx: Any) -> jax.Array:
    """Gather slots while preserving the measured fast paths."""
    if x.shape[0] == 1:
        return x[0]
    if idx is IDENTITY:
        return x
    return x[idx]


def scatter_add(out: jax.Array, idx: Any, val: jax.Array) -> jax.Array:
    """Scatter-add slot values while bypassing identity projections."""
    if idx is IDENTITY:
        return out + val
    return out.at[idx].add(val)


def vtrap(x: jax.Array, scale: float | jax.Array) -> jax.Array:
    threshold = jnp.sqrt(jnp.finfo(x.dtype).eps) * scale
    near_zero = jnp.abs(x) < threshold
    safe_x = jnp.where(near_zero, scale, x)
    exact = safe_x / -jnp.expm1(-safe_x / scale)
    return jnp.where(near_zero, scale + x / 2.0, exact)
