import jax.numpy as jnp


class Ns(dict):
    """Dictionary namespace with attribute access."""

    __getattr__ = dict.__getitem__


IDENTITY = object()


def _safe_exp(x):
    limit = -jnp.log(jnp.finfo(x.dtype).eps)
    return jnp.exp(jnp.minimum(x, limit))


def gather(x, idx):
    """Gather slots while preserving the measured fast paths."""
    if x.shape[0] == 1:
        return x[0]
    if idx is IDENTITY:
        return x
    return x[idx]


def scatter_add(out, idx, val):
    """Scatter-add slot values while bypassing identity projections."""
    if idx is IDENTITY:
        return out + val
    return out.at[idx].add(val)


def vtrap(x, scale):
    threshold = jnp.sqrt(jnp.finfo(x.dtype).eps) * scale
    near_zero = jnp.abs(x) < threshold
    safe_x = jnp.where(near_zero, scale, x)
    exact = safe_x / -jnp.expm1(-safe_x / scale)
    return jnp.where(near_zero, scale + x / 2.0, exact)
