import jax.numpy as jnp
import jax
import equinox as eqx

from jax import Array


def safe_exp(x: Array, max_value: float = 20.0):
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)


def _vtrap(x, y):
    return x / (safe_exp(x / y) - 1.0)


def tree_path_of_leaves(model, filter: str | list[str] = None, invert=False):
    # TODO: allow partial matches, like .channels -> should show all channels
    # TODO: prevent .c from matching .channels as well
    def label_fn(path, value):
        path_str = "".join([str(p) for p in path])
        cond = (path_str in filter) if filter else True
        return path_str if cond != invert else None

    return jax.tree_util.tree_map_with_path(label_fn, model)


def tree_filter_by_path(model, filter: str | list[str], invert=False):
    # TODO: add option for is_leaf, i.e. to allow filtering partial paths, like .channels -> should show all channels
    def label_fn(path, value):
        path_str = "".join([str(p) for p in path])
        return value if (path_str in filter) != invert else None

    return jax.tree_util.tree_map_with_path(label_fn, model)


def tree_apply_with_path(model, apply_dict):
    def label_fn(path, value):
        path_str = "".join([str(p) for p in path])
        return apply_dict.get(path_str, lambda x: None)(value)

    new_vals = jax.tree.map(lambda x: None, model)
    new_vals = eqx.combine(new_vals, jax.tree_util.tree_map_with_path(label_fn, model))

    old_vals = jax.tree.map(
        lambda old, new: old if new is None else None, model, new_vals
    )
    return eqx.combine(old_vals, new_vals)


def tree_set_with_path(model, set_dict):
    return tree_apply_with_path(
        model, {k: (lambda v: lambda _: v)(v) for k, v in set_dict.items()}
    )
