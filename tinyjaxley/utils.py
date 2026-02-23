import jax.numpy as jnp
import jax
import equinox as eqx
import re
from jax import Array


def safe_exp(x: Array, max_value: float = 20.0):
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)


def _vtrap(x, y):
    return x / (safe_exp(x / y) - 1.0)


def tree_getter(path_str: str):
    """Update pytree at a string path like "channels['na'].gbar" """
    path_str = path_str.lstrip(".")

    def getter(obj):
        for part in re.split(r"\.(?![^\[]*\])", path_str):
            if "[" in part:
                attr, key = re.match(r"(\w+)\['([^']+)'\]", part).groups()
                obj = getattr(obj, attr)[key]
            else:
                obj = getattr(obj, part)
        return obj

    return getter


# def tree_path_of_leaves(model, filter: str | list[str] = None, invert=False):
#     # TODO: allow partial matches, like .channels -> should show all channels
#     # TODO: prevent .c from matching .channels as well
#     def label_fn(path, value):
#         path_str = "".join([str(p) for p in path])
#         cond = (path_str in filter) if filter else True
#         return path_str if cond != invert else None

#     return jax.tree_util.tree_map_with_path(label_fn, model)


# def tree_filter_by_path(model, filter: str | list[str], invert=False):
#     # TODO: add option for is_leaf, i.e. to allow filtering partial paths, like .channels -> should show all channels
#     def label_fn(path, value):
#         path_str = "".join([str(p) for p in path])
#         return value if (path_str in filter) != invert else None

#     return jax.tree_util.tree_map_with_path(label_fn, model)


# def tree_apply_with_path(model, apply_dict):
#     def label_fn(path, value):
#         path_str = "".join([str(p) for p in path])
#         return apply_dict.get(path_str, lambda x: None)(value)

#     new_vals = jax.tree.map(lambda x: None, model)
#     new_vals = eqx.combine(new_vals, jax.tree_util.tree_map_with_path(label_fn, model))

#     old_vals = jax.tree.map(
#         lambda old, new: old if new is None else None, model, new_vals
#     )
#     return eqx.combine(old_vals, new_vals)


# def tree_set_with_path(model, set_dict, at=None):
#     return tree_apply_with_path(
#         model, {k: (lambda _v: lambda _: _v)(v if at is None else v.at[at]) for k, v in set_dict.items()}
#     )


def stack_leaves(*lvs, if_not_stackable=lambda x: x[0]):
    return jnp.hstack(lvs) if eqx.is_array(lvs[0]) else if_not_stackable(lvs)


is_instance_of = lambda cls: lambda x: isinstance(x, cls)
