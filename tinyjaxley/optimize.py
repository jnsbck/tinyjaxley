import equinox as eqx
from jax import Array
import jax.numpy as jnp
import jax

from tinyjaxley.modules.base import Module

from tinyjaxley.utils import tree_apply_with_path, tree_filter_by_path


class TrainWrapper(eqx.Module):
    module: Module
    trainables: Module
    train_inds: Module

    def __init__(self, module: Module):
        self.module = module
        self.trainables = jax.tree.map(lambda x: None, module)
        self.train_inds = jax.tree.map(lambda x: None, module)

    def make_trainable(self, train_spec: list[tuple[Array, Array, bool]]):
        """Set how and which parameters are trained.

        Args:
            train_spec: Specifies path of the parameter, at which indices it should be trained
                and wether the indices should share the parameter (param_path, at, share).
        """
        # NOTE: mask is reset here -> all trainables need to be set at once
        init_mask = jax.tree.map(lambda x: -1, self.module)

        def mask_setter(path, at, share):
            path_only = tree_filter_by_path(self.module, path)
            leaf = jax.tree.leaves(path_only)[0]

            def set_true(x):
                param_idx = 0 if at is None else at[0]
                x_size = x.size if isinstance(x, Array) else 1
                at_size = at.size if at is not None else x_size
                param_inds = jnp.arange(at_size) + param_idx
                param_inds = (
                    jnp.full(param_inds.shape, param_idx) if share else param_inds
                )
                if at is None:
                    return (
                        x.at[:].set(param_inds) if isinstance(x, Array) else param_inds
                    )
                else:
                    x = x if isinstance(x, Array) else jnp.full(leaf.shape, -1)
                    return x.at[at].set(param_inds)

            return set_true

        mask = init_mask
        for p, at, share in train_spec:  # loop to allow repeat of same path
            mask = tree_apply_with_path(mask, {p: mask_setter(p, at, share)})

        filter_trainables = lambda x, y: None if jnp.all(x == -1) else y[x != -1]
        trainables = jax.tree.map(filter_trainables, mask, self.module)
        train_inds = jax.tree.map(filter_trainables, mask, mask)

        train_wrapper = eqx.tree_at(
            lambda x: x.trainables, self, trainables, is_leaf=lambda x: x is None
        )
        train_wrapper = eqx.tree_at(
            lambda x: x.train_inds,
            train_wrapper,
            train_inds,
            is_leaf=lambda x: x is None,
        )
        return train_wrapper

    def apply_trainables(self):
        is_trainable = lambda x, vals: eqx.is_array(x) and vals is not None
        apply_fn = (
            lambda x, inds, vals: x.at[inds].set(vals) if is_trainable(x, vals) else x
        )
        return jax.tree.map(apply_fn, self.module, self.train_inds, self.trainables)

    def __call__(self, *args, **kwargs):
        return self.apply_trainables()(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if attr.startswith("__"):
            return super().__getattribute__(attr)

        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.apply_trainables(), attr)
