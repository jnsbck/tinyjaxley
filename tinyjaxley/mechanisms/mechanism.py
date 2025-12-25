import equinox as eqx
from jax import Array
import jax.numpy as jnp
import jax


class Mechanism(eqx.Module):
    name: str
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, name: str = None, index: Array = None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.index = index if index is not None else jnp.array(0)

    def __call__(self, t, u, v):
        return 0.0

    def i(self, t, u, v):
        return 0.0

    @property
    def at(self):
        return MechanismIndexer(self)


class MechanismIndexer(eqx.Module):
    # TODO: Make jit-able
    # TODO: use map(array.at[]) for indexing
    _mechanism: Mechanism
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, mechanism: Mechanism):
        self._mechanism = mechanism
        self.index = mechanism.index

    def __repr__(self):
        return f"{self._mechanism.__class__.__name__}@{self.index}"

    def __getitem__(self, index: Array):
        self_at = eqx.tree_at(lambda x: x.index, self, index)
        return self_at

    def set(self, set_dict: dict):
        pass

    def get(self):
        mech_idx = jnp.atleast_1d(self._mechanism.index)
        index = jnp.atleast_1d(self.index)
        pos_mask = jnp.isin(mech_idx, index)
        local_index = jnp.where(pos_mask)[0]

        def filter_param(x):
            if isinstance(x, Array):
                return x[local_index]
            return x

        return jax.tree.map(filter_param, self._mechanism)
