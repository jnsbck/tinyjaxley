import equinox as eqx
from jax import Array
import jax.numpy as jnp
import jax

from operator import attrgetter
from typing import Optional


class Mechanism(eqx.Module):
    name: str
    index: Array = eqx.field(converter=jnp.array)
    reads: tuple[str] = eqx.field(converter=tuple)
    writes: tuple[str] = eqx.field(converter=tuple)
    _is_density: bool = eqx.field(default=False)

    def __init__(self, name: str = None, index: Array = None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.index = index if index is not None else jnp.array(0)

    def i(self, t, u, args=None):
        return 0.0

    def init(self, t, u, args=None):
        return ()

    def _is_global(self, v):
        # any state that is in more than 1 mechs -> global
        return True if len(v) > len(self.index) else False
    
    def read(self, u):
        u_local = jnp.zeros((len(self.index), len(self.reads)))
        for i, k in enumerate(self.reads):
            u_k = u[k]
            u_k_at = u_k[self.index] if self._is_global(u_k) else u_k
            u_local = u_local.at[self.index, i].set(u_k_at)
        return u_local

    def write(self, u, u_local):
        _u = u.copy()
        for i, k in enumerate(self.writes):
            u_k = u[k]
            u_k_at = u_k.at[self.index] if self._is_global(u_k) else u_k.at[:]
            _u[k] = u_k_at.set(u_local[i])
        return _u

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
        self.index = jnp.array([])

    def __repr__(self):
        return f"{self._mechanism.__class__.__name__}@{self.index}"

    def __getitem__(self, index: Array):
        self_at = eqx.tree_at(lambda x: x.index, self, index)
        return self_at

    def set(self, path_str: str, value: Array):
        # TODO: should attrgetter be replaced by tree_at_path?
        getter = attrgetter(path_str.lstrip("."))
        replace_fn = lambda x: x.at[self.index].set(value)
        return eqx.tree_at(getter, self._mechanism, replace_fn=replace_fn)

    def get(self, path_str: Optional[str] = None):
        if path_str is not None:
            getter = attrgetter(path_str.lstrip("."))
            return getter(self._mechanism).at[self.index].get()
        return jax.tree.map(
            lambda x: x.at[self.index].get() if eqx.is_array(x) else x, self._mechanism
        )
