from .base import Module
import jax.numpy as jnp
import jax
from ..mechanisms.channel import Channel
from ..mechanisms.external import Stimulus

from jax import Array


class Comp(Module):
    def __init__(
        self,
        l: Array = 10.0,
        r: Array = 1.0,
        c: Array = 1.0,
        xyz: Array = jnp.array([0.0, 0.0, 0.0]),
        index: Array = jnp.array(0),
        id: Array = jnp.array(0),
        key: str = None,
    ):
        super().__init__(l=l, r=r, c=c, xyz=xyz, index=index, id=id, key=key)

    def __call__(self, t, u, args=None):
        is_instance = lambda cls: lambda x: isinstance(x, cls)

        def i_dist(c):
            u_ = u.get(c.name, {})
            return c.i(t, u_, u["v"])

        # TODO: Add clamping
        i_int = jax.tree.map(i_dist, self.channels, is_leaf=is_instance(Channel))
        i_ext = jax.tree.map(i_dist, self.stimuli, is_leaf=is_instance(Stimulus))
        # i_clamp = jax.tree.map(i_dist, self.clamps, is_leaf=is_instance(Clamp))

        i_ext_total = jax.tree.reduce(lambda x, y: x + y, i_ext) if i_ext else 0.0
        i_int_total = jax.tree.reduce(lambda x, y: x + y, i_int) if i_int else 0.0

        du = jax.tree.map(
            lambda c: c(t, u[c.name], u["v"]),
            self.channels,
            is_leaf=is_instance(Channel),
        )
        du["v"] = (i_ext_total * 1e5 / self.area - i_int_total * 1e3) / self.c
        return du

    def _edges_init(self):
        return jnp.array([]).reshape(0, 2)
