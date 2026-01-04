import jax
import jax.numpy as jnp
import equinox as eqx

from . import Module, Comp

from ..utils import stack_leaves


class Branch(Module):
    def __init__(self, comps: list[Comp]):
        comps = jax.tree.map(stack_leaves, *comps)
        comps = eqx.tree_at(lambda x: x.index, comps, jnp.arange(comps.num_comps))
        comps = eqx.tree_at(lambda x: x.parents, comps, jnp.arange(comps.num_comps) - 1)
        super().__init__(
            l=comps.l,
            r=comps.r,
            c=comps.c,
            ra=comps.ra,
            x=comps.x,
            y=comps.y,
            z=comps.z,
            parents=comps.parents,
            index=comps.index,
            id=comps.id,
        )
