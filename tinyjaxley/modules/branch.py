import jax
import jax.numpy as jnp
import equinox as eqx

from . import Module, Comp

from jax import Array


class Branch(Module):
    def __init__(self, comps: list[Comp]):
        stack_leaves = lambda *lvs: jnp.stack(lvs) if eqx.is_array(lvs[0]) else lvs[0]
        comps = jax.tree.map(stack_leaves, *comps)
        comps = eqx.tree_at(lambda x: x.index, comps, jnp.arange(comps.l.size))
        comps = eqx.tree_at(lambda x: x.parents, comps, jnp.arange(len(comps.l)) - 1)
        super().__init__(
            l=comps.l,
            r=comps.r,
            c=comps.c,
            ra=comps.ra,
            xyz=comps.xyz,
            parents=comps.parents,
            index=comps.index,
            id=comps.id,
        )
