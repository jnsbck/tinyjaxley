from . import Module, Branch
import jax
import jax.numpy as jnp
import equinox as eqx

from jax import Array


class Cell(Module):
    def __init__(self, branches: list[Branch], parents: Array):
        stack_leaves = (
            lambda *lvs: jnp.concatenate(lvs) if eqx.is_array(lvs[0]) else lvs[0]
        )
        comp_parents = self._combine_parents(branches, parents)

        comps = jax.tree.map(stack_leaves, *branches)
        comps = eqx.tree_at(lambda x: x.index, comps, jnp.arange(comps.l.size))
        comps = eqx.tree_at(lambda x: x.parents, comps, comp_parents)
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

    def _combine_parents(self, branches: list[Branch], parents: Array):
        cumsum_leading_zero = lambda x: jnp.concatenate([jnp.array([0]), jnp.cumsum(x)])

        branch_sizes = jnp.array([b.parents.size for b in branches])
        local_parents = jnp.concatenate([b.parents for b in branches])

        branch_offsets = cumsum_leading_zero(branch_sizes[:-1])
        comp_parents = local_parents + jnp.repeat(branch_offsets, branch_sizes)

        parent_last_comps = branch_offsets + branch_sizes - 1
        first_comp_parents = jnp.where(
            parents != -1, parent_last_comps[parents], comp_parents[branch_offsets]
        )

        return comp_parents.at[branch_offsets].set(first_comp_parents)
