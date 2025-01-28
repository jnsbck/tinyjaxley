from .base import Module
import jax.numpy as jnp
from ..utils import g_ij
import jax

class Branch(Module):
    def __init__(self, compartments):
        super().__init__(None, compartments)
        N = len(self.children)
        G = jnp.eye(N, k=-1) + jnp.eye(N, k=1)
        self.k_ij = jnp.stack(jnp.where(G)).T.tolist()

    @property
    def comps(self):
        return self.children

    def vf(self, t, u, p):
        u_branch, u_children = u
        p_branch, p_children = p
        def _vf(comp, u_i, p_i): return comp.vf(t, u_i, p_i)
        du = jax.tree_map(_vf, self.children, u_children, p_children)
        
        # TODO: Fix this!
        # for i,j in self.k_ij:
        #     du[i]["v"] += g_ij(p_children[i], p_children[j]) * (u_children[j]["v"] - u_children[i]["v"])
        return [{}, du]