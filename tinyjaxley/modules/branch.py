from .base import Module
import jax.numpy as jnp
from .utils import g_ij
import jax

class Branch(Module):
    def __init__(self, compartments):
        super().__init__(compartments)
        N = len(self.comps)
        G = jnp.eye(N, k=-1) + jnp.eye(N, k=1)
        self.k_ij = jnp.stack(jnp.where(G)).T.tolist()

    def vf(self, t, u, p):
        def _vf(comp, u_i, p_i): return comp.vf(t, u_i, p_i)
        du = jax.tree_map(_vf, self.submodules, u, p)
        
        for i,j in self.k_ij:
            du[i]["v"] += g_ij(p[i], p[j]) * (u[j]["v"] - u[i]["v"])
        return du

    def init(self, t = 0, u = None, p = None):
        u = self.states if u is None else u
        p = self.params if p is None else p
        u0 = []
        for i, comp in enumerate(self.comps):
            u0 += [comp.init(t, u[i], p[i])]
        return u0