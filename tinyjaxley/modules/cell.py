from .base import Module
import jax.numpy as jnp
from .utils import g_ij

class Cell(Module):
    def __init__(self, branches = None, parents = None):
        super().__init__(branches)
        self.parents = parents
        N = len(self.submodules)
        
        self.k_ij = []
        for i, parent in enumerate(self.parents):
            if parent >= 0:  # Skip root (-1)
                self.k_ij += [(i, parent), (parent, i)]

    # def vf(self, t, u, p):
    #     du = [comp.vf(t, u[i], p[i]) for i, comp in enumerate(self.comps)]
    #     for i,j in self.k_ij:
    #         du[i]["v"] += g_ij(p[i], p[j]) * (u[j]["v"] - u[i]["v"])
    #         return du