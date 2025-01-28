from ..tree import Module
import jax.numpy as jnp
from ..utils import g_ij

class Cell(Module):
    def __init__(self, branches = None, branch_tree = None):
        super().__init__(None, branches)
        self.branch_tree = branch_tree
        N = len(self.children)
        
        self.k_ij = []
        for i, parent in enumerate(self.branch_tree):
            if parent >= 0:  # Skip root (-1)
                self.k_ij += [(i, parent), (parent, i)]

    @property
    def branches(self):
        return self.children

    # def vf(self, t, u, p):
    #     du = [comp.vf(t, u[i], p[i]) for i, comp in enumerate(self.comps)]
    #     for i,j in self.k_ij:
    #         du[i]["v"] += g_ij(p[i], p[j]) * (u[j]["v"] - u[i]["v"])
    #         return du