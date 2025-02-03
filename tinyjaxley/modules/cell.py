import jax.numpy as jnp

from ..tree import Module
from ..utils import g_ij
from jax import Array


class Cell(Module):
    G: dict[tuple[int, int], Array]

    def __init__(self, branches = None, branch_tree = None):
        super().__init__(None, branches)
        self.G = {}

    @property
    def branches(self):
        return self.children

    # def vf(self, t, u, p):
    #     du = [comp.vf(t, u[i], p[i]) for i, comp in enumerate(self.comps)]
    #     for i,j in self.k_ij:
    #         du[i]["v"] += g_ij(p[i], p[j]) * (u[j]["v"] - u[i]["v"])
    #         return du