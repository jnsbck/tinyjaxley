from jax import Array

from ..utils import g_ij
from .base import Module


class Branch(Module):
    G: dict[tuple[int, int], Array]
    
    def __init__(self, compartments):
        super().__init__(None, compartments)
        self.G = {}
        for child_i, child_j in zip(self.children[1:], self.children[:-1]):
            self.G[(child_i.index, child_j.index)] = g_ij(child_i.p, child_j.p)
            self.G[(child_j.index, child_i.index)] = g_ij(child_j.p, child_i.p)

    @property
    def comps(self):
        return self.children

    def vf(self, t, u, args = None):
        # TODO: make fast --> see https://docs.kidger.site/equinox/tricks/
        u_branch, u_c = u

        du_c = [c.vf(t, u_c[c.index]) for c in self.children]
        for (i,j), g_ij in self.G.items():
            du_c[i][0]["v"] += g_ij * (u_c[j][0]["v"] - u_c[i][0]["v"])
        return [{}, du_c]