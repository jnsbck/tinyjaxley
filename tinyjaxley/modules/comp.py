from .base import Module
from typing import List, Callable
import jax.numpy as jnp
from math import pi
from ..utils import find
import jax

class Compartment(Module):
    recordings: List[str]
    externals: dict[str, List[Callable]]

    def __init__(self, channels = []):
        super().__init__(None, channels)
        self.recordings = []
        self.externals = {}
        self.p.update({"r": 1.0, "l": 10.0, "c": 1.0, "Ra": 5000.0})
        self.u.update({"v": -70.0, "i": 0.0})
        # self._xyzr = jnp.array([[0.0, 0.0, 0.0, self.get(("comp", "r"))]])
        # self._groups = {}

    @property
    def key(self):
        return "comp"
    
    @property
    def channels(self):
        return self.children
    
    def i(self, t, u):
        area = 2.0 * pi * self.p["r"] * self.p["l"] # μm²
        return u["i"] / area * 1e5  # nA/μm² -> μA/cm²
    
    def vf(self, t, u, args):
        u_comp, u_channels = u

        for key, ext in self.externals.items():
            u_comp[key] = sum(f(t, u_comp) for f in ext)

        du_comp = {"i": self.i(t, u_comp)}

        def body(c, u): return c.vf(t, u[c.index], u_comp["v"])
        du_channels = [body(c, u_channels) for c in self.channels]
        i_channels = find(["i"], du_channels, 0.0)
        i_channels = sum(jax.tree.flatten(i_channels)[0]) * 1000.0 # mA/cm² -> μA/cm².
        du_comp["v"] = (du_comp["i"] - i_channels) / self.p["c"]
        return [du_comp, du_channels]
