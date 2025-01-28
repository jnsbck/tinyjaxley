from .base import Module
import jax.numpy as jnp
from math import pi
from ..utils import find
import jax

class Compartment(Module):
    def __init__(self, channels = []):
        super().__init__(None, channels)
        self._recordings = []
        self._externals = {}
        self._channels = {}
        self._params.update({"r": 1.0, "l": 10.0, "c": 1.0, "Ra": 5000.0})
        self._states.update({"v": -70.0, "i": 0.0})
        # self._xyzr = jnp.array([[0.0, 0.0, 0.0, self.get(("comp", "r"))]])
        # self._groups = {}
        self._externals = {}
        self._recordings = []

    @property
    def key(self):
        return "comp"
    
    @property
    def channels(self):
        return self.children
    
    @staticmethod
    def i(t, u, p):
        area = 2.0 * pi * p["r"] * p["l"] # μm²
        return u["i"] / area * 1e5  # nA/μm² -> μA/cm²
    
    def vf(self, t, u, p):
        #TODO: fix this! AND MAKE THIS COMPILE FASTER!
        # for key, ext in self.externals.items():
        #     u[key] =  sum(f(t, u, p) for f in ext)

        p_comp, p_channels = p
        u_comp, u_channels = u
        du_comp = {"i": self.i(t, u_comp, p_comp)}

        def body(vf, u, p): return vf(t, u, p, u_comp["v"])
        channel_vfs = [c.vf for c in self.channels]
        du_channels = jax.tree.map(body, channel_vfs, u_channels, p_channels)
        
        i_channels = find(["i"], du_channels, 0.0)
        i_channels = sum(jax.tree.flatten(i_channels)[0]) * 1000.0 # mA/cm² -> μA/cm².
        du_comp["v"] = (du_comp["i"] - i_channels) / p_comp["c"]
        return [du_comp, du_channels]
