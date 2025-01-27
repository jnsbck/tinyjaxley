from .base import Module
import jax.numpy as jnp
from math import pi
from ..utils import find
import jax

class Compartment(Module):
    def __init__(self):
        super().__init__(None)
        self._recordings = []
        self._externals = {}
        self._channels = {}
        self._params.update({"r": 1.0, "l": 10.0, "c": 1.0, "Ra": 5000.0})
        self._states.update({"v": -70.0, "i": 0.0})
        self._xyzr = jnp.array([[0.0, 0.0, 0.0, self.params["r"]]])

    @property
    def param_states(self):
        leaves = {}
        for params_or_states in [self.params, self.states]:
            for k, v in params_or_states.items():
                if (k in leaves) & isinstance(v, dict):
                    leaves[k].update(v)
                else:
                    leaves[k] = v
        return leaves

    @property
    def name(self):
        return "comp"
    
    def __repr__(self, indent = ""):
        repr_str = f"{indent}{self.name}"
        repr_str += f"({", ".join(self.channels)})" if self.channels else ""
        return repr_str
    
    @staticmethod
    def i(t, u, p):
        area = 2.0 * pi * p["r"] * p["l"] # μm²
        return u["i"] / area * 1e5  # nA/μm² -> μA/cm²
    
    def vf(self, t, u, p):
        for key, ext in self.externals.items():
            u[key] =  sum(f(t, u, p) for f in ext)

        du = {}
        for cname, channel in self.channels.items():
            du[cname] = channel.vf(t, u[cname], p[cname], u["v"])
            
        i_channels = find(["i"], du, 0.0)
        i_channels = sum(jax.tree.flatten(i_channels)[0]) * 1000.0 # mA/cm² -> μA/cm².
        du["i"] = self.i(t, u, p)
        du["v"] = (du["i"] - i_channels) / p["c"]

        return du
    
    def init(self, t = 0, u = None, p = None):
        u = self.states if u is None else u
        p = self.params if p is None else p
        for key, ext in self._externals.items():
            u[key] =  sum(f(t, u, p) for f in ext)
            self.states = {key: u[key]}
        for cname, channel in self._channels.items():
            u[cname] = channel.init(t, u[cname], p[cname], u["v"])
            self.states = {cname: u[cname]}
        return u
