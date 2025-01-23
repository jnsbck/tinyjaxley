from .base import Module
import jax.numpy as jnp
from math import pi

class Compartment(Module):
    def __init__(self):
        super().__init__(None)
        self._recordings = []
        self._externals = {}
        self._channels = {}
        self._params.update({"r": 1.0, "l": 10.0, "c": 1.0})
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
        repr_str = f"{indent}{self.name}[{self.index}]"
        repr_str += f"({", ".join(self.channels)})" if self.channels else ""
        return repr_str
    
    def du(self, t, u, p):
        area = 2.0 * pi * p["r"] * p["l"] # μm²
        i = u["i"] / area * 1e5  # nA/μm² -> μA/cm²

        du = {"i": i}
        i_c = 0.0
        for cname in self._channels:
            i_c += u[cname]["i"] * 1000.0 # mA/cm^2 -> μA/cm^2.
        du["v"] = (i - i_c) / p["c"]
        return du
    
    def init(self, t, u = None, p = None):
        u = self.states if u is None else u
        p = self.params if p is None else p
        for key, ext in self._externals.items():
            u[key] =  sum(f(t, u) for f in ext)
            self.states = {key: u[key]}
        for cname, channel in self._channels.items():
            u[cname] = channel.init(t, u[cname], p[cname], u["v"])
            self.states = {cname: u[cname]}
        return u
