from .base import Module
import jax.numpy as jnp
from math import pi
from ..utils import find
import jax

class Compartment(Module):
    def __init__(self):
        super().__init__([])
        self._recordings = []
        self._externals = {}
        self._channels = {}
        self._params.update({"r": 1.0, "l": 10.0, "c": 1.0, "Ra": 5000.0})
        self._states.update({"v": -70.0, "i": 0.0})
        self._xyzr = jnp.array([[0.0, 0.0, 0.0, self.get(("comp", "r"))]])
        self.groups = {}

    @property
    def name(self):
        return "comp"
    
    @property
    def channels(self):
        return self.submodules
            
    def get(self, ref):
        ref = ref if isinstance(ref, tuple) else (ref,)
        keys = [self.name] + [c.name for c in self.submodules]
        
        data = self.params_states
        for key in ref:
            idx = keys.index(key) if key in keys else key
            data = data[idx]
        return data

    def set(self, ref, value):
        ref = ref if isinstance(ref, tuple) else (ref,)
        keys = [self.name] + [c.name for c in self.submodules]
        
        data = self.params_states
        for key in ref[:-1]:
            idx = keys.index(key) if key in keys else key
            data = data[idx]
        
        idx = keys.index(ref[-1]) if ref[-1] in keys else ref[-1]
        data[idx] = value
    
    @staticmethod
    def i(t, u, p):
        area = 2.0 * pi * p["r"] * p["l"] # μm²
        return u["i"] / area * 1e5  # nA/μm² -> μA/cm²
    
    def vf(self, t, u, p):
        # for key, ext in self.externals.items():
        #     u[key] =  sum(f(t, u, p) for f in ext)

        du = [{"i": self.i(t, u[0], p[0])}]
        for i, channel in enumerate(self.channels):
            du += [channel.vf(t, u[i+1], p[i+1], u[0]["v"])]
            
        i_channels = find(["i"], du[1:], 0.0)
        i_channels = sum(jax.tree.flatten(i_channels)[0]) * 1000.0 # mA/cm² -> μA/cm².
        du[0]["v"] = (du[0]["i"] - i_channels) / p[0]["c"]
        return du
    
    # def init(self, t = 0, u = None, p = None):
    #     u = self.states if u is None else u
    #     p = self.params if p is None else p
    #     for key, ext in self._externals.items():
    #         u[key] =  sum(f(t, u, p) for f in ext)
    #         self.states = {key: u[key]}
    #     for cname, channel in self._channels.items():
    #         u[cname] = channel.init(t, u[cname], p[cname], u["v"])
    #         self.states = {cname: u[cname]}
    #     return u
