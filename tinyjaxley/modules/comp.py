from .base import Module
import jax.numpy as jnp
from math import pi

class Compartment(Module):
    def __init__(self):
        super().__init__(None)
        self.recordings = []
        self.externals = {}
        self.channels = {}
        self._params.update({"r": 10, "l": 100, "c": 1.0})
        self._states.update({"v": -70, "t": 0.0})
        self._xyzr = jnp.array([[0, 0, 0, self.params["r"]]])

    @property
    def param_states(self):
        leaves = {}
        for k, v in {**self.params, **self.states}.items():
            if k in self.channels:
                leaves.update({k + "_" + kk: vv for kk, vv in v.items()})
            else:
                leaves[k] = v
        return leaves

    @property
    def name(self):
        return "comp"
    
    @property
    def stimuli(self):
        return self.externals["i"]
    
    def __repr__(self, indent = ""):
        repr_str = f"{indent}{self.name}[{self.index}]"
        repr_str += f"({", ".join(self.channels)})" if self.channels else ""
        return repr_str
    
    def area(self, p): return 2 * pi * p["r"] * p["l"] * 1e-8 # μm² to cm²
    
    def i_ext(self, u, p, t): return (u["i"] * 1e-3) / self.area(p) #nA to μA/cm²
    
    def dV(self, u, p, t):
        i_ext = self.i_ext(u, p, t)

        i_channels = 0
        for name, channel in self.channels.items():
            i_channels += channel.i({**u, **u[name]}, {**p, **p[name]}, t)
        return (i_ext - i_channels) / p["c"]
