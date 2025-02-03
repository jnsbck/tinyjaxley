from math import pi
from typing import Callable, List

import jax
import jax.numpy as jnp

from ..utils import find
from .base import Module


class Compartment(Module):
    recordings: List[str]
    externals: dict[str, List[Callable]]
    xyzr: jnp.ndarray

    def __init__(self, channels = [], u = {"v": -70.0, "i": 0.0}, p = {"r": 1.0, "l": 10.0, "c": 1.0, "Ra": 5000.0}):
        super().__init__(None, channels, u=u, p=p)
        self.recordings = []
        self.externals = {}
        self.xyzr = jnp.array([[0.0, 0.0, 0.0, self.p["r"]]])
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
    
    def vf(self, t, u, args = None):
        # TODO: make fast --> see https://docs.kidger.site/equinox/tricks/
        # TODO: currently "i" is garbage since treated as differentiable.
        u_comp, u_c = u

        for key, ext in self.externals.items():
            u_comp[key] = sum(f(t, u_comp) for f in ext)

        nan = jnp.array([float("nan")])
        du_comp = {"i": nan} # ignore i for now

        du_c, i_c, v = [], 0.0, u_comp["v"]
        for c in self.channels:
            du_c += [{**c.vf(t, u_c[c.index], v), "i": nan}]
            i_c += c.i(t, u_c[c.index], v)
        du_comp["v"] = (self.i(t, u_comp) - i_c * 1000.0) / self.p["c"] # mA/cm² -> μA/cm².
        return [du_comp, du_c]
