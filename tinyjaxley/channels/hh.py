from .channel import Channel, _vtrap, taux, xinf
from ..utils import safe_exp

class Leak(Channel):
    def __init__(self): super().__init__(params = {"g": 0.0003, "e": -54.3}, states = {})
    def __call__(self, t, u, args): return {"i": self.i(t, u, *args)}
    def i(self, t, u, p, v): return p["g"] * (v - p["e"])
    def init(self, t, u, p, v): return {"i": self.i(t, u, p, v)}

class Na(Channel):
    def __init__(self): super().__init__(params = {"g": 0.12, "e": 50.0}, states = {"m": 0.2, "h": 0.2})
    def __call__(self, t, u, args): return {"i": self.i(t, u, *args), "m": self.dm(t, u, *args), "h": self.dh(t, u, *args)}
    def i(self, t, u, p, v): return p["g"] * u["m"]**3 * u["h"] * (v - p["e"])
    def dm(self, t, u, p, v): return self.α_m(v) * (1 - u["m"]) - self.β_m(v) * u["m"]
    def α_m(self, v): return 0.1 * _vtrap(-(v + 40), 10)
    def β_m(self, v): return 4.0 * safe_exp(-(v + 65) / 18)
    def dh(self, t, u, p, v): return self.α_h(v) * (1 - u["h"]) - self.β_h(v) * u["h"]
    def α_h(self, v): return 0.07 * safe_exp(-(v + 65) / 20)
    def β_h(self, v): return 1.0 / (safe_exp(-(v + 35) / 10) + 1)
    def init(self, t, u, p, v): return {"i": self.i(t, u, p, v), "m": xinf(v, self.α_m, self.β_m), "h": xinf(v, self.α_h, self.β_h)}

class K(Channel):
    def __init__(self): super().__init__(params = {"g": 0.036, "e": -77.0}, states = {"n": 0.2})
    def __call__(self, t, u, args): return {"i": self.i(t, u, *args), "n": self.dn(t, u, *args)}
    def i(self, t, u, p, v): return p["g"] * u["n"]**4 * (v - p["e"])
    def dn(self, t, u, p, v): return self.α_n(v) * (1 - u["n"]) - self.β_n(v) * u["n"]
    def α_n(self, v): return 0.01 * _vtrap(-(v + 55), 10)
    def β_n(self, v): return 0.125 * safe_exp(-(v + 65) / 80)
    def init(self, t, u, p, v): return {"i": self.i(t, u, p, v), "n": xinf(v, self.α_n, self.β_n)}