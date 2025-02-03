from ..utils import _vtrap, safe_exp, taux, xinf
from .channel import Channel


def α_m(v): return 0.1 * _vtrap(-(v + 40), 10)
def β_m(v): return 4.0 * safe_exp(-(v + 65) / 18)
def α_h(v): return 0.07 * safe_exp(-(v + 65) / 20)
def β_h(v): return 1.0 / (safe_exp(-(v + 35) / 10) + 1)
def α_n(v): return 0.01 * _vtrap(-(v + 55), 10)
def β_n(v): return 0.125 * safe_exp(-(v + 65) / 80)

def dm(t, u, p, v): return α_m(v) * (1 - u["m"]) - β_m(v) * u["m"]
def dh(t, u, p, v): return α_h(v) * (1 - u["h"]) - β_h(v) * u["h"]
def dn(t, u, p, v): return α_n(v) * (1 - u["n"]) - β_n(v) * u["n"]

class Leak(Channel):
    def __init__(self): 
        super().__init__(p = {"g": 0.0003, "e": -54.3}, u = {})
    
    def vf(self, t, u, v): return {}
    
    def i(self, t, u, v): return self.p["g"] * (v - self.p["e"])
    
    def α(self, v): return  {}

    def β(self, v): return {}

    def init(self, t, u, v): return {"i": self.i(t, u, v)}

class Na(Channel):
    def __init__(self): 
        super().__init__(p = {"g": 0.12, "e": 50.0}, u = {"m": 0.2, "h": 0.2})
    
    def vf(self, t, u, v): 
        return {"m": dm(t, u, self.p, v), "h": dh(t, u, self.p, v)}
    
    def i(self, t, u, v): return self.p["g"] * u["m"]**3 * u["h"] * (v - self.p["e"])
    
    def α(self, v):
        return {"m": α_m(v), "h": α_h(v)}
    
    def β(self, v):
        return {"m": β_m(v), "h": β_h(v)}
    
    def init(self, t, u, v): 
        return {"i": self.i(t, u, v), "m": xinf(v, α_m, β_m), "h": xinf(v, α_h, β_h)}

class K(Channel):
    def __init__(self): 
        super().__init__(p = {"g": 0.036, "e": -77.0}, u = {"n": 0.2})
    
    def vf(self, t, u, v): return {"n": dn(t, u, self.p, v)}

    def i(self, t, u, v): return self.p["g"] * u["n"]**4 * (v - self.p["e"])
    
    def α(self, v):
        return {"n": α_n(v)}
    
    def β(self, v):
        return {"n": β_n(v)}
    
    def init(self, t, u, v): return {"i": self.i(t, u, v), "n": xinf(v, α_n, β_n)}