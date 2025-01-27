from .channel import Channel
from ..utils import safe_exp, _vtrap, taux, xinf

def α_m(v): return 0.1 * _vtrap(-(v + 40), 10)
def β_m(v): return 4.0 * safe_exp(-(v + 65) / 18)
def α_h(v): return 0.07 * safe_exp(-(v + 65) / 20)
def β_h(v): return 1.0 / (safe_exp(-(v + 35) / 10) + 1)
def α_n(v): return 0.01 * _vtrap(-(v + 55), 10)
def β_n(v): return 0.125 * safe_exp(-(v + 65) / 80)

def dm(t, u, p, v): return α_m(v) * (1 - u["m"]) - β_m(v) * u["m"]
def dh(t, u, p, v): return α_h(v) * (1 - u["h"]) - β_h(v) * u["h"]
def dn(t, u, p, v): return α_n(v) * (1 - u["n"]) - β_n(v) * u["n"]

def i_leak(t, u, p, v): return p["g"] * (v - p["e"])
def i_na(t, u, p, v): return p["g"] * u["m"]**3 * u["h"] * (v - p["e"])
def i_k(t, u, p, v): return p["g"] * u["n"]**4 * (v - p["e"])

class Leak(Channel):
    def __init__(self): 
        super().__init__(params = {"g": 0.0003, "e": -54.3}, states = {})
    
    @staticmethod
    def vf(t, u, p, v): 
        return {"i": i_leak(t, u, p, v)}
    
    @staticmethod
    def α(v): return  {}

    @staticmethod
    def β(v): return {}

    @staticmethod
    def init(t, u, p, v):
        return {"i": i_leak(t, u, p, v)}

class Na(Channel):
    def __init__(self): 
        super().__init__(params = {"g": 0.12, "e": 50.0}, states = {"m": 0.2, "h": 0.2})
    
    @staticmethod
    def vf(t, u, p, v): 
        return {"i": i_na(t, u, p, v), "m": dm(t, u, p, v), "h": dh(t, u, p, v)}
    
    @staticmethod
    def α(v):
        return {"m": α_m(v), "h": α_h(v)}
    
    @staticmethod
    def β(v):
        return {"m": β_m(v), "h": β_h(v)}
    
    @staticmethod
    def init(t, u, p, v): 
        return {"i": i_na(t, u, p, v), "m": xinf(v, α_m, β_m), "h": xinf(v, α_h, β_h)}

class K(Channel):
    def __init__(self): 
        super().__init__(params = {"g": 0.036, "e": -77.0}, states = {"n": 0.2})
    
    @staticmethod
    def vf(t, u, p, v): 
        return {"i": i_k(t, u, p, v), "n": dn(t, u, p, v)}
    
    @staticmethod
    def α(v):
        return {"n": α_n(v)}
    
    @staticmethod
    def β(v):
        return {"n": β_n(v)}
    
    @staticmethod
    def init(t, u, p, v): 
        return {"i": i_k(t, u, p, v), "n": xinf(v, α_n, β_n)}