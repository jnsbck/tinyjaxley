from .channel import Channel, _vtrap, taux, xinf
from ..utils import safe_exp

class Leak(Channel):
    def __init__(self): super().__init__(params = {"g": 0.0003, "e": -54.3}, states = {})
    def i(self, u, p, t): return p["g"] * (u["v"] - p["e"])
    def du(self, u, p, t): return {}
    def init(self, u, p): return {}

class Na(Channel):
    def __init__(self): super().__init__(params = {"g": 0.12, "e": 50.0}, states = {"m": 0.0, "h": 0.0})
    def i(self, u, p, t): return p["g"] * u["m"]**3 * u["h"] * (u["v"] - p["e"])
    def du(self, u, p, t): return {"m": self.dm(u, p, t), "h": self.dh(u, p, t)}
    def dm(self, u, p, t): return self.a_m(u["v"]) * (1 - u["m"]) - self.b_m(u["v"]) * u["m"]
    def dh(self, u, p, t): return self.a_h(u["v"]) * (1 - u["h"]) - self.b_h(u["v"]) * u["h"]
    def a_m(self, v): return 0.1 * _vtrap(-(v + 40), 10)
    def b_m(self, v): return 4.0 * safe_exp(-(v + 65) / 18)
    def a_h(self, v): return 0.07 * safe_exp(-(v + 65) / 20)
    def b_h(self, v): return 1.0 / (safe_exp(-(v + 35) / 10) + 1)
    def m_inf(self, v): return xinf(v, self.a_m, self.b_m)
    def h_inf(self, v): return xinf(v, self.a_h, self.b_h)
    def tau_m(self, v): return taux(v, self.a_m, self.b_m)
    def tau_h(self, v): return taux(v, self.a_h, self.b_h)
    def init(self, u, p): return {"m": self.m_inf(u["v"]), "h": self.h_inf(u["v"])}

class K(Channel):
    def __init__(self): super().__init__(params = {"g": 0.036, "e": -77.0}, states = {"n": 0.0})
    def i(self, u, p, t): return p["g"] * u["n"]**4 * (u["v"] - p["e"])
    def du(self, u, p, t): return {"n": self.dn(u, p, t)}
    def dn(self, u, p, t): return self.a_n(u["v"]) * (1 - u["n"]) - self.b_n(u["v"]) * u["n"]
    def a_n(self, v): return 0.01 * _vtrap(-(v + 55), 10)
    def b_n(self, v): return 0.125 * safe_exp(-(v + 65) / 80)
    def n_inf(self, v): return xinf(v, self.a_n, self.b_n)
    def tau_n(self, v): return taux(v, self.a_n, self.b_n)
    def init(self, u, p): return {"n": self.n_inf(u["v"])}