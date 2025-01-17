from jax import numpy as jnp
from .channel import Channel

class Leak(Channel):
    def __init__(self): super().__init__(params = {"g": 0.1, "e": -70}, states = {})
    def i(self, u, p, t): return p["g"] * (u["v"] - p["e"])
    def du(self, u, p, t): return {}
    def init(self, u, p): return {}

class Na(Channel):
    def __init__(self): super().__init__(params = {"g": 0.1, "e": 50}, states = {"m": 0, "h": 0})
    def i(self, u, p, t): return p["g"] * u["m"]**3 * u["h"] * (u["v"] - p["e"])
    def du(self, u, p, t): return {"m": self.dm(u, p, t), "h": self.dh(u, p, t)}
    def dm(self, u, p, t): return self.a_m(u["v"]) * (1 - u["m"]) - self.b_m(u["v"]) * u["m"]
    def dh(self, u, p, t): return self.a_h(u["v"]) * (1 - u["h"]) - self.b_h(u["v"]) * u["h"]
    def a_m(self, v): return 0.1 * (v + 40) / (1 - jnp.exp(-(v + 40) / 10))
    def b_m(self, v): return 4 * jnp.exp(-(v + 65) / 18)
    def a_h(self, v): return 0.07 * jnp.exp(-(v + 65) / 20)
    def b_h(self, v): return 1 / (1 + jnp.exp(-(v + 35) / 10))
    def init(self, u, p): return {"m": self.a_m(u["v"]) / (self.a_m(u["v"]) + self.b_m(u["v"])), "h": self.a_h(u["v"]) / (self.a_h(u["v"]) + self.b_h(u["v"]))}

class K(Channel):
    def __init__(self): super().__init__(params = {"g": 0.1, "e": -70}, states = {"n": 0})
    def i(self, u, p, t): return p["g"] * u["n"]**4 * (u["v"] - p["e"])
    def du(self, u, p, t): return {"n": self.dn(u, p, t)}
    def dn(self, u, p, t): return self.a_n(u["v"]) * (1 - u["n"]) - self.b_n(u["v"]) * u["n"]
    def a_n(self, v): return 0.01 * (v + 55) / (1 - jnp.exp(-(v + 55) / 10))
    def b_n(self, v): return 0.125 * jnp.exp(-(v + 65) / 80)
    def init(self, u, p): return {"n": self.a_n(u["v"]) / (self.a_n(u["v"]) + self.b_n(u["v"]))}