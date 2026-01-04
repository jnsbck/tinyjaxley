from abc import abstractmethod
import equinox as eqx
from ..utils import _vtrap, safe_exp
from jax import Array
import jax.numpy as jnp

from .mechanism import Mechanism


class Channel(Mechanism):
    gbar: Array = None
    e: Array = None

    def __init__(self, name: str = None, index: Array = None):
        super().__init__(name, index)

    def i(self, t, u, v):
        return self.g(u) * (v - self.e)

    @abstractmethod
    def __call__(self, t, u, v):
        return u

    def tau(self, u, v):
        return {}

    def xinf(self, u, v):
        return {}

    @abstractmethod
    def g(self, u):
        return 0.0

    def init(self, t, u, v):
        return {}


a_m = lambda v: 0.1 * _vtrap(-(v + 40), 10)
b_m = lambda v: 4.0 * safe_exp(-(v + 65) / 18)
a_h = lambda v: 0.07 * safe_exp(-(v + 65) / 20)
b_h = lambda v: 1.0 / (safe_exp(-(v + 35) / 10) + 1)
a_n = lambda v: 0.01 * _vtrap(-(v + 55), 10)
b_n = lambda v: 0.125 * safe_exp(-(v + 65) / 80)


class Leak(Channel):
    gbar: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)

    def __init__(self, gbar: Array = 0.0003, e: Array = -54.3):
        super().__init__()
        self.gbar = gbar
        self.e = e

    def g(self, u):
        return self.gbar

    def __call__(self, t, u, v):
        return {}

    def init(self, t, u, v):
        return {}


class Na(Channel):
    gbar: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)

    def __init__(self, gbar: Array = 0.12, e: Array = 50.0):
        super().__init__()
        self.gbar = gbar
        self.e = e

    def g(self, u):
        m = u["m"]
        h = u["h"]
        return self.gbar * m**3 * h

    def tau(self, u, v):
        tau_m = 1 / (a_m(v) + b_m(v))
        tau_h = 1 / (a_h(v) + b_h(v))
        return {"m": tau_m, "h": tau_h}

    def xinf(self, u, v):
        tau = self.tau(u, v)
        m_inf = a_m(v) * tau["m"]
        h_inf = a_h(v) * tau["h"]
        return {"m": m_inf, "h": h_inf}

    def __call__(self, t, u, v):
        m = u["m"]
        h = u["h"]

        xinf = self.xinf(u, v)
        tau = self.tau(u, v)
        dm = -(m - xinf["m"]) / tau["m"]
        dh = -(h - xinf["h"]) / tau["h"]
        return {"m": dm, "h": dh}

    def init(self, t, u, v):
        return self.xinf(u, v)


class K(Channel):
    gbar: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)

    def __init__(self, gbar: Array = 0.036, e: Array = -77.0):
        super().__init__()
        self.gbar = gbar
        self.e = e

    def g(self, u):
        n = u["n"]
        return self.gbar * n**4

    def tau(self, u, v):
        tau_n = 1 / (a_n(v) + b_n(v))
        return {"n": tau_n}

    def xinf(self, u, v):
        tau = self.tau(u, v)
        n_inf = a_n(v) * tau["n"]
        return {"n": n_inf}

    def __call__(self, t, u, v):
        n = u["n"]
        xinf = self.xinf(u, v)
        tau = self.tau(u, v)
        dn = -(n - xinf["n"]) / tau["n"]
        return {"n": dn}

    def init(self, t, u, v):
        return self.xinf(u, v)
