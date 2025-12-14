from abc import abstractmethod
import equinox as eqx
from ..utils import _vtrap, safe_exp
from jax import Array
import jax.numpy as jnp


class Channel(eqx.Module):
    name: str
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, name: str = None, index: Array = None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.index = index if index is not None else jnp.array(0)
        super().__init__()

    @abstractmethod
    def i(self, t, u, v):
        return 0.0

    @abstractmethod
    def __call__(self, t, u, v):
        return u

    def tau(self, u, v):
        return {}

    def xinf(self, u, v):
        return {}

    def init(self, t, u, v):
        return {}


def a_m(v):
    return 0.1 * _vtrap(-(v + 40), 10)


def b_m(v):
    return 4.0 * safe_exp(-(v + 65) / 18)


def a_h(v):
    return 0.07 * safe_exp(-(v + 65) / 20)


def b_h(v):
    return 1.0 / (safe_exp(-(v + 35) / 10) + 1)


def a_n(v):
    return 0.01 * _vtrap(-(v + 55), 10)


def b_n(v):
    return 0.125 * safe_exp(-(v + 65) / 80)


class Leak(Channel):
    g: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)

    def __init__(self, g: Array = 0.0003, e: Array = -54.3):
        super().__init__()
        self.g = g
        self.e = e

    def i(self, t, u, v):
        return self.g * (v - self.e)

    def __call__(self, t, u, v):
        return {}

    def init(self, t, u, v):
        return {}


class Na(Channel):
    g: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)

    def __init__(self, g: Array = 0.12, e: Array = 50.0):
        super().__init__()
        self.g = g
        self.e = e

    def i(self, t, u, v):
        m = u["m"]
        h = u["h"]
        return self.g * m**3 * h * (v - self.e)

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
    g: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)

    def __init__(self, g: Array = 0.036, e: Array = -77.0):
        super().__init__()
        self.g = g
        self.e = e

    def i(self, t, u, v):
        n = u["n"]
        return self.g * n**4 * (v - self.e)

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
