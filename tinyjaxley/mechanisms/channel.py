from abc import abstractmethod
import equinox as eqx
from ..utils import _vtrap, safe_exp
from jax import Array
import jax.numpy as jnp

from .mechanism import Mechanism


class Channel(Mechanism):
    reads: tuple[str] = ("v",) # NOTE: Allow reads from currents?
    writes: tuple[str] = () # NOTE: Allow writes to currents?
    gbar: Array = eqx.field(converter=jnp.array)
    e: Array = eqx.field(converter=jnp.array)
    ion: str = eqx.field(converter=str)

    def __init__(self, name: str = None, index: Array = None):
        super().__init__(name, index)

    def g(self, t, u, args=None):
        return 1.0
    
    def i(self, t, u, args=None):
        v = u[0]
        return self.g(t, u, args) * (v - self.e)

    def vf(self, t, u, args=None):
        return ()

    def tau(self, t, u, args=None):
        return ()

    def ss(self, t, u, args=None):
        return ()

    def init(self, t, u, args=None):
        return ()


a_m = lambda v: 0.1 * _vtrap(-(v + 40), 10)
b_m = lambda v: 4.0 * safe_exp(-(v + 65) / 18)
a_h = lambda v: 0.07 * safe_exp(-(v + 65) / 20)
b_h = lambda v: 1.0 / (safe_exp(-(v + 35) / 10) + 1)
a_n = lambda v: 0.01 * _vtrap(-(v + 55), 10)
b_n = lambda v: 0.125 * safe_exp(-(v + 65) / 80)

class Na(Channel):
    reads = ("v", "m", "h")
    writes = ("m", "h")
    ion: str = "na"
    
    def __init__(self, gbar: Array = 120.0, e: Array = 50.0, name: str = None, index: Array = None):
        super().__init__(name, index)
        self.gbar = gbar
        self.e = e

    def vf(self, t, u, args=None):
        v, m, h = u
        dm = a_m(v) * (1 - m) - b_m(v) * m
        dh = a_h(v) * (1 - h) - b_h(v) * h
        return dm, dh

    def g(self, t, u, args=None):
        v, m, h = u
        return self.gbar * m**3 * h

    def i(self, t, u, args=None):
        v, m, h = u
        return self.g(t, u, args) * (v - self.e)

    def tau(self, t, u, args=None):
        v, m, h = u
        tau_m = 1 / (a_m(v) + b_m(v))
        tau_h = 1 / (a_h(v) + b_h(v))
        return tau_m, tau_h

    def ss(self, t, u, args=None):
        v, m, h = u
        tau = self.tau(t, u, args)
        m_inf = a_m(v) * tau["m"]
        h_inf = a_h(v) * tau["h"]
        return m_inf, h_inf

    def init(self, t, u, args=None):
        return self.ss(t, u, args)

class K(Channel):
    reads = ("v", "n")
    writes = ("n",)
    ion: str = "k"

    def __init__(self, gbar: Array = 36.0, e: Array = -77.0, name: str = None, index: Array = None):
        super().__init__(name, index)
        self.gbar = gbar
        self.e = e

    def vf(self, t, u, args=None):
        v, n = u
        dn = a_n(v) * (1 - n) - b_n(v) * n
        return dn

    def g(self, t, u, args=None):
        v, n = u
        return self.gbar * n**4

    def i(self, t, u, args=None):
        v, n = u
        return self.g(t, u, args) * (v - self.e)

    def tau(self, t, u, args=None):
        v, n = u
        return 1 / (a_n(v) + b_n(v))

    def ss(self, t, u, args=None):
        v, n = u
        return a_n(v) * 1 / (a_n(v) + b_n(v))

    def init(self, t, u, args=None):
        return self.ss(t, u, args)

class Leak(Channel):
    reads = ("v",)
    writes = ()
    ion: str = "leak"

    def __init__(self, gbar: Array = 0.0003, e: Array = -54.3, name: str = None, index: Array = None):
        super().__init__(name, index)
        self.gbar = gbar
        self.e = e

    def vf(self, t, u, args=None):
        return ()

    def g(self, t, u, args=None):
        return self.gbar

    def i(self, t, u, args=None):
        v = u[0]
        return self.g(t, u, args) * (v - self.e)
    
    def tau(self, t, u, args=None):
        return ()

    def ss(self, t, u, args=None):
        return ()

    def init(self, t, u, args=None):
        return ()
