from __future__ import annotations

import jax.numpy as jnp

from tinyjaxley.mechanisms.mechanism import Channel
from tinyjaxley.utils import Param, State, _vtrap, safe_exp


def a_m(v):
    return 0.1 * _vtrap(-(v + 40.0), 10.0)


def b_m(v):
    return 4.0 * safe_exp(-(v + 65.0) / 18.0)


def a_h(v):
    return 0.07 * safe_exp(-(v + 65.0) / 20.0)


def b_h(v):
    return 1.0 / (1.0 + safe_exp(-(v + 35.0) / 10.0))


def a_n(v):
    return 0.01 * _vtrap(-(v + 55.0), 10.0)


def b_n(v):
    return 0.125 * safe_exp(-(v + 65.0) / 80.0)


class Leak(Channel):
    s0 = ("v",)
    p0 = (Param("leak.gbar", 0.3), Param("leak.e", -54.3))
    current = "ileak"

    def i(self, t, u, p, args):
        return -p.gbar * (u.v - p.e)


class Na(Channel):
    s0 = ("v", State("na.m", 0.05), State("na.h", 0.60))
    p0 = (Param("na.gbar", 120.0), Param("na.e", 50.0))
    current = "ina"

    def i(self, t, u, p, args):
        return -p.gbar * u.m**3 * u.h * (u.v - p.e)

    def init(self, t, u, p, args):
        am, bm = a_m(u.v), b_m(u.v)
        ah, bh = a_h(u.v), b_h(u.v)
        return jnp.array([am / (am + bm), ah / (ah + bh)])

    def vf(self, t, u, p, args):
        dm = a_m(u.v) * (1.0 - u.m) - b_m(u.v) * u.m
        dh = a_h(u.v) * (1.0 - u.h) - b_h(u.v) * u.h
        return jnp.array([dm, dh])


class K(Channel):
    s0 = ("v", State("k.n", 0.32))
    p0 = (Param("k.gbar", 36.0), Param("k.e", -77.0))
    current = "ik"

    def i(self, t, u, p, args):
        return -p.gbar * u.n**4 * (u.v - p.e)

    def init(self, t, u, p, args):
        an, bn = a_n(u.v), b_n(u.v)
        return jnp.array([an / (an + bn)])

    def vf(self, t, u, p, args):
        dn = a_n(u.v) * (1.0 - u.n) - b_n(u.v) * u.n
        return jnp.array([dn])
