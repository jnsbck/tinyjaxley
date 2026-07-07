from __future__ import annotations

from tinyjaxley.mechanisms.mechanism import Mechanism
from tinyjaxley.utils import Current, Param, State, _vtrap, safe_exp


class Channel(Mechanism):
    is_density = True


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
    i0 = (Current("ileak", 0.0),)

    def i(self, t, u, p, args=None):
        return (-p.gbar * (u.v - p.e),)


class HH(Channel):
    s0 = ("v", State("hh.m", 0.05), State("hh.h", 0.60), State("hh.n", 0.32))
    p0 = (
        Param("hh.gNa", 120.0),
        Param("hh.gK", 36.0),
        Param("hh.gLeak", 0.3),
        Param("hh.eNa", 50.0),
        Param("hh.eK", -77.0),
        Param("hh.eLeak", -54.3),
    )
    i0 = (Current("ihh", 0.0),)

    def i(self, t, u, p, args=None):
        ina = -p.gNa * u.m**3 * u.h * (u.v - p.eNa)
        ik = -p.gK * u.n**4 * (u.v - p.eK)
        ileak = -p.gLeak * (u.v - p.eLeak)
        return (ina + ik + ileak,)

    def init(self, t, u, p, args=None):
        am, bm = a_m(u.v), b_m(u.v)
        ah, bh = a_h(u.v), b_h(u.v)
        an, bn = a_n(u.v), b_n(u.v)
        return am / (am + bm), ah / (ah + bh), an / (an + bn)

    def vf(self, t, u, p, args=None):
        dm = a_m(u.v) * (1.0 - u.m) - b_m(u.v) * u.m
        dh = a_h(u.v) * (1.0 - u.h) - b_h(u.v) * u.h
        dn = a_n(u.v) * (1.0 - u.n) - b_n(u.v) * u.n
        return dm, dh, dn


class Na(Channel):
    s0 = ("v", State("na.m", 0.05), State("na.h", 0.60))
    p0 = (Param("na.gbar", 120.0), Param("na.e", 50.0))
    i0 = (Current("ina", 0.0),)

    def i(self, t, u, p, args=None):
        return (-p.gbar * u.m**3 * u.h * (u.v - p.e),)

    def init(self, t, u, p, args=None):
        am, bm = a_m(u.v), b_m(u.v)
        ah, bh = a_h(u.v), b_h(u.v)
        return am / (am + bm), ah / (ah + bh)

    def vf(self, t, u, p, args=None):
        dm = a_m(u.v) * (1.0 - u.m) - b_m(u.v) * u.m
        dh = a_h(u.v) * (1.0 - u.h) - b_h(u.v) * u.h
        return dm, dh


class K(Channel):
    s0 = ("v", State("k.n", 0.32))
    p0 = (Param("k.gbar", 36.0), Param("k.e", -77.0))
    i0 = (Current("ik", 0.0),)

    def i(self, t, u, p, args=None):
        return (-p.gbar * u.n**4 * (u.v - p.e),)

    def init(self, t, u, p, args=None):
        an, bn = a_n(u.v), b_n(u.v)
        return (an / (an + bn),)

    def vf(self, t, u, p, args=None):
        dn = a_n(u.v) * (1.0 - u.n) - b_n(u.v) * u.n
        return (dn,)
