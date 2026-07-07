from __future__ import annotations

import copy

import jax.numpy as jnp

from tinyjaxley.utils import is_field


class Mechanism:
    """Mechanisms declare fields and are broadcast to the comps they run on."""

    domain = "comp"
    s0 = ()
    p0 = ()
    i0 = ()
    is_density = True
    index = None

    def __init__(self, **overrides):
        remaining = dict(overrides)
        self.s0 = tuple(self._apply(entry, remaining) for entry in self.__class__.s0)
        self.p0 = tuple(self._apply(entry, remaining) for entry in self.__class__.p0)
        if remaining:
            raise ValueError(
                f"Unknown overrides for {type(self).__name__}: {set(remaining)}"
            )

    @staticmethod
    def _apply(entry, overrides):
        if is_field(entry) and entry.name in overrides:
            return entry.set(value=overrides.pop(entry.name))
        return entry

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def _broadcast(self, index, group=None):
        mech = copy.copy(self)
        mech.index = jnp.atleast_1d(index).astype(jnp.int32)
        mech.s0 = tuple(
            entry._broadcast(index=mech.index, group=group)
            if is_field(entry)
            else entry
            for entry in self.s0
        )
        mech.p0 = tuple(
            entry._broadcast(index=mech.index, group=group)
            if is_field(entry)
            else entry
            for entry in self.p0
        )
        mech.i0 = tuple(
            entry._broadcast(index=mech.index, group=group)
            if is_field(entry)
            else entry
            for entry in self.i0
        )
        return mech

    def init(self, t, u, p, args=None):
        return ()

    def vf(self, t, u, p, args=None):
        return ()

    def i(self, t, u, p, args=None):
        return ()

    # def step(self, t0, t1, u0, p, args):
    #     dt = t1 - t0
    #     return u0 + self.vf(t0, u0, p, args)*dt
