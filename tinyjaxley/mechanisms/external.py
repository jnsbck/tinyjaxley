from __future__ import annotations

import jax.numpy as jnp

from tinyjaxley.mechanisms.mechanism import Mechanism
from tinyjaxley.utils import Current, Param


class External(Mechanism):
    is_density = False


class StepCurrent(External):
    s0 = ()
    p0 = (Param("stim.amp", 1.0), Param("stim.dur", 1.0), Param("stim.delay", 0.0))
    i0 = (Current("istim", 0.0),)

    def i(self, t, u, p, args=None):
        return (jnp.where((t >= p.delay) & (t < p.delay + p.dur), p.amp, 0.0),)
