from __future__ import annotations

from numbers import Integral

import jax.numpy as jnp

from tinyjaxley.utils import Param, State, merge_fields


class Model:
    """Minimal model spec: insert(mech, at=inds); at omitted means all comps."""

    def __init__(
        self,
        ncomp,
    ):
        if not isinstance(ncomp, Integral) or ncomp <= 0:
            raise ValueError("ncomp must be a positive integer.")
        self.ncomp = int(ncomp)
        self.index = jnp.arange(self.ncomp, dtype=jnp.int32)
        self.s0 = {"v": State("v", -65.0)._broadcast(index=self.index)}
        self.p0 = {
            "rad": Param("rad", 1.0)._broadcast(index=self.index),
            "len": Param("len", 10.0)._broadcast(index=self.index),
            "cap": Param("cap", 1.0)._broadcast(index=self.index),
            "res_ax": Param("res_ax", 100.0)._broadcast(index=self.index),
        }
        self.mechanisms = []
        self.records = {}

    def lower(self):
        raise NotImplementedError(
            "Model.lower() will be implemented with lowering kernels."
        )

    def insert(self, mech, *, at=None):
        at = self.index if at is None else jnp.atleast_1d(at).astype(jnp.int32)
        mech = mech._broadcast(index=at)
        self.mechanisms.append(mech)

        for x0 in mech.s0 + mech.p0:
            if isinstance(x0, State):
                self.s0[x0.ref] = (
                    merge_fields(self.s0[x0.ref], x0) if x0.ref in self.s0 else x0
                )
            elif isinstance(x0, Param):
                self.p0[x0.ref] = (
                    merge_fields(self.p0[x0.ref], x0) if x0.ref in self.p0 else x0
                )
        return self

    def vf(self, t, u, p, args):
        raise NotImplementedError(
            "Call Model.lower() to get a runnable model with a vf method."
        )