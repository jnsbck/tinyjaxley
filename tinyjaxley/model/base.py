from __future__ import annotations

import jax.numpy as jnp

from tinyjaxley.utils import (
    Current,
    IndexMap,
    Param,
    State,
    build_kernel,
    compute_index_maps,
    gather,
    merge_fields,
    insert_field,
    ravel_fields,
    scatter_add,
)


class Model:
    """Minimal model spec: insert(mech, at=inds); at omitted means all comps."""

    def __init__(
        self,
        ncomp: int,
    ):
        if ncomp <= 0:
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
        self.i0 = {}
        self.mechanisms = {}
        self.kernels = []
        self._built = False

    def insert(self, mech, *, at=None):
        at = self.index if at is None else jnp.atleast_1d(at)
        mech = mech._broadcast(index=at)
        self.mechanisms[mech.name] = mech # TODO: broadcast and merge mechs too!

        # move this to global cache in lower phase
        # for x0 in mech.s0 + mech.p0 + mech.i0:
        #     if isinstance(x0, State):
        #         self.s0 = insert_field(self.s0, x0)
        #     elif isinstance(x0, Param):
        #         self.p0 = insert_field(self.p0, x0)
        #     elif isinstance(x0, Current):
        #         self.i0 = insert_field(self.i0, x0)
        self._built = False
        return self

    def build(self):
        self.u0, self.u_inds, self.unravel_u = ravel_fields(self.s0)
        self.currents0, self.i_inds, self.unravel_i = ravel_fields(self.i0)
        self.p, self.p_inds, self.unravel_p = ravel_fields(self.p0)
        self.kernels = tuple(self._build_kernel(mech) for mech in self.mechanisms.values())
        
        # self.i_to_v = self._ravel_i_map("v", self.u_inds)
        # self.i_to_cap = self._ravel_i_map("cap", self.p_inds)
        # self.i_to_rad = self._ravel_i_map("rad", self.p_inds)
        # self.i_to_len = self._ravel_i_map("len", self.p_inds)
        self._built = True
        return self

    def vf(self, t, u=None, p=None, args=None):

        if not self._built:
            self.build()
        
        du = jnp.zeros_like(u)
        currents = jnp.zeros_like(self.currents0)

        for kernel in self.kernels:
            u_loc = gather(u, kernel.reads.u, kernel.reads.u._fields)
            p_loc = gather(p, kernel.reads.p, kernel.reads.p._fields)

            # # TODO: compute these gathers once per kernel and pass to vf and i
            # u_loc = kernel.u.gather(u) 
            # p_loc = kernel.p.gather(p)

            # TODO: make the scatter add behaviour depend on the kernel
            # TODO: accumulate scatters before scattering to reduce overhead
            du_loc = kernel.vf(t, u_loc, p_loc, args)
            du = scatter_add(du, du_loc, kernel.writes.u) 

            i_loc = kernel.i(t, u_loc, p_loc, args)
            currents = scatter_add(currents, i_loc, kernel.writes.i)

        # TODO: do area divide with precomputed mapping
        # area = 2 * jnp.pi * p[self.i_to_len] * r[self.i_to_rad]
        # currents = scatter_div(currents, area, self.i_to_cap)
        return du.at[self.i_to_v].add(currents / p[self.i_to_cap])

    def _build_kernel(self, mech):
        reads = IndexMap(
            compute_index_maps(self.s0, self.u_inds, mech.index, mech.s0),
            compute_index_maps(self.i0, self.i_inds, mech.index, mech.i0),
            compute_index_maps(self.p0, self.p_inds, mech.index, mech.p0),
        )
        writes = IndexMap(
            compute_index_maps(self.s0, self.u_inds, mech.index, mech.s0, True),
            compute_index_maps(self.i0, self.i_inds, mech.index, mech.i0, True),
            None,
        )
        area = compute_index_maps(self.p0, self.p_inds, mech.index, ("rad", "len"))
        return build_kernel(mech, reads, writes, area)

    def _ravel_i_map(self, ref, inds):
        vals, _, _ = ravel_fields(
            {
                key: field.set(value=inds[ref][field.index])
                for key, field in self.i0.items()
            }
        )
        return vals.astype(jnp.int32)
