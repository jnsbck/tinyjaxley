from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from tinyjaxley import HH, Model, StepCurrent
from tinyjaxley.mechanisms.channel import a_h, a_m, a_n, b_h, b_m, b_n
from tinyjaxley.solve import forward_euler


def _hh_model():
    return (
        Model(1)
        .insert(HH(), at=jnp.array([0]))
        .insert(StepCurrent(amp=10.0, delay=1.0, dur=3.0), at=jnp.array([0]))
        .build()
    )


def _baseline_vf(model):
    params = model.unravel_p(model.p)
    area = 2 * jnp.pi * params["rad"] * params["len"]

    def vf(t, u):
        state = model.unravel_u(u)
        v, m, h, n = state["v"], state["hh.m"], state["hh.h"], state["hh.n"]
        ina = -params["hh.gNa"] * m**3 * h * (v - params["hh.eNa"])
        ik = -params["hh.gK"] * n**4 * (v - params["hh.eK"])
        ileak = -params["hh.gLeak"] * (v - params["hh.eLeak"])
        istim = jnp.where(
            (t >= params["stim.delay"])
            & (t < params["stim.delay"] + params["stim.dur"]),
            params["stim.amp"],
            0.0,
        )
        dv = (ina + ik + ileak + istim / area) / params["cap"]
        dm = a_m(v) * (1.0 - m) - b_m(v) * m
        dh = a_h(v) * (1.0 - h) - b_h(v) * h
        dn = a_n(v) * (1.0 - n) - b_n(v) * n

        du = jnp.zeros_like(u)
        du = du.at[model.u_inds["v"]].set(dv)
        du = du.at[model.u_inds["hh.m"]].set(dm)
        du = du.at[model.u_inds["hh.h"]].set(dh)
        return du.at[model.u_inds["hh.n"]].set(dn)

    return vf


def _integrate(rhs, y0, ts):
    return forward_euler(lambda t, y, args: rhs(t, y), y0, ts)


def _timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return time.perf_counter() - t0, out


def test_single_comp_hh_matches_baseline():
    model = _hh_model()
    baseline_vf = _baseline_vf(model)
    model_vf = jax.jit(lambda t, u: model.vf(t, u))
    baseline_vf = jax.jit(baseline_vf)

    y0 = model.u0
    model_vf(0.0, y0).block_until_ready()
    baseline_vf(0.0, y0).block_until_ready()

    model_t, model_du = _timed(model_vf, 1.5, y0)
    baseline_t, baseline_du = _timed(baseline_vf, 1.5, y0)
    print(f"forward model={model_t:.6f}s baseline={baseline_t:.6f}s")
    assert jnp.allclose(model_du, baseline_du, rtol=1e-6, atol=1e-6)

    ts = jnp.linspace(0.0, 5.0, 101)
    model_integrate = jax.jit(lambda y: _integrate(model_vf, y, ts))
    baseline_integrate = jax.jit(lambda y: _integrate(baseline_vf, y, ts))
    model_integrate(y0).block_until_ready()
    baseline_integrate(y0).block_until_ready()

    model_t, model_sol = _timed(model_integrate, y0)
    baseline_t, baseline_sol = _timed(baseline_integrate, y0)
    print(f"integrate model={model_t:.6f}s baseline={baseline_t:.6f}s")
    assert jnp.allclose(model_sol, baseline_sol, rtol=1e-5, atol=1e-5)
