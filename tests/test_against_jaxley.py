from tinyjaxley.modules import Comp, Branch, Cell
from tinyjaxley.mechanisms import SquarePulse, Na, K, Leak
import jax.numpy as jnp
import diffrax

from jaxley.channels import HH
import jaxley as jx


def test_comp():
    comp = Comp()
    comp = comp.insert(Na())
    comp = comp.insert(K())
    comp = comp.insert(Leak())

    t0 = 0.0
    t1 = 25.0
    dt = 0.01
    ts = jnp.arange(t0, t1, dt)

    stim_start = 5.0
    stim_end = 20.0
    stim_amp = 0.01
    stim = SquarePulse(stim_amp, stim_start, stim_end)
    comp = comp.insert(stim)

    solver = diffrax.Tsit5()
    u0 = comp.init(0.0, {"v": jnp.array(-70.0)})

    ode = diffrax.ODETerm(comp)
    saveat = diffrax.SaveAt(ts=ts)
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)
    sol = diffrax.diffeqsolve(
        ode,
        solver,
        t0,
        t1,
        dt,
        y0=u0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

    jx_comp = jx.Compartment()
    jx_comp.insert(HH())
    jx_comp.record("v")
    jx_comp.init_states()

    jx_stim = jx.step_current(stim_start, stim_end - stim_start, stim_amp, dt, t1)
    jx_comp.stimulate(jx_stim)

    v = jx.integrate(jx_comp, delta_t=dt).flatten()[:-2]

    assert jnp.allclose(sol.ys["v"], v, atol=1e-4)


def test_branch():
    ncomps = 3
    comp = Comp()
    branch = Branch([comp] * ncomps)
    branch = branch.insert(Na())
    branch = branch.insert(K())
    branch = branch.insert(Leak())

    stim = SquarePulse(0.01, 5.0, 20.0)
    branch = branch.insert(stim, at=jnp.array([0]))
    t0 = 0.0
    t1 = 25.0
    dt = 0.025
    solver = diffrax.Tsit5()
    u0 = branch.init(0.0, {"v": jnp.array([-70.0] * ncomps)})

    ode = diffrax.ODETerm(branch)
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    sol = diffrax.diffeqsolve(
        ode,
        solver,
        t0,
        t1,
        dt,
        y0=u0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

    jx_branch = jx.Branch([jx.Compartment()] * ncomps)
    jx_branch.insert(HH())

    jx_stim = jx.step_current(5.0, 15.0, 0.01, dt, t1)
    jx_branch.select(0).stimulate(jx_stim)

    jx_branch.record("v")
    jx_branch.init_states()
    v = jx.integrate(jx_branch, delta_t=dt)[:, :-2]

    assert jnp.allclose(sol.ys["v"], v, atol=1e-4)


def test_cell():
    ncomps = 3
    nbranches = 3
    comp = Comp()
    branch = Branch([comp] * ncomps)
    cell = Cell([branch] * nbranches, jnp.array([-1, 0, 0]))
    cell = cell.insert(K())
    cell = cell.insert(Na())
    cell = cell.insert(Leak())

    stim = SquarePulse(0.1, 5.0, 20.0)
    cell = cell.insert(stim, at=jnp.array([0]))
    t0 = 0.0
    t1 = 25.0
    dt = 0.025
    solver = diffrax.Tsit5()
    u0 = cell.init(0.0, {"v": jnp.array([-70.0] * ncomps * nbranches)})

    ode = diffrax.ODETerm(cell)
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    sol = diffrax.diffeqsolve(
        ode,
        solver,
        t0,
        t1,
        dt,
        y0=u0,
        saveat=saveat,
    )

    jx_branch = jx.Branch([jx.Compartment()] * ncomps)
    jx_cell = jx.Cell([jx_branch] * nbranches, jnp.array([-1, 0, 0]))
    jx_cell.insert(HH())

    ts = jnp.arange(0.0, t1, dt)
    jx_stim = jx.step_current(5.0, 15.0, 0.1, dt, t1)
    jx_cell.select(0).stimulate(jx_stim)

    jx_cell.record("v")
    jx_cell.init_states()
    v = jx.integrate(jx_cell, delta_t=dt)[:, :-2]

    assert jnp.allclose(sol.ys["v"], v, atol=1e-6)
