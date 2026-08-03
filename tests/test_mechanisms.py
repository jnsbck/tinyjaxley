import jax
import jax.numpy as jnp

from tinycable import K, Na, Leak, Ns


def _namespaces(mechanism, voltages):
    states = {
        key.rsplit(".", 1)[-1]: voltages
        if value is None
        else jnp.full_like(voltages, value)
        for key, value in mechanism.states.items()
    }
    params = {
        key.rsplit(".", 1)[-1]: jnp.full_like(voltages, value)
        for key, value in mechanism.params.items()
    }
    return Ns(states), Ns(params)


def test_hh_mechanism_contract():
    voltages = jnp.array([-80.0, -65.0, -40.0, -20.0])

    for mechanism in (Leak(), Na(), K()):
        # Declarations provide the static information compilation will consume.
        assert mechanism.density
        assert mechanism.states["v"] is None
        assert len(mechanism.currents) == 1

        states, params = _namespaces(mechanism, voltages)
        derivatives = mechanism.d(0.0, states, params)

        # Ns views are built inside tracing rather than passed as runtime pytrees.
        current = jax.jit(
            lambda state, parameters: mechanism.i(0.0, Ns(state), Ns(parameters))
        )(dict(states), dict(params))
        declared_states = {
            key.rsplit(".", 1)[-1]
            for key, value in mechanism.states.items()
            if value is not None
        }

        # Derivatives agree with declarations and vectorize over instances.
        assert set(derivatives) == declared_states
        assert current.shape == voltages.shape
        assert jnp.all(jnp.isfinite(current))
        assert all(value.shape == voltages.shape for value in derivatives.values())
        assert all(jnp.all(jnp.isfinite(value)) for value in derivatives.values())

        # Currents remain differentiable with respect to conductance fields.
        current_grad = jax.grad(
            lambda conductance: jnp.sum(
                mechanism.i(0.0, states, Ns({**params, "g": conductance}))
            )
        )(params.g)
        assert jnp.all(jnp.isfinite(current_grad))

        # Membrane currents are outward-positive above their reversal potential.
        reversal = next(key for key in params if key != "g")
        above_reversal = Ns({**states, "v": params[reversal] + 1.0})
        assert jnp.all(mechanism.i(0.0, above_reversal, params) > 0.0)


def test_hh_rate_singularities_have_finite_gradients():
    def sodium_m(v):
        state = Ns(v=v, m=jnp.array(0.0), h=jnp.array(0.5961))
        return Na().d(0.0, state, Ns())["m"]

    def potassium_n(v):
        state = Ns(v=v, n=jnp.array(0.0))
        return K().d(0.0, state, Ns())["n"]

    # Removable HH singularities are safe in forward and reverse mode.
    for rate, voltage in ((sodium_m, -40.0), (potassium_n, -55.0)):
        voltage = jnp.array(voltage)
        assert jnp.isfinite(rate(voltage))
        assert jnp.isfinite(jax.grad(rate)(voltage))
