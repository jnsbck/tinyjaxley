import jax.numpy as jnp

from .utils import safe_exp, taux, xinf


def fw_euler(du, u, t, dt = 0.025): 
    return u + du * dt, t + dt

def exp_euler(du, u, t, dt = 0.025):
    return u * safe_exp(du * dt), t + dt

def gate_impl_euler(x, dt, α, β):
    α_x = x + dt * α
    β_x = 1.0 + dt * α + dt * β
    return α_x / β_x

def gate_exp_euler(x, α, β, dt = 0.025):
    τ = taux(x, α, β)
    return x * safe_exp(-dt / τ) + xinf(x, α, β) * (1.0 - safe_exp(-dt / τ))