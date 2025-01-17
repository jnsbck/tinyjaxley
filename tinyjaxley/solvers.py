import jax.numpy as jnp
from .utils import safe_exp

def fw_euler(du, u, t, dt = 0.025): 
    return u + du * dt, t + dt

def exp_euler(du, u, t, dt = 0.025):
    return u * safe_exp(du * dt), t + dt

def gate_exp_euler(x, x_inf, x_tau, dt = 0.025):
    exp = safe_exp(-dt / x_tau)
    return x * exp + x_inf * (1.0 - exp)