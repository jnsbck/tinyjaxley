import jax.numpy as jnp

def safe_exp(x, max_value: float = 20.0):
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)