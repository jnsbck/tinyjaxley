import jax.numpy as jnp
import jax.tree_util as jtu

def safe_exp(x, max_value: float = 20.0):
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)

def _vtrap(x, y): return x / (safe_exp(x/y) - 1.0)
def taux(v, a, b): return 1 / (a(v) + b(v))
def xinf(v, a, b): return a(v) * taux(v, a, b)

def find(keys, tree, fill_leaf = None):
    def process_leaf(path, value):
        if path[-1].key in keys: return value
        return fill_leaf
    
    mapped = jtu.tree_map_with_path(process_leaf, tree)
    return jtu.tree_map(lambda x: x, mapped)