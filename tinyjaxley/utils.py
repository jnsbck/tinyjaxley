import jax.numpy as jnp
import jax.tree_util as jtu
import pandas as pd
from collections import ChainMap

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

def g_ij(p_i, p_j):
    """
    from `https://en.wikipedia.org/wiki/Compartmental_neuron_models`.
    `radius`: um, `Ra`: ohm cm, `l`: um, `g`: mS / cm^2
    """
    r_i, Ra_i, l_i = p_i["r"], p_i["Ra"], p_i["l"]
    # if u_j is None: return r_j / Ra_j / l_i**2
    r_j, Ra_j, l_j = p_j["r"], p_j["Ra"], p_j["l"]
    g = r_i*r_j**2 / (Ra_i * r_j**2 * l_i + Ra_j * r_i**2 * l_j) / l_i
    return g * 10**7 # S/cm/um -> mS / cm²

def comp_only(func):
    def wrapper(self, *args, **kwargs):
        if self.key == "comp": return func(self, *args, **kwargs)
        else: 
            if len(out := [func(c, *args, **kwargs) for c in self.children]) > 0:
                if all([item is None for item in out]): return None
            return out
    return wrapper