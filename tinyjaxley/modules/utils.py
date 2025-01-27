import pandas as pd

def recurse(f):
    def wrapper(self, *args, **kwargs):
        out = [recurse(f)(sm, *args, **kwargs) for sm in self]
        if self.submodules is None:
            return f(self, *args, **kwargs)
        if out[0] == None:
            return None
        return out
    return wrapper

def nested_dict_to_df(d):
    def get_nested_kv(d, pad=""): 
        keys, values = [], []
        for k,v in d.items():
            kk, vv = get_nested_kv(v, k) if isinstance(v, dict) else ([(pad, k)], [v])
            keys += kk
            values += vv
        return keys, values

    keys, values = get_nested_kv(d)
    keys = [k if k[0] != "" else k[::-1] for k in keys]
    df = pd.DataFrame([values], columns=pd.MultiIndex.from_tuples(keys))
    return df

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