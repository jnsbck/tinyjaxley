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