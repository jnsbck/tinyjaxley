from ..tree import Node
from copy import copy
import pandas as pd
from ..utils import comp_only
import equinox as eqx
import jax.numpy as jnp
class Module(Node):
    def __init__(self, parent = None, children = []):
        super().__init__(parent, children)

    def __getattr__(self, key):
        if key.startswith("__"):
            return self.__getattribute__(key)
        
        if len(self.children) > 0:
            if key == self.children[0].key:
                assert self.channels is None, "No channel indexing allowed."
                return self.at

    def show(self):
        rows = []
        for comp in self.flatten(comps_only=True):
            df_data = {
                **{('comp', k): [v] for k, v in {**comp.p, **comp.u}.items()},
                **{(c.key, k): [v] for c in comp.children for k, v in {**c.p, **c.u}.items()}
            }
            rows.append(pd.DataFrame(df_data, index=[comp.index]))
        df = pd.concat(rows)
        is_1d_array = lambda x: isinstance(x, jnp.ndarray) and x.size == x.ndim == 1
        df = df.map(lambda x: x.item() if is_1d_array(x) else x)
        return df

    @property
    def shape(self):
        if not self.is_leaf():
            if self.children[0].key == "comp":
                return (len(self),) + self.children[0].shape
        return ()
    
    def select(self, index):
        # TODO: Fix this!
        flat_module = copy(self)
        comps = [comp for i, comp in enumerate(self.flatten(comps_only=True)) if i in index]
        flat_module.children = comps
        return flat_module

    def flatten(self, comps_only = False):
        if comps_only:
            return [sm for sm in super().flatten() if sm.key == "comp"]
        return super().flatten()
    
    @comp_only
    def insert(self, channel):
        channel = eqx.tree_at(lambda x: x.parent, channel, self)
        channel = eqx.tree_at(lambda x: x.index, channel, len(self.children))
        self.children.append(channel)

    @comp_only
    def record(self, key):
        self.recordings.append(key)

    @comp_only
    def clamp(self, key, func):
        if key not in self.externals:
            self.externals[key] = []
        self.externals[key] += [func]

    @comp_only
    def stimulate(self, func):
        self.clamp("i", func)    

    # @comp_only
    # def xyzr(self):
    #     pass

    @comp_only
    def init(self, t = 0, u = None):
        u = self.all_states if u is None else u
        u_comp, u_channels = u
        for key, ext in self.externals.items():
            u_comp[key] = sum(f(t, u) for f in ext)
        for channel in self.channels:
            u_channels[channel.index] = channel.init(t, u_channels[channel.index], u_comp["v"])
        return [u_comp, u_channels]