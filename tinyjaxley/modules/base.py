from ..tree import Node
from copy import copy, deepcopy
import pandas as pd
from ..utils import nested_dict_to_df, comp_only

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
        # TODO: Simplify this.
        param_states = []
        inds = []
        for comp in self.flatten(comps_only=True):
            if not comp.is_leaf():
                channel_keys = [c.key for c in comp.children]
                node_ps, channel_ps = comp.params_states
                node_ps = {comp.key: node_ps}
                channel_ps = {k: v for k, v in zip(channel_keys, channel_ps)}
                param_states += [{**node_ps, **channel_ps}]
            else:
                param_states += [comp.params_states]
            inds += [comp.index]
        df = pd.concat([nested_dict_to_df(d) for d in param_states])
        df.index = inds
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
        flat_module._children = comps
        return flat_module

    def flatten(self, comps_only = False):
        if comps_only:
            return [sm for sm in super().flatten() if sm.key == "comp"]
        return super().flatten()
    
    @comp_only
    def insert(self, submodule):
        self._children.append(deepcopy(submodule))
        for i, sm in enumerate(self.children):
            sm.index = i
            sm.parent = self

    @comp_only
    def record(self, key):
        self._recordings.append(key)

    @comp_only
    def clamp(self, key, value):
        if key not in self._externals:
            self._externals[key] = []
        self._externals[key] += [value]

    @comp_only
    def stimulate(self, value):
        self.clamp("i", value)

    # @comp_only
    # def xyzr(self):
    #     pass

    # def init(self, t = 0, u = None, p = None):
    #     u = self.states if u is None else u
    #     p = self.params if p is None else p
    #     for key, ext in self._externals.items():
    #         u[key] =  sum(f(t, u, p) for f in ext)
    #         self.states = {key: u[key]}
    #     for cname, channel in self._channels.items():
    #         u[cname] = channel.init(t, u[cname], p[cname], u["v"])
    #         self.states = {cname: u[cname]}
    #     return u
