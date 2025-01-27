from ..tree import Tree
from copy import copy, deepcopy
import pandas as pd
from ..utils import nested_dict_to_df

class Module(Tree):
    def __init__(self, submodules = []):
        super().__init__(submodules)

    def __getattr__(self, key):
        if key.startswith("__"):
            return self.__getattribute__(key)
        
        if len(self.submodules) > 0:
            if key == self.submodules[0].name:
                assert self.channels is None, "No channel indexing allowed."
                return self.at

    def show(self):
        param_states = []
        inds = []
        for comp in self.flatten(comps_only=True):
            keys = [comp.name] + [c.name for c in comp.submodules]
            param_states += [{k: d for k, d in zip(keys, comp.params_states)}]
            inds += [comp.index]
        df = pd.concat([nested_dict_to_df(d) for d in param_states])
        df.index = inds
        return df
    
    def get(self, ref):
        ref = ref if isinstance(ref, tuple) else (ref,)
        param_states = [comp.get(ref) for comp in self.flatten(comps_only=True)]
        return param_states
    
    def set(self, ref, value):
        ref = ref if isinstance(ref, tuple) else (ref,)
        for comp in self.flatten(comps_only=True):
            comp.set(ref, value)

    @property
    def shape(self):
        if self.submodules[0].name == "comp":
            return (len(self),) + self.submodules[0].shape
        return ()
    
    def __str__(self):
        return self.__repr__()
    
    def select(self, index):
        flat_module = copy(self)
        comps = [comp for i, comp in enumerate(self.flatten(comps_only=True)) if i in index]
        flat_module._submodules = comps
        return flat_module

    def flatten(self, comps_only = False):
        sms = sum([sm.flatten(comps_only) for sm in self.submodules if hasattr(sm, "flatten")], [])
        if self.name == "comp": 
            return [self]
        if comps_only and self.submodules[0].name == "comp": return sms
        return [self] + sms
    
    def insert(self, submodule, index = None):
        index = len(self.submodules) if index is None else index
        if self.name == "comp":
            self.submodules.insert(index, deepcopy(submodule))
            for i, sm in enumerate(self.submodules):
                sm.index = i
        else:
            for sm in self.submodules:
                sm.insert(submodule, index)

    def record(self, key):
        pass

    def stimulate(self, key, value):
        pass

    def clamp(self, key, value):
        pass