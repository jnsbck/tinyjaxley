from copy import deepcopy, copy
import pandas as pd
import jax.numpy as jnp

from .utils import recurse, nested_dict_to_df

class Module:
    def __init__(self, submodules = [], index = 0):
        self.groups = {}
        self._submodules = None
        self.index = index
        self._states = {}
        self._params = {}

        self.submodules = submodules
        if submodules is not None:
            for i, sm in enumerate(self.submodules):
                sm.index = i

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def submodules(self):
        return self._submodules
    
    @submodules.setter
    def submodules(self, submodules):
        if submodules is not None:
            self._submodules = [deepcopy(u) for u in submodules]

    @property
    @recurse
    def states(self):
        channel_states = {}
        for name, c in self.channels.items():
            channel_states.update({name: c.states})
        return {**self._states, **channel_states}
    
    @states.setter
    @recurse
    def states(self, dct):
        for k,v in dct.items():
            if k in self.states:
                if isinstance(v, dict) and k in self.channels:
                    self.channels[k].states.update(v)
                else:
                    self._states[k] = v
        
    @property
    @recurse
    def params(self):
        channel_params = {}
        for name, c in self.channels.items():
            channel_params.update({name: c.params})
        return {**self._params, **channel_params}
    
    @params.setter
    @recurse
    def params(self, dct):
        for k,v in dct.items():
            if k in self.params:
                if isinstance(v, dict) and k in self.channels:
                    self.channels[k].params.update(v)
                else:
                    self._params[k] = v

    @property
    def flat(self):
        if self.submodules is not None:
            flat_module = copy(self)
            flat_module._submodules = self.flatten()
            return flat_module
        return self
    
    @property
    @recurse
    def param_states(self):
        return self.param_states
    
    def show(self):
        if self.submodules is None:
            return nested_dict_to_df(self.flat.param_states)
        
        nodes, inds = [], []
        for i, sm in self.enumerate():
            nodes.append(sm.show())
            inds.append(i)
        df = pd.concat(nodes, ignore_index = True)
        df.index = pd.MultiIndex.from_tuples(inds)
        return df
    
    @property
    @recurse
    def xyzr(self):
        return self._xyzr
    
    @property
    @recurse
    def stimuli(self):
        return self._externals["i"]
    
    @property
    @recurse
    def recordings(self):
        return self._recordings
    
    @property
    @recurse
    def externals(self):
        return self._externals
    
    @property
    @recurse
    def channels(self):
        return self._channels

    @property
    def shape(self):
        if self.submodules is not None:
            return (len(self),) + self.submodules[0].shape()
        return ()
    
    def __repr__(self, indent = ""):
        repr_str = f"{indent}{self.name}[{self.index}]"
        if self.submodules is not None:
            repr_str += "(\n"
            repr_str += "".join([f"{sm.__repr__(indent + '    ')},\n" for sm in self]) 
            repr_str += indent + ")"
        return repr_str
    
    def __str__(self):
        return self.__repr__()
    
    def __len__(self):
        return len(self.submodules)
    
    def __getattr__(self, key):
        if key.startswith("__"):
            return self.__getattribute__(key)
        
        if self.submodules is not None:
            sub_name = self.submodules[0].name
            if key == sub_name:
                return self.at
            
            if key == sub_name + "s" or key == sub_name + "es":
                return self.submodules
            
    def __getitem__(self, index):
        return self.at(index)
    
    def __iter__(self):
        if self.submodules is not None:
            for sm in self.submodules:
                yield sm

    def enumerate(self):
        for i, sm in enumerate(self):
            for index, sm in sm.enumerate():
                yield (i, *index), sm
        if self.submodules is None:
            yield (), self

    def at(self, index):
        if self.submodules is not None:
            return self.submodules[index]

    @recurse
    def add(self, mech):
        self.channels[mech.name] = deepcopy(mech)

    @recurse
    def set(self, key, value):
        self.states = {key: value}
        self.params = {key: value}      

    @recurse
    def get(self, key):
        if key in self.states:
            return self.states[key]
        if key in self.params:
            return self.params[key]
        return None

    def flatten(self, comps_only = False):
        submodules = sum([sm.flatten(comps_only) for sm in self], [])
        if comps_only and self.submodules is not None:
            return submodules
        return [self] + submodules

    def select(self, index):
        flat_module = copy(self)
        comps = [comp for i, (inds, comp) in enumerate(self.enumerate()) if i in index]
        flat_module._submodules = comps
        return flat_module
    
    @recurse
    def record(self, key = "v"):
        self._recordings += [key]
    
    @recurse
    def clamp(self, key, values):
        if key not in self._externals:
            self._externals[key] = []
        self._externals[key] += [values]

    def stimulate(self, values):
        self.clamp("i", values)

    @recurse
    def move(self, x, y, z):
        self.xyzr[:, :3] += jnp.array([x, y, z])