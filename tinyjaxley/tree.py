from copy import deepcopy
from abc import ABC, abstractmethod
from collections import ChainMap

class Tree(ABC):
    def __init__(self, submodules = [], index = 0):
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
    def states(self):
        d = [self._states]
        if self.submodules is None: return d[0]
        d += [sm.states for sm in self.submodules]
        return d
        
    @property
    def params(self):
        d = [self._params]
        if self.submodules is None: return d[0]
        d += [sm.params for sm in self.submodules]
        return d
   
    @property
    def params_states(self):
        d = [ChainMap(self._states, self._params)] # ChainMap avoids copy
        if self.submodules is None: return d[0]
        d += [sm.params_states for sm in self.submodules]
        return d

    def __getitem__(self, index):
        view = self
        index = index if isinstance(index, tuple) else (index,)
        for i in index:
            view = view.at(i)
        return view
    
    def __iter__(self):
        if self.submodules is not None:
            for sm in self.submodules:
                yield sm

    def at(self, index):
        if self.submodules is not None:
            return self.submodules[index]
    
    def __repr__(self, indent = ""):
        params_states = "" #self.param_states[self.name]
        repr_str = f"{indent}{self.name}{params_states}[{self.index}]"
        if self.submodules is not None:
            repr_str += "(\n"
            repr_str += "".join([f"{sm.__repr__(indent + '    ')},\n" for sm in self]) 
            repr_str += indent + ")"
        return repr_str
    
    def __str__(self):
        return self.__repr__()
    
    def __len__(self):
        return len(self.submodules)
