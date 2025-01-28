from copy import deepcopy
from abc import ABC, abstractmethod
from collections import ChainMap

class Node(ABC):
    def __init__(self, parent = None, children = [], index = 0):
        self._parent = parent
        self._children = None
        self.index = index
        self._states = {}
        self._params = {}
        self.children = children

    @property
    def key(self):
        return self.__class__.__name__.lower()
    
    def is_leaf(self):
        if self.children is None: return True
        return len(self.children) == 0
    
    def is_root(self):
        return self._parent is None
    
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def parents(self):
        if self.is_root(): return []
        return [self.parent] + self.parent.parents

    @property
    def children(self):
        return self._children
    
    @children.setter
    def children(self, children):
        if children is not None:
            self._children = []
            for i, c in enumerate(children):
                c.index = i
                c.parent = self
                self._children += [deepcopy(c)]

    def keys(self):
        if self.is_leaf(): return [self.key]
        return [self.key] + [c.key for c in self.children]

    def tree_map(self, f):
        return f(self) if self.is_leaf() else [f(self), [c.tree_map(f) for c in self.children]]
      
    @property
    def states(self):
        return self.tree_map(lambda x: x._states)
        
    @property
    def params(self):
        return self.tree_map(lambda x: x._params)
    
    @property
    def params_states(self):
        return self.tree_map(lambda x: ChainMap(x._states, x._params))
    
    # TODO: Implement this!
    # def get(self, ref):
    #     ref = ref if isinstance(ref, tuple) else (ref,)
        
    #     data = self.params_states
    #     node_keys = self.keys()
    #     for k in ref:
    #         idx = node_keys.index(k) if k in node_keys else k
    #         data = data[idx]
    #     return data

    # def set(self, ref, value):
    #     ref = ref if isinstance(ref, tuple) else (ref,)
        
    #     data = self.params_states
    #     node_keys = self.keys()
    #     for k in ref[:-1]:
    #         idx = node_keys.index(k) if k in node_keys else k
    #         data = data[idx]
        
    #     idx = node_keys.index(ref[-1]) if ref[-1] in node_keys else ref[-1]
    #     data[idx] = value

    def __getitem__(self, index):
        view = self
        index = index if isinstance(index, tuple) else (index,)
        for i in index:
            view = view.at(i)
        return view
    
    def __iter__(self):
        yield from self.children

    def at(self, index):
        if self.children is not None:
            return self.children[index]
    
    def __repr__(self, indent = ""):
        repr_str = f"{indent}{self.key}[{self.index}]"
        if not self.is_leaf():
            repr_str += "(\n"
            repr_str += "".join([f"{sm.__repr__(indent + '    ')},\n" for sm in self]) 
            repr_str += indent + ")"
        return repr_str
    
    def __str__(self):
        return self.__repr__()
    
    def __len__(self):
        return len(self.children)
    
    def dfs(self, incl_self = True):
        if incl_self: yield self
        if not self.is_leaf():
            for node in self.children:
                yield from node.dfs()

    def bfs(self, incl_self = True):
        if incl_self: yield self
        if self.is_leaf(): return
        queue = self.children[:]
        while queue:
            node = queue.pop(0)
            yield node
            if not node.is_leaf():
                queue.extend(node.children)

    def flatten(self):
        return [n for n in self.dfs()]
