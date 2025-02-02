from __future__ import annotations

from collections import ChainMap
from typing import List

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Node(eqx.Module):
    parent: Node | None
    children: List[Node]
    index: int
    u: dict[str, Array] = eqx.field(converter=lambda x: {k: jnp.asarray([v]) for k, v in x.items()})
    p: dict[str, Array] = eqx.field(converter=lambda x: {k: jnp.asarray([v]) for k, v in x.items()})

    def __init__(self, parent = None, children = [], index = 0, u = {}, p = {}):
        self.parent = parent
        self.children = self.init_children(children)
        self.index = index
        self.u = u
        self.p = p

    @property
    def key(self):
        return self.__class__.__name__.lower()
    
    def is_leaf(self):
        if self.children is None: return True
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None

    @property
    def parents(self):
        if self.is_root(): return []
        return [self.parent] + self.parent.parents
    
    def init_children(self, children):
        init_children = []
        for i, c in enumerate(children):
            c = eqx.tree_at(lambda x: x.index, c, i)    
            c = eqx.tree_at(lambda x: x.parent, c, self)
            init_children += [c]
        return init_children
    
    def tree_map(self, f):
        if self.is_leaf(): return f(self)
        else: return [f(self), [c.tree_map(f) for c in self.children]]
    
    def tree_filter(self, f):
        if f(self): return self
        else: return [c.tree_filter(f) for c in self.children]

    @property
    def all_states(self):
        return self.tree_map(lambda x: x.u)
        
    @property
    def all_params(self):
        return self.tree_map(lambda x: x.p)
    
    @property
    def all_params_states(self):
        return self.tree_map(lambda x: ChainMap(x.u, x.p))
    
    def get(self, ref):
        def getter(ref):
            ref = ref if isinstance(ref, tuple) else (ref,)
            header, key = ref if len(ref) == 2 else (None, ref[0])
            def _get(x):
                cond = header is None or header in x.key
                if key in x.p and cond: return x.p[key]
                elif key in x.u and cond: return x.u[key]
                else: return None
            return _get
        return self.tree_map(getter(ref))

    def set(self, ref, value):
        def setter(ref, v):
            ref = ref if isinstance(ref, tuple) else (ref,)
            header, key = ref if len(ref) == 2 else (None, ref[0])
            def _set(x):
                cond = header is None or header in x.key
                if key in x.p and cond: x.p[key] = v if isinstance(v, jnp.ndarray) else jnp.asarray([v])
                elif key in x.u and cond: x.u[key] = v if isinstance(v, jnp.ndarray) else jnp.asarray([v])
            return _set 
        self.tree_map(setter(ref, value))
        
    def __getitem__(self, index):
        view = self
        index = index if isinstance(index, tuple) else (index,)
        for i in index:
            view = view.at(i)
        return view
    
    def __iter__(self):
        yield from self.children

    def at(self, index):
        if not self.is_leaf(): 
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
