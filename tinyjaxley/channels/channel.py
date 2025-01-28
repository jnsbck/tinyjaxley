from abc import ABC, abstractmethod
from ..tree import Node

class Channel(Node):
    def __init__(self, params = {}, states = {}):
        super().__init__(None, None)
        self._states["i"] = 0.0
        self._states.update(states)
        self._params.update(params)

    @staticmethod
    @abstractmethod
    def vf(u, p, t): return u

    @staticmethod
    @abstractmethod
    def α(v): return {}
    
    @staticmethod
    @abstractmethod
    def β(v): return {}
    
    @staticmethod
    @abstractmethod
    def init(t, u, p, v): return u