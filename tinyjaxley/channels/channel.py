from abc import abstractmethod

from ..tree import Node


class Channel(Node):
    def __init__(self, p = {}, u = {}):
        super().__init__(p = p, u = {"i": 0.0, **u})

    @abstractmethod
    def vf(self, t, u, v): return u

    @abstractmethod
    def α(self, v): return {}
    
    @abstractmethod
    def β(self, v): return {}
    
    @abstractmethod
    def init(self, t, u, v): return u