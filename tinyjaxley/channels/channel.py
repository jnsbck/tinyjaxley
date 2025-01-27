from abc import ABC, abstractmethod

class Channel(ABC):
    def __init__(self, params = {}, states = {}):
        self.states = {"i": 0.0}
        self.params = {}
        self.states.update(states)
        self.params.update(params)

    @property 
    def name(self): return self.__class__.__name__.lower()

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