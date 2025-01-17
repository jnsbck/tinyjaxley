from abc import ABC, abstractmethod
import jax.numpy as jnp

class Channel(ABC):
    def __init__(self, params = {}, states = {}):
        self.states = states
        self.params = params

    @property 
    def name(self): return self.__class__.__name__.lower()

    @abstractmethod
    def i(self, u, p, t): return 0
    
    @abstractmethod
    def du(self, u, p, t): return {}
    
    @abstractmethod
    def init(self, u, p): return {}

def _vtrap(x, y): return x / (1 - jnp.exp(-x/y))
def taux(v, a, b): return 1 / (a(v) + b(v))
def xinf(v, a, b): return a(v) * taux(v, a, b)