from abc import ABC, abstractmethod
import jax.numpy as jnp
from ..utils import safe_exp

class Channel(ABC):
    def __init__(self, params = {}, states = {}):
        self.states = {"i": 0.0}
        self.params = {}
        self.states.update(states)
        self.params.update(params)

    @property 
    def name(self): return self.__class__.__name__.lower()

    @abstractmethod
    def __call__(self, u, p, t): return u
    
    @abstractmethod
    def init(self, u, *args): return u

def _vtrap(x, y): return x / (safe_exp(x/y) - 1.0)
def taux(v, a, b): return 1 / (a(v) + b(v))
def xinf(v, a, b): return a(v) * taux(v, a, b)