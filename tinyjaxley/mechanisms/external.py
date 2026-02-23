import equinox as eqx
from jax import Array
import jax.numpy as jnp

from .mechanism import Mechanism


class Stimulus(Mechanism):
    reads: tuple[str] = ()
    writes: tuple[str] = ()
    ion: str = "ext"
    start: Array = eqx.field(converter=jnp.array)
    stop: Array = eqx.field(converter=jnp.array)
    amp: Array = eqx.field(converter=jnp.array)

    def __init__(self, start: Array = 0.0, stop: Array = 1.0, amp: Array = 1.0, name: str = None, index: Array = None):
        super().__init__(name, index)
        self.start = start
        self.stop = stop
        self.amp = amp

    def apply(self, t, u, args=None):
        return ()

    def i(self, t, u, args=None):
        return self.amp * (t >= self.start and t <= self.stop)

class Clamp(Mechanism):
    reads: tuple[str] = ()
    writes: tuple[str] = ()
    ion: str = "/" # no current

    def __init__(self, name: str = None, index: Array = None):
        super().__init__(name, index)

    def apply(self, t, u, args=None):
        return ()

    def i(self, t, u, args=None):
        return jnp.nan # TODO: nan means no clamp