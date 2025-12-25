import equinox as eqx
from jax import Array
import jax.numpy as jnp

from .mechanism import Mechanism


class Stimulus(Mechanism):
    def __init__(self, name: str = None, index: Array = None):
        name = "i" if name is None else name
        index = jnp.array(0) if index is None else index
        super().__init__(name, index)


class Clamp(Mechanism):
    def __init__(self, name: str = None, index: Array = None):
        name = "v" if name is None else name
        index = jnp.array(0) if index is None else index
        super().__init__(name, index)


class SquarePulse(Stimulus):
    value: Array = eqx.field(converter=jnp.array)
    start: Array = eqx.field(converter=jnp.array)
    end: Array = eqx.field(converter=jnp.array)

    def __init__(self, value: Array, start: Array, end: Array, name: str = None):
        super().__init__(name)
        self.start = start
        self.end = end
        self.value = value

    def __call__(self, t, u, v):
        return 0.0

    def i(self, t, u, v):
        return self.value * (t >= self.start) * (t <= self.end)


class CurrentClamp(Clamp):
    value: Array = eqx.field(converter=jnp.array)
    start: Array = eqx.field(converter=jnp.array)
    end: Array = eqx.field(converter=jnp.array)

    def __init__(self, value: Array, start: Array, end: Array, name: str = None):
        super().__init__(name)
        self.value = value

    def __call__(self, t, u, v):
        return 0.0

    def i(self, t, u, v):
        return self.value * (t >= self.start) * (t <= self.end)


class VoltageClamp(Clamp):
    value: Array = eqx.field(converter=jnp.array)
    start: Array = eqx.field(converter=jnp.array)
    end: Array = eqx.field(converter=jnp.array)

    def __init__(self, value: Array, start: Array, end: Array, name: str = None):
        super().__init__(name if name is not None else "v")
        self.value = value

    def __call__(self, t, u, v):
        return self.value * (t >= self.start) * (t <= self.end)

    def i(self, t, u, v):
        return 0.0
