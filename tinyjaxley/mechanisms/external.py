import equinox as eqx
from jax import Array
import jax.numpy as jnp


class Stimulus(eqx.Module):
    name: str
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, name: str = None, index: Array = None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.index = index if index is not None else jnp.array(0)

    def __call__(self, t, u, v):
        return 0.0

    def i(self, t, u, v):
        return 0.0


class SquarePulse(Stimulus):
    value: Array = eqx.field(converter=jnp.array)
    start: Array = eqx.field(converter=jnp.array)
    end: Array = eqx.field(converter=jnp.array)

    def __init__(self, value: Array, start: Array, end: Array, name: str = None):
        super().__init__(name if name is not None else "i")
        self.start = start
        self.end = end
        self.value = value

    def __call__(self, t, u, v):
        return 0.0

    def i(self, t, u, v):
        return self.value * (t >= self.start) * (t <= self.end)


class Clamp(eqx.Module):
    name: str
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, name: str = None, index: Array = None):
        self.name = self.__class__.__name__.lower() if name is None else name
        self.index = index if index is not None else jnp.array(0)

    def __call__(self, t, u, v):
        return 0.0

    def i(self, t, u, v):
        return 0.0


class CurrentClamp(Clamp):
    value: Array = eqx.field(converter=jnp.array)
    start: Array = eqx.field(converter=jnp.array)
    end: Array = eqx.field(converter=jnp.array)

    def __init__(self, value: Array, start: Array, end: Array, name: str = None):
        super().__init__(name if name is not None else "i")
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
