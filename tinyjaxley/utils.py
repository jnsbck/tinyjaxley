from __future__ import annotations

import jax.numpy as jnp
from jax import Array


class Field:
    """Owned field: group has index shape and selects rows of value."""

    __slots__ = ("ref", "value", "index", "group", "train", "name")

    def __init__(
        self,
        ref: str,
        value: float | Array,
        *,
        index: int | Array = 0,
        group: int | Array = 0,
        train: bool | Array = False,
    ):
        self.ref = ref
        self.name = ref.rsplit(".", 1)[-1]
        self.index = jnp.atleast_1d(index)
        self.value = jnp.atleast_1d(value)
        self.group = jnp.broadcast_to(jnp.atleast_1d(group), self.index.shape)
        self.train = jnp.broadcast_to(jnp.asarray(train), self.index.shape)

    def set(self, **kwargs):
        data = {
            "ref": self.ref,
            "value": self.value,
            "index": self.index,
            "group": self.group,
            "train": self.train,
        }
        data.update(kwargs)
        return type(self)(**data)

    def _broadcast(self, **kwargs):
        return broadcast_field(self, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({self.ref!r}, {self.value})"


class State(Field):
    pass


class Param(Field):
    pass


def broadcast_field(field, index, group=None):
    index = jnp.atleast_1d(index)
    group = jnp.arange(len(index)) if group is None else jnp.atleast_1d(group)
    value = jnp.broadcast_to(
        field.value, (int(group.max()) + 1, *field.value.shape[1:])
    )
    return field.set(value=value, index=index, group=group)


def merge_fields(field1, field2):
    merged = {}
    for field in (field1, field2):
        for pos, comp_idx in enumerate(field.index.tolist()):
            merged[int(comp_idx)] = field.value[field.group[pos]], field.train[pos]

    index = sorted(merged)
    return field1.__class__(
        field1.ref,
        [merged[i][0] for i in index],
        index=index,
        group=jnp.arange(len(index)),
        train=[merged[i][1] for i in index],
    )


def safe_exp(x):
    return jnp.exp(jnp.clip(x, -100.0, 100.0))


def _vtrap(x, y):
    z = x / y
    safe_z = jnp.where(jnp.abs(z) < 1e-6, 1e-6, z)
    raw = x / jnp.expm1(safe_z)
    approx = y * (1.0 - z / 2.0)
    return jnp.where(jnp.abs(z) < 1e-6, approx, raw)
