from __future__ import annotations

from collections import namedtuple

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree


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

    @property
    def shape(self):
        # TODO: which shape? per comp shape, projected shape or actual shape?
        return self.value.shape


class State(Field):
    pass


class Param(Field):
    pass


class Current(Field):
    pass


class Aux(Field):
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

def insert_field(fields, new_field):
    if new_field.ref in fields:
        new_field = merge_fields(fields[new_field.ref], new_field)
    fields[new_field.ref] = new_field
    return fields


IndexMap = namedtuple("IndexMap", ("u", "i", "p"))
Kernel = namedtuple("Kernel", ("mech", "i", "vf", "reads", "writes", "area"))


def is_field(obj):
    return isinstance(obj, Field)


def gather(x, indices, keys=None):
    vals = tuple(x[i] for i in indices)
    return namedtuple("gathered", keys)(*vals) if keys is not None else vals


def scatter_add(x, x_at, indices):
    for idx, x_k in zip(indices, x_at):
        x = x.at[idx].add(x_k)
    return x


def build_kernel(mech, reads, writes, area):
    return Kernel(mech, mech.i, mech.vf, reads, writes, area)


def compute_index_maps(x, x_inds, at, reads_writes, fields_only=False):
    if fields_only:
        reads_writes = [rw for rw in reads_writes if is_field(rw)]
    if len(reads_writes) == 0:
        return namedtuple("empty", [])()

    x = x if isinstance(x, dict) else {field.ref: field for field in x}
    inds_map = {}
    for rw in reads_writes:
        key = rw.ref if is_field(rw) else rw
        inds_map[_name(rw)] = field_indices(x[key], x_inds[key], at)
    return namedtuple("Map", inds_map.keys())(*inds_map.values())


def field_indices(field, field_inds, at):
    comp_to_group = {int(i): int(g) for i, g in zip(field.index, field.group)}
    return jnp.asarray([field_inds[comp_to_group[int(i)]] for i in at])


def ravel_fields(fields):
    xs = tuple(fields.values()) if isinstance(fields, dict) else fields
    xs = {x.ref: x.value for x in xs}
    xs_flat, unravel_fn = ravel_pytree(xs)
    xs_inds = unravel_fn(jnp.arange(len(xs_flat)))
    return xs_flat, xs_inds, unravel_fn


def _ref(entry):
    return entry.ref if is_field(entry) else entry


def _name(entry):
    return entry.name if is_field(entry) else entry.rsplit(".", 1)[-1]


def safe_exp(x):
    return jnp.exp(jnp.clip(x, -100.0, 100.0))


def _vtrap(x, y):
    z = x / y
    safe_z = jnp.where(jnp.abs(z) < 1e-6, 1e-6, z)
    raw = x / jnp.expm1(safe_z)
    approx = y * (1.0 - z / 2.0)
    return jnp.where(jnp.abs(z) < 1e-6, approx, raw)
