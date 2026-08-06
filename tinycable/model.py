import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from tinycable.field import Field
from tinycable.mechanism import Mechanism
from tinycable.morphology import Morphology


def _insert_field(fields: dict[str, Field], field: Field) -> None:
    previous = fields.get(field.name)
    fields[field.name] = field if previous is None else previous.extend(field)


@dataclass(frozen=True, init=False, eq=False)
class Model:
    """Immutable morphology, field, and mechanism declarations."""

    morph: Morphology
    _fields: Mapping[str, Field]
    _mechs: Mapping[str, Mechanism]

    def __init__(
        self,
        morph: Morphology,
        v: Any = -65.0,
        Ra: Any = 100.0,
        cm: Any = 1.0,
    ) -> None:
        index = np.arange(morph.n, dtype=np.int32)
        fields = {
            "v": Field("v", v, dynamic=True, index=index),
            "Ra": Field("Ra", Ra, index=index),
            "cm": Field("cm", cm, index=index),
            "rad": Field("rad", morph.rad, index=index),
            "len": Field("len", morph.len, index=index),
        }
        assert all(field.value_shape == () for field in fields.values()), (
            "model base fields must be scalar-valued"
        )
        object.__setattr__(self, "morph", morph)
        object.__setattr__(self, "_fields", MappingProxyType(fields))
        object.__setattr__(self, "_mechs", MappingProxyType({}))

    def _with(
        self,
        *,
        fields: dict[str, Field] | None = None,
        mechs: dict[str, Mechanism] | None = None,
    ) -> Self:
        model = object.__new__(type(self))
        object.__setattr__(model, "morph", self.morph)
        object.__setattr__(
            model,
            "_fields",
            self._fields if fields is None else MappingProxyType(fields),
        )
        object.__setattr__(
            model,
            "_mechs",
            self._mechs if mechs is None else MappingProxyType(mechs),
        )
        return model

    def insert(self, item: Field | Mechanism) -> Self:
        if isinstance(item, Field):
            fields = dict(self._fields)
            if item.index is None:
                item = item.place(np.arange(self.morph.n, dtype=np.int32))
            _insert_field(fields, item)
            return self._with(fields=fields)

        fields = dict(self._fields)
        mechs = dict(self._mechs)
        if item.index is None:
            item = replace(item, index=np.arange(self.morph.n, dtype=np.int32))
        assert np.all((item.index >= 0) & (item.index < self.morph.n)), (
            "mechanism index must reference model compartments"
        )
        previous = mechs.get(item.name)
        # Name alone selects the instance set; a different type may overwrite it.
        mech_index = (
            np.union1d(previous.index, item.index).astype(np.int32)
            if previous is not None
            else item.index
        )
        item = replace(item, index=mech_index)
        mechs[item.name] = item

        for k, v in {**item.states, **item.params}.items():
            if v is not None:
                new_field = Field(k, v, dynamic=k in item.states)
                _insert_field(fields, new_field.place(mech_index))
        return self._with(fields=fields, mechs=mechs)

    def share(
        self,
        name: str,
        groups: npt.ArrayLike | None = None,
        *,
        reduce: Callable[[np.ndarray], Any] | None = None,
    ) -> Self:
        if name not in self._fields:
            raise ValueError(f"unknown field {name!r}")
        fields = dict(self._fields)
        field = fields[name]
        if groups is None:
            groups = np.zeros(len(field.index), dtype=np.int32)
        fields[name] = field.regroup(groups, reduce=reduce)
        return self._with(fields=fields)

    def unshare(self, name: str) -> Self:
        if name not in self._fields:
            raise ValueError(f"unknown field {name!r}")
        groups = np.arange(len(self._fields[name].index), dtype=np.int32)
        return self.share(name, groups)

    def remove(self, name: str) -> Self:
        if name not in self._mechs:
            return self

        warnings.warn(
            f"Removing mechanism {name!r} leaves its declared fields in the model.",
            UserWarning,
            stacklevel=2,
        )
        mechs = dict(self._mechs)
        del mechs[name]
        # TODO: Track field ownership so removing or replacing a mechanism can
        # remove stale fields and support contributed only by that mechanism.
        return self._with(mechs=mechs)
