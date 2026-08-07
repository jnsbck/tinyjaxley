import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from tinycable.field import Field
from tinycable.mechanism import Mechanism, Synapse
from tinycable.morphology import Morphology
from tinycable.utils import dict2mapping


def _insert_field(fields: dict[str, Field], field: Field) -> None:
    previous = fields.get(field.name)
    fields[field.name] = field if previous is None else previous.extend(field)


def _insert_declared_fields(
    fields: dict[str, Field], mech: Mechanism, index: npt.ArrayLike
) -> None:
    for key, value in {**mech.states, **mech.params}.items():
        if value is not None:
            new_field = Field(key, value, dynamic=key in mech.states)
            _insert_field(fields, new_field.place(index))


@dataclass(frozen=True, init=False, eq=False)
class Model:
    """Immutable morphology, field, mechanism, and Synapse declarations."""

    morph: Morphology
    _fields: Mapping[str, Field]
    _mechs: Mapping[str, Mechanism]
    _syns: Mapping[str, Synapse]
    _num_syns: int

    def __init__(
        self,
        morph: Morphology,
        v: Any = -65.0,
        Ra: Any = 100.0,
        cm: Any = 1.0,
    ) -> None:
        index = morph.index
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
        object.__setattr__(self, "_fields", dict2mapping(fields))
        object.__setattr__(self, "_mechs", dict2mapping({}))
        object.__setattr__(self, "_syns", dict2mapping({}))
        object.__setattr__(self, "_num_syns", 0)

    def _with(
        self,
        *,
        fields: dict[str, Field] | None = None,
        mechs: dict[str, Mechanism] | None = None,
        syns: dict[str, Synapse] | None = None,
        num_syns: int | None = None,
    ) -> Self:
        model = object.__new__(type(self))
        object.__setattr__(model, "morph", self.morph)
        object.__setattr__(
            model,
            "_fields",
            self._fields if fields is None else dict2mapping(fields),
        )
        object.__setattr__(
            model,
            "_mechs",
            self._mechs if mechs is None else dict2mapping(mechs),
        )
        object.__setattr__(
            model,
            "_syns",
            self._syns if syns is None else dict2mapping(syns),
        )
        object.__setattr__(
            model,
            "_num_syns",
            self._num_syns if num_syns is None else num_syns,
        )
        return model

    def insert(self, item: Field | Mechanism) -> Self:
        if isinstance(item, Field):
            if item.index is None and self._syns:
                raise ValueError(
                    "unplaced Fields require an explicit index after Synapses "
                    "have been inserted"
                )
            fields = dict(self._fields)
            if item.index is None:
                item = item.place(self.morph.index)
            _insert_field(fields, item)
            return self._with(fields=fields)

        if isinstance(item, Synapse):
            assert item.index is None, (
                "Synapse edge indices are assigned by Model.insert"
            )
            assert item.pre_index is not None and item.post_index is not None, (
                "Synapse insertion requires pre_index and post_index"
            )
            assert len(item.pre_index) > 0, (
                "Synapse insertion requires at least one edge"
            )
            assert np.all((item.pre_index >= 0) & (item.pre_index < self.morph.n)), (
                "Synapse pre_index must reference model compartments"
            )
            assert np.all((item.post_index >= 0) & (item.post_index < self.morph.n)), (
                "Synapse post_index must reference model compartments"
            )
            n_new = len(item.pre_index)

            fields = dict(self._fields)
            syns = dict(self._syns)
            previous = syns.get(item.name)

            # Same-name insertions append edge instances and replace declarations.
            new_index = np.arange(
                self.morph.n + self._num_syns,
                self.morph.n + self._num_syns + n_new,
                dtype=np.int32,
            )
            if previous is not None:
                new_index = np.concatenate((previous.index, new_index))
                pre_index = np.concatenate((previous.pre_index, item.pre_index))
                post_index = np.concatenate((previous.post_index, item.post_index))
            else:
                pre_index = item.pre_index
                post_index = item.post_index

            item = replace(
                item,
                index=new_index,
                pre_index=pre_index,
                post_index=post_index,
            )
            syns[item.name] = item
            _insert_declared_fields(fields, item, item.index)
            return self._with(
                fields=fields,
                syns=syns,
                num_syns=self._num_syns + n_new,
            )

        fields = dict(self._fields)
        mechs = dict(self._mechs)
        if item.index is None:
            item = replace(item, index=self.morph.index)
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
        _insert_declared_fields(fields, item, mech_index)
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
