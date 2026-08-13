from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from .utils import assert_index, readonly


def _first(values: np.ndarray) -> Any:
    return values[0]


def _explicit_groups(groups: npt.ArrayLike | None, n_index: int) -> np.ndarray:
    groups = np.arange(n_index, dtype=np.int32) if groups is None else groups
    groups = readonly(groups, dtype=np.int32)
    assert groups.ndim == 1, "indices must be one-dimensional"
    assert groups.shape == (n_index,), "groups must align with index"
    unique = np.unique(groups)
    expected = np.arange(len(unique), dtype=np.int32)
    assert np.array_equal(unique, expected), "group ids must be compact from zero"
    return groups


def _locate(index: np.ndarray, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(index, requested)
    present = positions < len(index)
    present[present] &= index[positions[present]] == requested[present]
    return positions, present


def _enumerate_unique(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, first, inverse = np.unique(tokens, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int32)
    remap[order] = np.arange(len(order), dtype=np.int32)
    return remap[inverse], unique[order]


@dataclass(frozen=True, eq=False)
class Field:
    """Immutable slot storage over an optional site index."""

    key: str
    slots: npt.ArrayLike
    dynamic: bool = False
    public: bool = False
    ref: str | None = None
    index: npt.ArrayLike | None = None
    groups: npt.ArrayLike | None = None

    def __post_init__(self) -> None:
        slots = np.array(self.slots, copy=True)
        index = None
        groups = None
        ref = self.key if self.ref is None else self.ref

        if self.index is None:
            assert self.groups is None, "unplaced fields cannot define groups"
            slots = slots.reshape((1, *slots.shape))
        else:
            index = assert_index(self.index)
            groups = _explicit_groups(self.groups, len(index))
            n_slots = 0 if not len(groups) else int(groups.max()) + 1
            if slots.ndim == 0:
                slots = slots.reshape(1)
            if len(slots) == 1:
                slots = np.broadcast_to(slots, (n_slots, *slots.shape[1:])).copy()
            elif len(slots) != n_slots:
                raise ValueError(
                    f"field {self.ref!r} has {len(slots)} slot values for "
                    f"{n_slots} groups"
                )

        slots = readonly(slots)
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "groups", groups)

    def _require_index(self) -> np.ndarray:
        if self.index is None:
            raise ValueError(f"field {self.ref!r} must be inserted first")
        return self.index

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.slots.shape[1:]

    @property
    def value(self) -> np.ndarray:
        return self.slots if self.index is None else self.slots[self.groups]

    def _insert(
        self,
        index: npt.ArrayLike,
        *,
        prefix: str | None = None,
        dynamic: bool | None = None,
    ) -> Self:
        """Insert an unplaced declaration into a model support."""
        if self.index is not None:
            raise ValueError(f"field {self.ref!r} is already inserted")
        ref = self.key if self.public or prefix is None else f"{prefix}_{self.key}"
        return replace(
            self,
            ref=ref,
            index=index,
            dynamic=self.dynamic if dynamic is None else dynamic,
        )

    def set(self, value: Any, *, at: npt.ArrayLike | None = None) -> Self:
        """Return a copy with selected slot values replaced."""
        support = self._require_index()
        sites = support if at is None else assert_index(at, sorted=False)
        projection = self.slot_index(sites)

        values = np.asarray(value)
        payload = self.value_shape
        target_shape = (len(sites), *payload)
        if values.ndim == 0 or values.shape == payload:
            values = np.broadcast_to(values, target_shape)
        elif values.shape != target_shape:
            raise ValueError(
                f"values for field {self.ref!r} must be scalar or have shape "
                f"{payload} or {target_shape}, got {values.shape}"
            )

        groups, first, inverse = np.unique(
            projection,
            return_index=True,
            return_inverse=True,
        )

        members = support[np.isin(self.groups, groups)]
        if not np.array_equal(np.sort(sites), members):
            raise ValueError(f"field {self.ref!r} requires complete groups")

        if not np.array_equal(values, values[first][inverse]):
            raise ValueError(f"values for field {self.ref!r} disagree within a group")

        slots = np.array(self.slots, copy=True)
        slots[groups] = values[first]
        return replace(self, slots=slots)

    def render(
        self,
        fmt: str = "str",
        *,
        role: bool = False,
        shape: bool = False,
        slot: bool = True,
        sites: bool = True,
        filter: Any = None,
        compress: bool = False,
    ) -> str | dict[str, Any]:
        """Return a per-slot Field representation."""
        from tinycable.extra.render import field_dict, fields_dict_to_str

        if fmt not in {"str", "dict"}:
            raise ValueError(f"unknown render format {fmt!r}")
        data = field_dict(
            self,
            role=role,
            shape=shape,
            slot=slot,
            sites=sites,
            filter=filter,
            compress=compress,
        )
        if not data:
            return {} if fmt == "dict" else ""
        return data if fmt == "dict" else fields_dict_to_str({self.ref: data})

    def slot_index(self, index: npt.ArrayLike) -> np.ndarray:
        """Return one raw slot index per requested site."""
        support = self._require_index()
        requested = assert_index(index, sorted=False)
        positions, present = _locate(support, requested)
        if not np.all(present):
            missing = requested[~present][:10].tolist()
            raise ValueError(f"field {self.ref!r} does not exist at indices {missing}")
        return np.asarray(self.groups[positions], dtype=np.int32)

    def regroup(
        self,
        groups: npt.ArrayLike,
        *,
        reduce: Callable[[np.ndarray], Any] | None = None,
    ) -> Self:
        """Reduce current site values into a new slot grouping."""
        index = self._require_index()
        groups = _explicit_groups(groups, len(index))
        reduce = _first if reduce is None else reduce
        if not len(index):
            slots = np.empty((0, *self.value_shape), dtype=self.slots.dtype)
        elif reduce is _first:
            _, first = np.unique(groups, return_index=True)
            slots = self.value[first]
        else:
            order = np.argsort(groups, kind="stable")
            cuts = np.flatnonzero(np.diff(groups[order])) + 1
            values = self.value[order]
            slots = np.stack(
                [
                    np.broadcast_to(np.asarray(reduce(block)), self.value_shape)
                    for block in np.split(values, cuts)
                ]
            )
        return replace(self, slots=slots, index=index, groups=groups)

    def extend(self, other: Self, *, left_wins: bool = False) -> Self:
        """Merge supports while preserving the winning source-slot ties."""
        assert self.ref == other.ref, "extended fields must have the same ref"
        assert self.value_shape == other.value_shape, (
            "extended fields must have the same value shape"
        )
        left_index = self._require_index()
        right_index = other._require_index()
        dynamic = self.dynamic or other.dynamic

        if not len(left_index):
            return replace(other, dynamic=dynamic)
        if not len(right_index):
            return replace(self, dynamic=dynamic)

        index = np.union1d(left_index, right_index).astype(np.int32)
        left_pos, in_left = _locate(left_index, index)
        right_pos, in_right = _locate(right_index, index)
        use_left = in_left & (~in_right | left_wins)

        # Namespace source slots so equal left/right group ids never merge.
        tokens = np.empty(len(index), dtype=np.int64)
        tokens[use_left] = self.groups[left_pos[use_left]]
        tokens[~use_left] = self.n_slots + other.groups[right_pos[~use_left]]
        groups, source_slots = _enumerate_unique(tokens)

        slots = np.empty(
            (len(source_slots), *self.value_shape),
            dtype=np.result_type(self.slots, other.slots),
        )
        from_left = source_slots < self.n_slots
        slots[from_left] = self.slots[source_slots[from_left]]
        slots[~from_left] = other.slots[source_slots[~from_left] - self.n_slots]
        return replace(self, slots=slots, index=index, groups=groups, dynamic=dynamic)
