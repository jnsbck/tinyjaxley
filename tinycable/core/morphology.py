from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from .swc import section_tree, segment_swc
from .utils import dict2mapping, readonly


def _label_tree(tree: np.ndarray) -> dict[str, np.ndarray]:
    comp = np.arange(len(tree), dtype=np.int32)
    roots = tree == comp

    root = comp.copy()
    while np.any(root != tree[root]):
        root = tree[root]

    _, cell = np.unique(root, return_inverse=True)
    cell = cell.astype(np.int32)

    branch = np.full(len(tree), -1, np.int32)
    for i, seg in enumerate(section_tree(tree)):
        branch[seg[1:]] = i
        if roots[seg[0]] and branch[seg[0]] < 0:
            branch[seg[0]] = i

    # A singleton root has no section; it is its own branch.
    missing = np.flatnonzero(branch < 0)
    if len(missing):
        start = 0 if not np.any(branch >= 0) else int(branch.max()) + 1
        branch[missing] = start + np.arange(len(missing), dtype=np.int32)

    return {"cell": cell, "branch": branch, "comp": comp}


def _labels(
    labels: Mapping[str, npt.ArrayLike] | None,
    tree: np.ndarray,
) -> Mapping[str, np.ndarray]:
    result = _label_tree(tree)

    if labels is not None:
        result.update(dict(labels))

    for name, values in result.items():
        raw = np.asarray(values)
        assert raw.ndim == 1, f"{name} must be one-dimensional"
        assert len(raw) == len(tree), f"{name} must have length {len(tree)}"

        if name in {"cell", "branch", "comp"}:
            assert np.issubdtype(raw.dtype, np.integer), (
                f"{name} must be integer-valued"
            )
            unique = np.unique(raw)
            assert np.array_equal(unique, np.arange(len(unique), dtype=np.int32)), (
                f"{name} labels must be compact from zero"
            )
            result[name] = readonly(raw, dtype=np.int32)
        else:
            assert np.issubdtype(raw.dtype, np.bool_), f"{name} must be boolean-valued"
            result[name] = readonly(raw, dtype=np.bool_)

    return dict2mapping(result)


def _extend_swc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not left.size:
        return right
    if not right.size:
        return left

    shifted = np.array(right, copy=True)
    offset = int(left[:, 0].max()) + 1 - int(right[:, 0].min())
    shifted[:, 0] += offset
    shifted[shifted[:, 6] >= 0, 6] += offset
    return np.concatenate((left, shifted), axis=0)


def _init_geometry(
    ln: np.ndarray,
    rad: np.ndarray,
    area: npt.ArrayLike,
    vol: npt.ArrayLike,
    rin: npt.ArrayLike,
    rout: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill omitted cylindrical geometry and resistive-load arrays."""
    area = np.asarray(area)
    area = 2 * np.pi * rad * ln * 1e-8 if not area.size else area

    vol = np.asarray(vol)
    vol = np.pi * rad**2 * ln if not vol.size else vol

    rin = np.asarray(rin)
    rout = np.asarray(rout)
    n = len(ln)
    if (not rin.size or not rout.size) and len(rad) == n:
        load = np.zeros(n)
        np.divide(ln, 2 * np.pi * rad**2, where=rad > 0, out=load)
        if not rin.size:
            rin = load
        if not rout.size:
            rout = load
    return area, vol, rin, rout


@dataclass(frozen=True, eq=False)
class Morphology:
    tree: npt.ArrayLike

    len: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    rad: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    xyz: npt.ArrayLike = field(default_factory=lambda: np.empty((0, 3)))

    swc: npt.ArrayLike = field(default_factory=lambda: np.empty((0, 7)))

    area: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    volume: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    rin: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    rout: npt.ArrayLike = field(default_factory=lambda: np.empty(0))

    labels: Mapping[str, npt.ArrayLike] | None = None

    index: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        tree = readonly(self.tree, dtype=np.int32)
        length = readonly(self.len)
        radius = readonly(self.rad)
        xyz = readonly(self.xyz)
        n = len(tree)

        area, volume, rin, rout = _init_geometry(
            length,
            radius,
            self.area,
            self.volume,
            self.rin,
            self.rout,
        )

        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "len", length)
        object.__setattr__(self, "rad", radius)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "area", readonly(area))
        object.__setattr__(self, "volume", readonly(volume))
        object.__setattr__(self, "rin", readonly(rin))
        object.__setattr__(self, "rout", readonly(rout))
        object.__setattr__(self, "swc", readonly(self.swc))
        object.__setattr__(
            self,
            "index",
            readonly(np.arange(n, dtype=np.int32)),
        )
        object.__setattr__(self, "labels", _labels(self.labels, tree))

    @property
    def n(self) -> int:
        return len(self.index)

    def plot(
        self,
        *,
        dims: str = "xy",
        ax: Any | None = None,
        marker: str | None = None,
        kind: str = "swc",
    ) -> Any:
        from tinycable.extra.plot import plot_morphology

        return plot_morphology(self, dims=dims, ax=ax, marker=marker, kind=kind)

    @classmethod
    def from_swc(
        cls,
        swc: npt.ArrayLike,
        *,
        nseg_per_sec: int = 1,
    ) -> Self:
        tree, attrs, labels = segment_swc(swc, nseg_per_sec=nseg_per_sec)
        return cls(
            tree,
            xyz=attrs[:, :3],
            rad=attrs[:, 3],
            len=attrs[:, 4],
            area=attrs[:, 5],
            volume=attrs[:, 6],
            rin=attrs[:, 7],
            rout=attrs[:, 8],
            swc=swc,
            labels=labels,
        )

    def extend(self, other: "Morphology") -> "Morphology":
        offset = self.n
        reserved = {"comp", "branch", "cell"}
        names = [name for name in self.labels if name not in reserved]
        names.extend(
            name for name in other.labels if name not in reserved and name not in names
        )
        labels = {
            name: np.concatenate(
                (
                    self.labels.get(name, np.zeros(self.n, dtype=bool)),
                    other.labels.get(name, np.zeros(other.n, dtype=bool)),
                )
            )
            for name in names
        }
        return Morphology(
            np.concatenate((self.tree, other.tree + offset)),
            len=np.concatenate((self.len, other.len)),
            rad=np.concatenate((self.rad, other.rad)),
            xyz=np.concatenate((self.xyz, other.xyz)),
            area=np.concatenate((self.area, other.area)),
            volume=np.concatenate((self.volume, other.volume)),
            rin=np.concatenate((self.rin, other.rin)),
            rout=np.concatenate((self.rout, other.rout)),
            swc=_extend_swc(self.swc, other.swc),
            labels=labels,
        )


class Cable(Morphology):
    def __init__(
        self, n: int, *, comp_len: float = 10.0, comp_rad: float = 1.0
    ) -> None:
        assert n > 0, "Cable requires n > 0"
        tree = np.arange(n, dtype=np.int32)
        tree[1:] -= 1
        length = np.full(n, comp_len)
        radius = np.full(n, comp_rad)
        xyz = np.zeros((n, 3), dtype=float)
        xyz[:, 0] = (np.arange(n) + 0.5) * comp_len

        swc = np.zeros((n + 1, 7), dtype=float)
        swc[:, 0] = np.arange(1, n + 2)
        swc[:, 1] = 3
        swc[:, 2] = np.arange(n + 1) * comp_len
        swc[:, 5] = comp_rad
        swc[:, 6] = np.concatenate(([-1], np.arange(1, n + 1)))

        super().__init__(tree, len=length, rad=radius, xyz=xyz, swc=swc)


class Point(Cable):
    def __init__(self, *, comp_len: float = 10.0, comp_rad: float = 1.0) -> None:
        super().__init__(1, comp_len=comp_len, comp_rad=comp_rad)
