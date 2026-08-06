from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, eq=False)
class Morphology:
    """Compartment topology and initial cylindrical geometry."""

    tree: npt.ArrayLike
    len: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    rad: npt.ArrayLike = field(default_factory=lambda: np.empty(0))
    xyz: npt.ArrayLike = field(default_factory=lambda: np.empty((0, 3)))
    swc: npt.ArrayLike = field(default_factory=lambda: np.empty((0, 7)))

    def __post_init__(self) -> None:
        tree = np.array(self.tree, dtype=np.int32, copy=True)
        length = np.array(self.len, copy=True)
        radius = np.array(self.rad, copy=True)
        xyz = np.array(self.xyz, copy=True)
        swc = np.array(self.swc, copy=True)
        for vals in (tree, length, radius, xyz, swc):
            vals.setflags(write=False)
        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "len", length)
        object.__setattr__(self, "rad", radius)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "swc", swc)

    @classmethod
    def from_swc(cls, swc: npt.ArrayLike) -> Self:
        raise NotImplementedError("Morphology.from_swc is not yet implemented")

    @property
    def n(self) -> int:
        return len(self.tree)

    @property
    def area(self) -> np.ndarray:
        """Lateral membrane area in cm^2."""
        return 2.0 * np.pi * self.rad * self.len * 1e-8

    @property
    def volume(self) -> np.ndarray:
        """Cylindrical compartment volume in um^3."""
        return np.pi * self.rad**2 * self.len


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
