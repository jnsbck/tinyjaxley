from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Morphology:
    """Compartment topology and initial cylindrical geometry."""

    parent: np.ndarray
    len: np.ndarray = field(default_factory=lambda: np.empty(0))
    rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    swc: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))

    def __post_init__(self):
        parent = np.array(self.parent, dtype=np.int32, copy=True)
        length = np.array(self.len, copy=True)
        radius = np.array(self.rad, copy=True)
        xyz = np.array(self.xyz, copy=True)
        swc = np.array(self.swc, copy=True)
        for values in (parent, length, radius, xyz, swc):
            values.setflags(write=False)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "len", length)
        object.__setattr__(self, "rad", radius)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "swc", swc)

    @classmethod
    def single(cls, *, len=100.0, rad=10.0, xyz=(0.0, 0.0, 0.0)):
        return cls(
            np.array([0], dtype=np.int32),
            len=np.array([len]),
            rad=np.array([rad]),
            xyz=np.array([xyz]),
            swc=np.array([[1, 1, *xyz, rad, -1]], dtype=float),
        )

    @property
    def n(self):
        return len(self.parent)

    @property
    def area(self):
        """Lateral membrane area in cm^2."""
        return 2.0 * np.pi * self.rad * self.len * 1e-8

    @property
    def volume(self):
        """Cylindrical compartment volume in um^3."""
        return np.pi * self.rad**2 * self.len
