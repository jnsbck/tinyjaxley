from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Field:
    """A named single-slot scalar or vector-valued pool field."""

    name: str
    default: Any
    dynamic: bool = False
    payload: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        default = np.array(self.default, copy=True)
        default.setflags(write=False)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "payload", default.shape)

    def alloc(self):
        return self.default[None].copy()
