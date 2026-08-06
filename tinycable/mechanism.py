from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from tinycable.utils import Ns, _safe_exp, vtrap


@dataclass(frozen=True, eq=False)
class Mechanism:
    """Pure state, parameter, derivative, and current declarations."""

    name: str | None = None
    index: npt.ArrayLike | None = field(default=None, compare=False)
    currents: ClassVar[tuple[str, ...]] = ()
    states: ClassVar[dict[str, Any]] = {}
    params: ClassVar[dict[str, Any]] = {}
    density: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", type(self).__name__.lower())
        if self.index is not None:
            index = np.array(self.index, dtype=np.int32, copy=True)
            assert index.ndim == 1, "index must be one-dimensional"
            assert np.all(index[1:] > index[:-1]), (
                "index must be sorted and deduplicated"
            )
            index.setflags(write=False)
            object.__setattr__(self, "index", index)

    def d(self, t: Any, s: Ns, p: Ns) -> dict[str, Any]:
        return {}

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return {}


class Channel(Mechanism):
    density: ClassVar[bool] = True


class Leak(Channel):
    currents: ClassVar[tuple[str, ...]] = ("i_leak",)
    states: ClassVar[dict[str, Any]] = {"v": None}
    params: ClassVar[dict[str, Any]] = {"Leak.g": 0.3, "eL": -54.3}

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return p.g * (s.v - p.eL)


class Na(Channel):
    currents: ClassVar[tuple[str, ...]] = ("i_na",)
    states: ClassVar[dict[str, Any]] = {
        "Na.m": 0.0529,
        "Na.h": 0.5961,
        "v": None,
    }
    params: ClassVar[dict[str, Any]] = {"Na.g": 120.0, "eNa": 50.0}

    def d(self, t: Any, s: Ns, p: Ns) -> dict[str, Any]:
        a_m = 0.1 * vtrap(s.v + 40.0, 10.0)
        b_m = 4.0 * _safe_exp(-(s.v + 65.0) / 18.0)
        a_h = 0.07 * _safe_exp(-(s.v + 65.0) / 20.0)
        b_h = 1.0 / (1.0 + _safe_exp(-(s.v + 35.0) / 10.0))
        return {
            "m": a_m * (1.0 - s.m) - b_m * s.m,
            "h": a_h * (1.0 - s.h) - b_h * s.h,
        }

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return p.g * s.m**3 * s.h * (s.v - p.eNa)


class K(Channel):
    currents: ClassVar[tuple[str, ...]] = ("i_k",)
    states: ClassVar[dict[str, Any]] = {"K.n": 0.3177, "v": None}
    params: ClassVar[dict[str, Any]] = {"K.g": 36.0, "eK": -77.0}

    def d(self, t: Any, s: Ns, p: Ns) -> dict[str, Any]:
        a_n = 0.01 * vtrap(s.v + 55.0, 10.0)
        b_n = 0.125 * _safe_exp(-(s.v + 65.0) / 80.0)
        return {"n": a_n * (1.0 - s.n) - b_n * s.n}

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return p.g * s.n**4 * (s.v - p.eK)
