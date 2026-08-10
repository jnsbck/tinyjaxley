from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from tinycable.utils import Ns, _safe_exp, assert_index, vtrap


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
            index = assert_index(self.index)
            object.__setattr__(self, "index", index)

    def d(self, t: Any, s: Ns, p: Ns) -> dict[str, Any]:
        return {}

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return {}


class Channel(Mechanism):
    density: ClassVar[bool] = True


@dataclass(frozen=True, eq=False, init=False)
class Synapse(Mechanism):
    """Pure declaration for a directed edge mechanism."""

    pre_index: npt.ArrayLike | None = None
    post_index: npt.ArrayLike | None = None
    pre: ClassVar[tuple[str, ...]] = ()
    post: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        pre_index: npt.ArrayLike | None = None,
        post_index: npt.ArrayLike | None = None,
        name: str | None = None,
        index: npt.ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "pre_index", self._endpoint_array(pre_index))
        object.__setattr__(self, "post_index", self._endpoint_array(post_index))
        Mechanism.__post_init__(self)
        if self.pre_index is not None and self.post_index is not None:
            assert len(self.pre_index) == len(self.post_index), (
                "pre_index and post_index must have equal lengths"
            )

    @staticmethod
    def _endpoint_array(values: npt.ArrayLike | None) -> np.ndarray | None:
        if values is None:
            return None
        return assert_index(values, sorted=False)

    def d(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> dict[str, Any]:
        return {}

    def i(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> Any:
        return {}


class Exp2Syn(Synapse):
    currents: ClassVar[tuple[str, ...]] = ("i_syn",)
    states: ClassVar[dict[str, Any]] = {"Exp2Syn.g": 0.0}
    params: ClassVar[dict[str, Any]] = {
        "Exp2Syn.gmax": 1e-4,
        "Exp2Syn.tau": 2.0,
        "Exp2Syn.e": 0.0,
        "Exp2Syn.vth": -35.0,
        "Exp2Syn.k": 5.0,
    }
    pre: ClassVar[tuple[str, ...]] = ("v",)
    post: ClassVar[tuple[str, ...]] = ("v",)

    def d(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> dict[str, Any]:
        drive = 1.0 / (1.0 + _safe_exp(-(pre.v - p.vth) / p.k))
        return {"g": (drive - s.g) / p.tau}

    def i(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> Any:
        return p.gmax * s.g * (post.v - p.e)


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
