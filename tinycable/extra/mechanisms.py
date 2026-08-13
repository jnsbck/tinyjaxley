from typing import Any, ClassVar

from tinycable.core.field import Field
from tinycable.core.mechanism import Channel, Synapse
from tinycable.core.utils import Ns, _safe_exp, vtrap


class Exp2Syn(Synapse):
    currents: ClassVar[tuple[str, ...]] = ("i_syn",)
    states = (Field("g", 0.0),)
    params = (
        Field("gmax", 1e-4),
        Field("tau", 2.0),
        Field("e", 0.0),
        Field("vth", -35.0),
        Field("k", 5.0),
    )
    pre: ClassVar[tuple[str, ...]] = ("v",)
    post: ClassVar[tuple[str, ...]] = ("v",)

    def d(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> dict[str, Any]:
        drive = 1.0 / (1.0 + _safe_exp(-(pre.v - p.vth) / p.k))
        return {"g": (drive - s.g) / p.tau}

    def i(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> Any:
        return p.gmax * s.g * (post.v - p.e)


class Leak(Channel):
    currents: ClassVar[tuple[str, ...]] = ("i_leak",)
    states = ("v",)
    params = (Field("g", 0.3), Field("el", -54.3, public=True))

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return p.g * (s.v - p.el)


class Na(Channel):
    currents: ClassVar[tuple[str, ...]] = ("i_na",)
    states = ("v", Field("m", 0.0529), Field("h", 0.5961))
    params = (Field("g", 120.0), Field("ena", 50.0, public=True))

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
        return p.g * s.m**3 * s.h * (s.v - p.ena)


class K(Channel):
    currents: ClassVar[tuple[str, ...]] = ("i_k",)
    states = ("v", Field("n", 0.3177))
    params = (Field("g", 36.0), Field("ek", -77.0, public=True))

    def d(self, t: Any, s: Ns, p: Ns) -> dict[str, Any]:
        a_n = 0.01 * vtrap(s.v + 55.0, 10.0)
        b_n = 0.125 * _safe_exp(-(s.v + 65.0) / 80.0)
        return {"n": a_n * (1.0 - s.n) - b_n * s.n}

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return p.g * s.n**4 * (s.v - p.ek)
