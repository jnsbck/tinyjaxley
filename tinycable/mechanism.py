from tinycable.utils import _safe_exp, vtrap


class Mechanism:
    """Pure state, parameter, derivative, and current declarations."""

    currents = ()
    states = {}
    params = {}
    density = False

    def d(self, t, s, p):
        return {}

    def i(self, t, s, p):
        return {}


class Leak(Mechanism):
    currents = ("i_leak",)
    states = {"v": None}
    params = {"Leak.g": 0.3, "eL": -54.3}
    density = True

    def i(self, t, s, p):
        return p.g * (s.v - p.eL)


class Na(Mechanism):
    currents = ("i_na",)
    states = {"Na.m": 0.0529, "Na.h": 0.5961, "v": None}
    params = {"Na.g": 120.0, "eNa": 50.0}
    density = True

    def d(self, t, s, p):
        a_m = 0.1 * vtrap(s.v + 40.0, 10.0)
        b_m = 4.0 * _safe_exp(-(s.v + 65.0) / 18.0)
        a_h = 0.07 * _safe_exp(-(s.v + 65.0) / 20.0)
        b_h = 1.0 / (1.0 + _safe_exp(-(s.v + 35.0) / 10.0))
        return {
            "m": a_m * (1.0 - s.m) - b_m * s.m,
            "h": a_h * (1.0 - s.h) - b_h * s.h,
        }

    def i(self, t, s, p):
        return p.g * s.m**3 * s.h * (s.v - p.eNa)


class K(Mechanism):
    currents = ("i_k",)
    states = {"K.n": 0.3177, "v": None}
    params = {"K.g": 36.0, "eK": -77.0}
    density = True

    def d(self, t, s, p):
        a_n = 0.01 * vtrap(s.v + 55.0, 10.0)
        b_n = 0.125 * _safe_exp(-(s.v + 65.0) / 80.0)
        return {"n": a_n * (1.0 - s.n) - b_n * s.n}

    def i(self, t, s, p):
        return p.g * s.n**4 * (s.v - p.eK)
