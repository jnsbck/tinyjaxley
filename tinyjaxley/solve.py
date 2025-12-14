def exp_euler(
    x: ArrayLike,
    dt: float,
    x_inf: ArrayLike,
    x_tau: ArrayLike,
):
    """An exact solver for the linear dynamical system `dx = -(x - x_inf) / x_tau`."""
    exp_term = safe_exp(-dt / x_tau)
    return x * exp_term + x_inf * (1.0 - exp_term)