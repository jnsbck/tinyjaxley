from __future__ import annotations

import jax
import jax.numpy as jnp


def forward_euler(rhs, y0, ts, args=None):
    def step(y, t_pair):
        t0, t1 = t_pair
        dy = rhs(t0, y, args)
        y_next = jax.tree.map(lambda yi, dyi: yi + (t1 - t0) * dyi, y, dy)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, (ts[:-1], ts[1:]))
    return jax.tree.map(
        lambda y, y_hist: jnp.concatenate([jnp.expand_dims(y, 0), y_hist]), y0, ys
    )


def integrate(model, ts, *, args=None):
    model.build()
    return forward_euler(
        lambda t, y, args: model.vf(t, y, args=args), model.u0, ts, args
    )
