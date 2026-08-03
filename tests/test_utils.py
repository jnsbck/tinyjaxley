import jax
import jax.numpy as jnp

from tinycable import IDENTITY, Ns, gather, scatter_add
from tinycable.utils import _safe_exp


def _primitives(jaxpr):
    return {equation.primitive.name for equation in jaxpr.jaxpr.eqns}


def test_namespace():
    ns = Ns(value=3)

    # Mechanisms can use mapping or attribute access without conversion.
    assert ns["value"] == 3
    assert ns.value == 3


def test_gather():
    values = jnp.arange(4.0)

    # Full coverage bypasses indexing; partial coverage follows its projection.
    assert gather(values, IDENTITY) is values
    assert jnp.array_equal(
        gather(values, jnp.array([3, 1], dtype=jnp.int32)),
        jnp.array([3.0, 1.0]),
    )

    # A shared one-slot value broadcasts through later array operations.
    singleton = jnp.array([[1.0, 2.0]])
    assert jnp.array_equal(gather(singleton, jnp.array([0, 0])), singleton[0])

    # The identity path must not leave a gather primitive in compiled work.
    jaxpr = jax.make_jaxpr(lambda x: gather(x, IDENTITY))(values)
    assert not any("gather" in primitive for primitive in _primitives(jaxpr))


def test_scatter_add():
    out = jnp.zeros(3)
    values = jnp.array([2.0, 3.0, 4.0])

    # Full coverage adds directly; projected writes accumulate repeated targets.
    assert jnp.array_equal(scatter_add(out, IDENTITY, values), values)
    assert jnp.array_equal(
        scatter_add(out, jnp.array([1, 1, 2], dtype=jnp.int32), values),
        jnp.array([0.0, 5.0, 4.0]),
    )

    # The identity path must not leave a scatter primitive in compiled work.
    jaxpr = jax.make_jaxpr(
        lambda target, update: scatter_add(target, IDENTITY, update)
    )(out, values)
    assert not any("scatter" in primitive for primitive in _primitives(jaxpr))


def test_safe_exp():
    ordinary = jnp.array(2.0)
    extreme = jnp.array(1e6)

    # Ordinary values retain the exact exponential and its derivative.
    assert jnp.allclose(_safe_exp(ordinary), jnp.exp(ordinary))
    assert jnp.allclose(jax.grad(_safe_exp)(ordinary), jnp.exp(ordinary))

    # Extreme inputs remain finite and stop contributing an exploding gradient.
    assert jnp.isfinite(_safe_exp(extreme))
    assert jax.grad(_safe_exp)(extreme) == 0.0
