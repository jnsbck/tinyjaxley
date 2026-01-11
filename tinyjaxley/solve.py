from jax import Array
from tinyjaxley.utils import safe_exp

import diffrax
import jax
import jax.numpy as jnp
from jax import vmap
from .mechanisms.channel import Channel
from .mechanisms.external import Stimulus
from .utils import is_instance_of
import jax.experimental.sparse as jsp


def exp_euler(
    x: Array,
    dt: float,
    x_inf: Array,
    x_tau: Array,
):
    """An exact solver for the linear dynamical system `dx = -(x - x_inf) / x_tau`."""
    exp_term = safe_exp(-dt / x_tau)
    return x * exp_term + x_inf * (1.0 - exp_term)


# TODO: Move to utils ?
def bcoo_to_csr(rows, cols, data, n):
    """
    Convert BCOO format (rows, cols, data) to CSR format for a square matrix.

    Args:
        rows: Array of row indices (shape: [nnz])
        cols: Array of column indices (shape: [nnz])
        data: Array of values (shape: [nnz])
        n: Matrix dimension (n x n square matrix)

    Returns:
        Tuple of (indptr, indices, data) in CSR format where:
        - indptr: Row pointer array (shape: [n + 1])
        - indices: Column indices (shape: [nnz])
        - data: Values (shape: [nnz])
    """
    nnz = len(data)

    # Sort by row index first, then column index
    sort_idx = jnp.lexsort((cols, rows))
    sorted_rows = rows[sort_idx]
    sorted_cols = cols[sort_idx]
    sorted_data = data[sort_idx]

    # Compute indptr using bincount
    # Count occurrences of each row index
    row_counts = jnp.bincount(sorted_rows, length=n)
    indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(row_counts)])

    return sorted_data, sorted_cols, indptr


class GateExpEuler(diffrax.AbstractSolver):
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        channels = terms.term.vector_field.channels
        dt = t1 - t0

        def step_channel(c):
            xinf = c.xinf(y0.get(c.name, {}), y0["v"][c.index])
            tau = c.tau(y0.get(c.name, {}), y0["v"][c.index])
            step_exp_euler = lambda x, xinf, tau: exp_euler(x, dt, xinf, tau)
            return jax.tree.map(step_exp_euler, y0.get(c.name, {}), xinf, tau)

        y1 = jax.tree.map(step_channel, channels, is_leaf=is_instance_of(Channel))
        return y1, None, dict(y0=y0, y1=y1), None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        channels = terms.term.vector_field.channels
        channel_vf = lambda c: c(t0, y0.get(c.name, {}), y0["v"][c.index])
        return jax.tree.map(channel_vf, channels, is_leaf=is_instance_of(Channel))


class VoltageBackwardEuler(diffrax.AbstractSolver):
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        hh = terms.term.vector_field
        dt = t1 - t0
        y0_channels, y1_channels = args

        def split_linear_terms(acc, c):
            lin, const = acc
            conds = c.g(y1_channels[c.name])
            return (lin.at[c.index].add(-conds), const.at[c.index].add(conds * c.e))

        lin, const = jax.tree.reduce(
            split_linear_terms,
            hh.channels,
            (jnp.zeros(hh.num_comps), jnp.zeros(hh.num_comps)),
            is_leaf=is_instance_of(Channel),
        )
        lin *= 1e3 / hh.c
        const *= 1e3 / hh.c

        def sum_i_ext(i0, stim):
            i = stim.i(t0, y0_channels, y0[stim.index])
            return i0.at[stim.index].add(i)

        i_ext = jax.tree.reduce(
            sum_i_ext,
            hh.stimuli,
            jnp.zeros(hh.num_comps),
            is_leaf=is_instance_of(Stimulus),
        )
        i_ext_total = i_ext * 1e5 / hh.area / hh.c

        i, j = hh.edges.T
        g_ij = vmap(hh.g_coupling)(i, j)
        inds_diag = jnp.arange(hh.num_comps)[:, None].repeat(2, axis=1)

        # Diagonal: negative sum of all conductances (edges contain both directions)
        g_ii = -jnp.bincount(i, weights=g_ij, length=hh.num_comps)

        # Build the matrix: (1 - dt*L - dt*G)
        A_ii = jnp.ones(hh.num_comps) - dt * lin - dt * g_ii
        A_ij = -dt * g_ij

        A = jnp.concatenate([A_ii, A_ij])
        inds = jnp.concatenate([inds_diag, hh.edges])
        lhs = bcoo_to_csr(*inds.T, A, hh.num_comps)

        # Right-hand side
        rhs = y0 + dt * const + dt * i_ext_total

        # Solve: (1 - dt*L - dt*G) @ v_{t+1} = v_t - dt*C - dt*i_ext/C
        y1 = jsp.linalg.spsolve(*lhs, rhs, tol=1e-6)
        return y1, None, dict(y0=y0, y1=y1), None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)["v"]


class ForwardEuler(diffrax.AbstractSolver):
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation
    gate_solver: GateExpEuler = GateExpEuler()

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        hh = terms.term.vector_field
        dt = t1 - t0

        # step gates
        y1, *_ = self.gate_solver.step(terms, t0, t1, y0, args, None, made_jump)

        # compute currents
        i_total = hh.compute_i_total(t0, y0, y0["v"])

        i, j = hh.edges.T
        dv0_i = i_total / hh.c
        dv0_ij = vmap(hh.g_coupling)(i, j) * (y0["v"][j] - y0["v"][i])
        dv0 = dv0_i + jnp.bincount(i, weights=dv0_ij, length=len(dv0_i))

        y1["v"] = y0["v"] + dt * dv0

        # no error estimate
        y_error = None
        # Dense info for linear interpolation
        dense_info = dict(y0=y0, y1=y1)
        # Solver state remains None
        new_solver_state = None
        # Result: successful step
        solver_result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, new_solver_state, solver_result

    def func(self, terms, t0, y0, args):
        """Return the vector field (for use in other contexts)."""
        return terms.vf(t0, y0, args)


class BackwardEuler(diffrax.AbstractSolver):
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation
    gate_solver: GateExpEuler = GateExpEuler()
    voltage_solver: VoltageBackwardEuler = VoltageBackwardEuler()

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        # step gates
        y1, *_ = self.gate_solver.step(terms, t0, t1, y0, args, None, made_jump)

        # step voltage
        v0 = y0["v"]
        v1, *_ = self.voltage_solver.step(terms, t0, t1, v0, (y0, y1), None, made_jump)
        y1["v"] = v1

        # no error estimate
        y_error = None
        # Dense info for linear interpolation
        dense_info = dict(y0=y0, y1=y1)
        # Solver state remains None
        new_solver_state = None
        # Result: successful step
        solver_result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, new_solver_state, solver_result

    def func(self, terms, t0, y0, args):
        """Return the vector field (for use in other contexts)."""
        return terms.vf(t0, y0, args)


class AdaptiveBackwardEuler(diffrax.AbstractAdaptiveSolver):
    """Backward Euler with embedded error estimation for adaptive timestepping."""

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation
    gate_solver: GateExpEuler = GateExpEuler()
    voltage_solver: VoltageBackwardEuler = VoltageBackwardEuler()

    def order(self, terms):
        return 1

    def error_order(self, terms):
        return 2

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0

        # Full step with backward Euler
        y1_gates, *_ = self.gate_solver.step(terms, t0, t1, y0, args, None, made_jump)
        v0 = y0["v"]
        v1, *_ = self.voltage_solver.step(
            terms, t0, t1, v0, (y0, y1_gates), None, made_jump
        )
        y1_gates["v"] = v1

        # Half-step approximation for error estimation
        t_mid = t0 + dt / 2.0

        # First half-step
        y_mid_gates, *_ = self.gate_solver.step(
            terms, t0, t_mid, y0, args, None, made_jump
        )
        v_mid, *_ = self.voltage_solver.step(
            terms, t0, t_mid, v0, (y0, y_mid_gates), None, made_jump
        )
        y_mid_gates["v"] = v_mid

        # Second half-step
        y1_half_gates, *_ = self.gate_solver.step(
            terms, t_mid, t1, y_mid_gates, args, None, made_jump
        )
        v1_half, *_ = self.voltage_solver.step(
            terms, t_mid, t1, v_mid, (y_mid_gates, y1_half_gates), None, made_jump
        )
        y1_half_gates["v"] = v1_half

        # Error estimate: difference between full step and two half-steps
        # For backward Euler with embedded half-stepping, error scales as O(dt^2)
        def compute_error(leaf1, leaf_half):
            return jnp.abs(leaf1 - leaf_half)

        y_error = jax.tree.map(compute_error, y1_gates, y1_half_gates)

        # Use the more accurate half-step solution
        y1 = y1_half_gates

        # Dense info for interpolation
        dense_info = dict(y0=y0, y1=y1)

        return y1, y_error, dense_info, None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        """Return the vector field."""
        return terms.vf(t0, y0, args)
