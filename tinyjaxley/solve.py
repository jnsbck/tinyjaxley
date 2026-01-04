from jax import Array
from tinyjaxley.utils import safe_exp

import diffrax
import jax
import jax.numpy as jnp
from jax import vmap
from tinyjaxley.mechanisms.channel import Channel
from tinyjaxley.mechanisms.external import Stimulus
from tinyjaxley.solve import exp_euler
from tinyjaxley.utils import is_instance_of
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


class ForwardEuler(diffrax.AbstractSolver):
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def _step_gates(self, terms, t0, t1, y0):
        hh = terms.term.vector_field
        dt = t1 - t0

        def step_channel(c):
            xinf = c.xinf(y0.get(c.name, {}), y0["v"][c.index])
            tau = c.tau(y0.get(c.name, {}), y0["v"][c.index])
            step_exp_euler = lambda x, xinf, tau: exp_euler(x, dt, xinf, tau)
            return jax.tree.map(step_exp_euler, y0.get(c.name, {}), xinf, tau)

        y1 = jax.tree.map(step_channel, hh.channels, is_leaf=is_instance_of(Channel))
        return y1

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        hh = terms.term.vector_field

        dt = t1 - t0

        # Forward Euler: y1 = y0 + dt * dy0
        # step gates
        y1 = self._step_gates(terms, t0, t1, y0)

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

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def _step_gates(self, terms, t0, t1, y0):
        hh = terms.term.vector_field
        dt = t1 - t0

        def step_channel(c):
            xinf = c.xinf(y0.get(c.name, {}), y0["v"][c.index])
            tau = c.tau(y0.get(c.name, {}), y0["v"][c.index])
            step_exp_euler = lambda x, xinf, tau: exp_euler(x, dt, xinf, tau)
            return jax.tree.map(step_exp_euler, y0.get(c.name, {}), xinf, tau)

        y1 = jax.tree.map(step_channel, hh.channels, is_leaf=is_instance_of(Channel))
        return y1

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        hh = terms.term.vector_field

        dt = t1 - t0

        # step gates
        y1 = self._step_gates(terms, t0, t1, y0)

        def split_linear_terms(acc, c):
            lin, const = acc
            conds = c.g(y1[c.name])
            return (lin.at[c.index].add(-conds), const.at[c.index].add(conds * c.e))

        lin, const = jax.tree.reduce(
            split_linear_terms,
            hh.channels,
            (jnp.zeros(hh.num_comps), jnp.zeros(hh.num_comps)),
            is_leaf=is_instance_of(Channel),
        )
        lin *= 1e3 / hh.c
        const *= 1e3 / hh.c

        # Compute ONLY external stimulus currents
        def sum_i_ext(i0, stim):
            i = stim.i(t0, y0, y0["v"][stim.index])
            return i0.at[stim.index].add(i)

        i_ext = jax.tree.reduce(
            sum_i_ext,
            hh.stimuli,
            jnp.zeros(hh.num_comps),
            is_leaf=is_instance_of(Stimulus),
        )
        i_ext_total = i_ext * 1e5 / hh.area / hh.c  # Divide by capacitance

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
        rhs = y0["v"] - dt * const - dt * i_ext_total

        # Solve: (1 - dt*L - dt*G) @ v_{t+1} = v_t - dt*C - dt*i_ext/C
        y1["v"] = jsp.linalg.spsolve(*lhs, rhs, tol=1e-6)

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
