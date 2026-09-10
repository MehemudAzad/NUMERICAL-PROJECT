"""Arm A/B one-step explicit solvers, Adams-Bashforth 2, and the shared integrator
(guide step 3).

All steppers share one signature, ``stepper(rhs, x, a, b) -> x_next``, where
``rhs(x, node) -> dx`` is either ``pf_rhs_t`` (arm A, node = t) or
``pf_rhs_lambda`` (arm B, node = lambda) from :mod:`src.testbeds` -- the same
function marches both arms, just fed a different grid and rhs.

``midpoint``, ``heun3`` are exactly the paper's Appendix E.4 schemes (explicit
midpoint for RK2; Heun's third-order with r1=1/3, r2=2/3, mirroring
DPM-Solver-3's default intermediate points) so arm-A/B error curves are
comparable to arm C's. ``ab2`` (Adams-Bashforth 2, order 2, 1 NFE/step after a
1-step RK2 startup) does not fit the one-step signature -- it is a multistep
method with history -- so :func:`integrate` handles it as a special branch
rather than through :data:`STEPPERS`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["euler", "midpoint", "heun3", "rk4", "STEPPERS", "NFE_PER_STEP", "integrate"]


def euler(rhs, x, a, b):
    """Forward Euler, order 1."""
    h = b - a
    return x + h * rhs(x, a)


def midpoint(rhs, x, a, b):
    """Explicit midpoint (RK2), order 2."""
    h = b - a
    k1 = rhs(x, a)
    k2 = rhs(x + 0.5 * h * k1, a + 0.5 * h)
    return x + h * k2


def heun3(rhs, x, a, b):
    """Heun's third-order method (nodes 0, 1/3, 2/3), order 3."""
    h = b - a
    k1 = rhs(x, a)
    k2 = rhs(x + (h / 3.0) * k1, a + h / 3.0)
    k3 = rhs(x + (2.0 * h / 3.0) * k2, a + 2.0 * h / 3.0)
    return x + h * (0.25 * k1 + 0.75 * k3)


def rk4(rhs, x, a, b):
    """Classic 4th-order Runge-Kutta, order 4."""
    h = b - a
    k1 = rhs(x, a)
    k2 = rhs(x + 0.5 * h * k1, a + 0.5 * h)
    k3 = rhs(x + 0.5 * h * k2, a + 0.5 * h)
    k4 = rhs(x + h * k3, a + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


STEPPERS = {"euler": euler, "midpoint": midpoint, "heun3": heun3, "rk4": rk4}

# Reference/sanity only -- integrate() returns the *measured* NFE, this is what
# it should equal for a non-diverged run.
NFE_PER_STEP = {"euler": 1, "midpoint": 2, "heun3": 3, "rk4": 4, "ab2": 1}


# --- backend-agnostic state helpers ----------------------------------------
#
# Tiers 1-2 march float64 numpy arrays; Tier 3 (guide step 10) marches a float32
# CUDA torch tensor of shape (n, 3, 32, 32). The four steppers above already
# work on both -- they are pure arithmetic, and `np.float64 * torch.Tensor`
# returns a `torch.Tensor` with the tensor's dtype and device preserved -- so
# only these two spots in `integrate` needed generalising. Both duck-type on a
# method numpy arrays do not have, which keeps this module numpy-only: it must
# never import torch (Tiers 1-2 run on laptops with no CUDA at all).


def _copy_state(x0):
    """Copy an initial state without forcing it into float64 numpy.

    Tier 3's float32 dtype and CUDA device must survive: float32 *is* the
    measurement floor there (guide step 10), and silently promoting to float64
    would both change the physics being measured and blow up GPU memory.
    """
    if hasattr(x0, "clone"):  # torch.Tensor
        return x0.clone()
    return np.array(x0, dtype=np.float64, copy=True)


def _all_finite(xv) -> bool:
    """True if every entry is finite, for numpy arrays or torch tensors."""
    if hasattr(xv, "isfinite"):  # torch.Tensor; numpy arrays have no such method
        return bool(xv.isfinite().all())
    return bool(np.all(np.isfinite(xv)))


def integrate(rhs, x0, grid, stepper):
    """March ``rhs`` along ``grid`` and return ``(x_final, nfe)``.

    ``stepper`` is a key of :data:`STEPPERS`, or the string ``"ab2"``.

    ``nfe`` is *measured* (every call to ``rhs`` is counted through a wrapper),
    not assumed -- so AB2's cached-``f_prev`` savings and any partial work
    before a divergence fall out automatically rather than needing separate
    bookkeeping. If any state goes non-finite, stops immediately and returns
    ``nfe = nan`` (the run is diverged; a partial NFE count is not meaningful).
    """
    grid = np.asarray(grid, dtype=np.float64)
    x = _copy_state(x0)
    n = len(grid) - 1

    calls = 0

    def counted(xv, node):
        nonlocal calls
        calls += 1
        return rhs(xv, node)

    def diverged(xv) -> bool:
        return not _all_finite(xv)

    if stepper == "ab2":
        if n == 0:
            return x, calls
        # Startup: one explicit-midpoint (RK2) step for x_1. Its local error is
        # O(h^3), so it can't contaminate the order-2 AB2 slope. f0 = k1 of this
        # step is exactly f(x_0, node_0), so it's cached and reused as the AB2
        # recurrence's f_{n-1} for the very first AB2 step.
        h0 = grid[1] - grid[0]
        f0 = counted(x, grid[0])
        x_mid = x + 0.5 * h0 * f0
        f_mid = counted(x_mid, grid[0] + 0.5 * h0)
        x = x + h0 * f_mid
        if diverged(x):
            return x, np.nan
        f_prev = f0
        for i in range(1, n):
            h = grid[i + 1] - grid[i]
            f_cur = counted(x, grid[i])
            x = x + h * (1.5 * f_cur - 0.5 * f_prev)
            f_prev = f_cur
            if diverged(x):
                return x, np.nan
        return x, calls

    step_fn = STEPPERS[stepper]
    for i in range(n):
        x = step_fn(counted, x, grid[i], grid[i + 1])
        if diverged(x):
            return x, np.nan
    return x, calls
