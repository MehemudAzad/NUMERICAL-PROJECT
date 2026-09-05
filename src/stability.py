"""Stability envelope (guide step 8): the largest step size before a solver
diverges, swept over the testbed's condition number.

Section 4.2 of the paper claims explicit methods on this semi-linear ODE
become unstable at large step sizes near the stiff boundary ``t -> 0``, cites
the literature, and never measures it. This module measures it directly: for
a solver and a testbed, bisect on step count for the coarsest grid (largest
h) that still stays bounded.
"""

from __future__ import annotations

import numpy as np

from .grids import grid_lambda, grid_t
from .solvers import integrate

__all__ = ["diverged", "max_stable_h"]


def diverged(x, x_T, factor: float = 10.0) -> bool:
    """A run has diverged if its final state is non-finite, or has blown up
    to more than ``factor`` times the norm of the initial condition."""
    x = np.asarray(x, dtype=np.float64)
    x_T = np.asarray(x_T, dtype=np.float64)
    return (not np.all(np.isfinite(x))) or np.linalg.norm(x) > factor * np.linalg.norm(x_T)


def max_stable_h(
    tb,
    arm: str,
    stepper: str,
    T: float = 1.0,
    t_end: float = 1e-3,
    n_lo: int = 2,
    n_hi: int = 4096,
) -> float:
    """The largest stable step size for ``stepper`` on ``tb``, arm ``"A"`` or
    ``"B"``, found by bisecting on step count between ``n_lo`` and ``n_hi``.

    Returns the ``h`` of the coarsest (smallest-``n``) grid that still stays
    bounded, or ``nan`` if even ``n_hi`` steps diverge (the method is unstable
    across the whole tested range).
    """
    rhs = tb.rhs_t if arm == "A" else tb.rhs_lambda
    gfun = grid_t if arm == "A" else grid_lambda
    xT = tb.x_T(1)

    def stable(n: int) -> bool:
        x, _ = integrate(rhs, xT, gfun(T, t_end, n), stepper)
        return not diverged(x, xT)

    if not stable(n_hi):
        return np.nan

    lo, hi = n_lo, n_hi  # lo assumed unstable, hi stable
    if stable(lo):
        hi = lo
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if stable(mid):
                hi = mid
            else:
                lo = mid

    g = gfun(T, t_end, hi)
    return float(abs(g[1] - g[0]))
