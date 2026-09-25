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
    factor: float = 10.0,
    ref=None,
) -> float:
    """The largest stable step size for ``stepper`` on ``tb``, arm ``"A"`` or
    ``"B"``, found by bisecting on step count between ``n_lo`` and ``n_hi``.

    Returns the ``h`` of the coarsest (smallest-``n``) grid that still stays
    bounded, or ``nan`` if even ``n_hi`` steps diverge (the method is unstable
    across the whole tested range).

    ``factor`` and ``ref`` are :func:`diverged`'s own divergence-rule knobs,
    passed straight through (M12.2: how sensitive is "stable" to the choice of
    rule?). The default (``factor=10``, ``ref=None`` meaning ``x_T``) is what
    every earlier milestone used; a stricter rule -- e.g. ``factor=2`` against
    the largest data-mode norm rather than the noise norm -- is a one-line
    sensitivity check, not a different function.
    """
    rhs = tb.rhs_t if arm == "A" else tb.rhs_lambda
    gfun = grid_t if arm == "A" else grid_lambda
    xT = tb.x_T(1)
    ref_norm = xT if ref is None else ref

    def stable(n: int) -> bool:
        x, _ = integrate(rhs, xT, gfun(T, t_end, n), stepper)
        return not diverged(x, ref_norm, factor=factor)

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
