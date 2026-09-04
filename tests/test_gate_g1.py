"""Gate G1 (guide step 6) -- the critical checkpoint before building past arm C.

1. dpm_solver_1 and ddim_step must be the *same* algebraic identity (paper eq
   4.1), marched over a shared lambda-grid: max|xa-xb| < 1e-14. If this fails,
   lambda handling is broken -- stop and debug before anything else.
2. dpm_solver_1 must measure as first order on Tier-1. If this fails, the
   error-measurement machinery (not the solver) is broken.
"""

from __future__ import annotations

import numpy as np

from src.dpm import ddim_step, dpm_solver_1, march
from src.grids import grid_lambda
from src.schedule import t_of_lmbda
from src.testbeds import GaussianTier1


def _t_grid(T, t_end, n):
    """A lambda-uniform t-grid, with the endpoints snapped exactly to T/t_end
    rather than left as t_of_lmbda(lmbda(T)) -- an unnecessary round-trip
    error that has nothing to do with the two solvers under test."""
    t_grid = t_of_lmbda(grid_lambda(T, t_end, n))
    t_grid[0], t_grid[-1] = T, t_end
    return t_grid


def test_gate_g1_dpm_solver_1_equals_ddim():
    tb = GaussianTier1(kappa=100.0, d=4, seed=0)
    xT = tb.x_T(5)
    T, t_end = 1.0, 1e-3
    t_grid = _t_grid(T, t_end, 40)

    xa = march(dpm_solver_1, tb.eps, xT, t_grid)
    xb = march(ddim_step, tb.eps, xT, t_grid)
    assert np.max(np.abs(xa - xb)) < 1e-14


def test_gate_g1_dpm_solver_1_is_first_order():
    tb = GaussianTier1(kappa=10.0, d=3, seed=0)
    xT = tb.x_T(5)
    T, t_end = 1.0, 0.2
    ref = tb.exact(xT, T, t_end)

    ns = [16, 32, 64, 128]
    errs = []
    for n in ns:
        t_grid = _t_grid(T, t_end, n)
        xf = march(dpm_solver_1, tb.eps, xT, t_grid)
        errs.append(np.sqrt(np.mean((xf - ref) ** 2)))

    hs = 1.0 / np.asarray(ns, dtype=np.float64)
    A = np.vstack([np.log(hs), np.ones_like(hs)]).T
    slope, _ = np.linalg.lstsq(A, np.log(errs), rcond=None)[0]
    assert 0.85 <= slope <= 1.15, (slope, errs)
