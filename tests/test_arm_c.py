"""Tests for the arm-C wrapper around the vendored DPM-Solver (guide step 6).

``sample_dpm_solver`` order=1 must reproduce the hand-written
:func:`src.dpm.dpm_solver_1` on the same grid -- if it doesn't, the wrapper
(schedule construction, model_fn plumbing, or settings) is wrong, independent
of whether the vendored solver itself is correct. Orders 2 and 3 are checked
against their textbook convergence rate, same as Gate G1 does for order 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.arm_c import sample_dpm_solver
from src.dpm import dpm_solver_1, march
from src.grids import grid_lambda
from src.schedule import t_of_lmbda
from src.testbeds import GaussianTier1

T, T_END = 1.0, 0.2
KAPPA = 10.0


def _slope(ns, errs) -> float:
    hs = 1.0 / np.asarray(ns, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    A = np.vstack([np.log(hs), np.ones_like(hs)]).T
    slope, _ = np.linalg.lstsq(A, np.log(errs), rcond=None)[0]
    return float(slope)


def test_wrapper_order_1_matches_hand_written_dpm_solver_1():
    tb = GaussianTier1(kappa=KAPPA, d=4, seed=0)
    xT = tb.x_T(5)
    n = 32

    xf_wrap, nfe = sample_dpm_solver(tb.eps, xT, T, T_END, order=1, steps=n)
    assert nfe == n

    t_grid = t_of_lmbda(grid_lambda(T, T_END, n))
    t_grid[0], t_grid[-1] = T, T_END
    xf_hand = march(dpm_solver_1, tb.eps, xT, t_grid)

    assert np.max(np.abs(xf_wrap - xf_hand)) < 1e-10


@pytest.mark.parametrize(
    "order,steps_list,tol",
    [
        (1, [16, 32, 64, 128], 0.15),
        (2, [16, 32, 64, 128], 0.2),
        (3, [18, 36, 72, 144], 0.25),
    ],
)
def test_wrapper_hits_textbook_order(order, steps_list, tol):
    tb = GaussianTier1(kappa=KAPPA, d=3, seed=0)
    xT = tb.x_T(5)
    ref = tb.exact(xT, T, T_END)

    errs = []
    for steps in steps_list:
        xf, nfe = sample_dpm_solver(tb.eps, xT, T, T_END, order=order, steps=steps)
        assert nfe == (steps // order) * order
        errs.append(float(np.sqrt(np.mean((xf - ref) ** 2))))

    assert abs(_slope(steps_list, errs) - order) < tol, (order, errs)


def test_real_nfe_is_steps_floor_div_order_times_order():
    tb = GaussianTier1(kappa=KAPPA, d=2, seed=1)
    xT = tb.x_T(2)
    for order in (1, 2, 3):
        for steps in (10, 11, 12, 13):
            _, nfe = sample_dpm_solver(tb.eps, xT, T, T_END, order=order, steps=steps)
            assert nfe == (steps // order) * order
