"""Tests for the crossover finder (guide step 11 / M8's headline result)."""

from __future__ import annotations

import numpy as np
import pytest

from src.crossover import crossover_h


def test_clean_power_laws_cross_at_the_analytic_point():
    """err1 = C1 h, err3 = C3 h^3 cross where C1 h = C3 h^3, i.e. h* = sqrt(C1/C3)."""
    C1, C3 = 0.4, 2.0
    h_star = np.sqrt(C1 / C3)
    hs = np.logspace(-3, 0, 40)
    err1 = C1 * hs
    err3 = C3 * hs**3
    crossings = crossover_h(hs, err1, hs, err3)
    assert len(crossings) == 1
    assert crossings[0] == pytest.approx(h_star, rel=1e-3)


def test_order3_wins_below_and_loses_above_the_crossing():
    C1, C3 = 0.4, 2.0
    h_star = np.sqrt(C1 / C3)
    hs = np.logspace(-3, 0, 40)
    err1 = C1 * hs
    err3 = C3 * hs**3
    (h,) = crossover_h(hs, err1, hs, err3)
    below, above = h * 0.5, h * 1.5
    assert (C3 * below**3) < (C1 * below)   # order 3 wins at smaller h
    assert (C3 * above**3) > (C1 * above)   # order 3 loses at larger h


def test_no_crossing_when_one_curve_always_wins():
    hs = np.logspace(-3, 0, 20)
    err1 = 100.0 * hs          # order 3 always better, never crosses
    err3 = 1e-6 * hs**3
    assert len(crossover_h(hs, err1, hs, err3)) == 0


def test_no_overlap_returns_empty():
    h1 = np.logspace(-3, -2, 10)
    h3 = np.logspace(1.0, 2.0, 10)
    assert len(crossover_h(h1, np.ones_like(h1), h3, np.ones_like(h3))) == 0


def test_non_monotonic_curve_reports_every_sign_change():
    """A curve that dips below and back above the other must report both
    crossings, not just the first -- exactly the failure mode a naive
    'find the first flip and stop' implementation would have on DPM-Solver-3's
    genuine non-monotonic bump at very coarse grids."""
    hs = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.01])
    err1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # dips below err1 in the middle, then rises back above, then drops below again
    err3 = np.array([5.0, 0.5, 2.0, 0.2, 0.2, 0.01])
    crossings = crossover_h(hs, err1, hs, err3)
    assert len(crossings) >= 3
    assert np.all(np.diff(crossings) > 0)  # returned ascending in h


def test_ignores_nonfinite_and_nonpositive_points():
    hs = np.array([1.0, 0.5, 0.25, 0.1, 0.05])
    err1 = np.array([1.0, 0.5, 0.25, 0.1, 0.05])
    err3 = np.array([np.nan, 5.0, 0.001, -1.0, 0.00001])
    crossings = crossover_h(hs, err1, hs, err3)
    # only the finite, positive points (h=0.5,err=5.0) and (h=0.05,err=1e-5) remain
    # usable alongside err1's five points restricted to the overlap [0.05, 0.5];
    # must not raise, and any crossing found must lie within that overlap.
    for c in crossings:
        assert 0.05 <= c <= 0.5


def test_real_solver_curves_locate_a_crossing_in_the_swept_range():
    """End-to-end sanity check on this project's own solvers: DPM-Solver-1 vs
    DPM-Solver-3 on the Tier-1 Gaussian must cross somewhere within the swept
    h range. Both are marched with the same number of *macro* steps (so they
    are compared at matched h, not matched NFE -- order 3 spends 3x the NFE
    per macro step); at 1 macro step over the whole T->t_end span, order 3's
    cubic correction is bad enough to lose to order 1 (this is the guide's
    pre-asymptotic finding directly), and it wins by 2 macro steps."""
    from src.arm_c import sample_dpm_solver
    from src.metrics import l2
    from src.testbeds import GaussianTier1

    T, T_END = 1.0, 0.2
    tb = GaussianTier1(kappa=10.0, d=3, seed=0)
    xT = tb.x_T(5)
    ref = tb.exact(xT, T, T_END)

    from src.grids import grid_lambda

    def h_of(steps_macro):
        g = grid_lambda(T, T_END, steps_macro)
        return abs(g[1] - g[0])

    macros = [1, 2, 3, 4, 6, 8, 12, 16, 24]

    h1, err1 = [], []
    for m in macros:
        xf, nfe = sample_dpm_solver(tb.eps, xT, T, T_END, order=1, steps=m)
        h1.append(h_of(m))
        err1.append(l2(xf, ref))

    h3, err3 = [], []
    for m in macros:
        xf, nfe = sample_dpm_solver(tb.eps, xT, T, T_END, order=3, steps=3 * m)
        h3.append(h_of(m))
        err3.append(l2(xf, ref))

    assert err3[0] > err1[0], "expected order 3 to lose to order 1 at 1 macro step"
    assert err3[-1] < err1[-1], "expected order 3 to win at the finest grid"

    crossings = crossover_h(h1, err1, h3, err3)
    assert len(crossings) >= 1
    lo = min(min(h1), min(h3))
    hi = max(max(h1), max(h3))
    for c in crossings:
        assert lo <= c <= hi
