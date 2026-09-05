"""Tests for the sliding-window order fitter (guide step 7)."""

from __future__ import annotations

import numpy as np

from src.metrics import fit_order, l2


def test_fit_order_recovers_known_slope_on_a_pure_power_law():
    hs = 0.5 ** np.arange(1, 9, dtype=np.float64)
    errs = 3.7 * hs**2  # a clean order-2 curve, no noise, no floor
    fit = fit_order(hs, errs)
    assert abs(fit["slope"] - 2.0) < 1e-8
    assert fit["r2"] > 0.999999
    assert fit["n"] == len(hs)  # nothing to exclude, so the widest window wins


def test_fit_order_excludes_the_round_off_tail():
    """errs = h^3 down to a ~1e-13 round-off floor, then noisy flat -- the
    shape a real float64 experiment produces (the floor isn't a clean
    plateau, it scatters). The fitted slope must still read ~3, and the
    window must not swallow the noisy tail."""
    rng = np.random.default_rng(0)
    hs = 0.5 ** np.arange(1, 22, dtype=np.float64)
    clean = hs**3
    floor = 1e-13
    noisy_floor = floor * rng.uniform(0.2, 5.0, size=hs.shape)
    errs = np.where(clean > 3 * floor, clean, noisy_floor)

    fit = fit_order(hs, errs)
    assert abs(fit["slope"] - 3.0) < 0.15
    assert fit["n"] < len(hs)  # excluded at least part of the tail

    lo_h, hi_h = fit["window"]
    # the last few points, deep in the noisy floor, must fall outside the window
    assert not (min(lo_h, hi_h) <= hs[-1] <= max(lo_h, hi_h))
    assert not (min(lo_h, hi_h) <= hs[-2] <= max(lo_h, hi_h))


def test_fit_order_too_few_points_is_nan():
    fit = fit_order([0.1, 0.05], [1e-3, 2.5e-4], min_pts=4)
    assert np.isnan(fit["slope"])
    assert np.isnan(fit["r2"])
    assert fit["window"] is None


def test_fit_order_drops_diverged_runs():
    """A diverged run reports err = nan/inf; it must be dropped, not fitted."""
    hs = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    errs = np.array([1e-3, 2.5e-4, 6.25e-5, np.nan, np.inf])
    fit = fit_order(hs, errs, min_pts=3)
    assert fit["n"] == 3


def test_l2_matches_euclidean_norm():
    x = np.array([1.0, 2.0, 3.0])
    ref = np.array([1.0, 0.0, 3.0])
    assert l2(x, ref) == 2.0
