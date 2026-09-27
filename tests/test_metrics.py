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


def test_fit_order_floor_recovers_a_slope_the_window_search_misses():
    """A known error floor must be excluded explicitly, not left to the search.

    `fit_order` scores windows as `r2 + 0.01 * width`, so a wide window running
    into a flat floor can outscore a narrower clean one and drag the slope
    toward zero. This is the real Tier-3 situation -- float32 plus the network's
    own non-smoothness put a hard floor under every curve -- and the guide's
    step 10 says plainly not to fit through it. There, `dpm3` measured 2.473
    naively and 2.995 with the floor excluded.
    """
    hs = np.logspace(0.0, -1.5, 10)
    floor = 0.05
    observed = np.maximum(30.0 * hs**3, floor)   # a clean cubic that flattens out

    naive = fit_order(hs, observed)["slope"]
    fitted = fit_order(hs, observed, floor=3 * floor)["slope"]

    assert abs(fitted - 3.0) < 0.05, f"floor-excluded fit should recover 3, got {fitted}"
    assert fitted - naive > 0.1, (
        f"this test is pointless unless the naive fit is actually biased "
        f"(naive={naive}, floor-excluded={fitted})"
    )


def test_fit_order_floor_is_a_no_op_when_nothing_reaches_it():
    """Tiers 1-2 sit far above their round-off floor; passing one must change nothing."""
    hs = np.logspace(0.0, -1.5, 8)
    errs = 5.0 * hs**2
    assert fit_order(hs, errs, floor=1e-12)["slope"] == fit_order(hs, errs)["slope"]


def test_fit_order_floor_leaving_too_few_points_returns_nan():
    """Dropping below `min_pts` must fail loudly-as-nan, not fit two points."""
    out = fit_order(np.logspace(0.0, -1.0, 6), np.full(6, 0.5), floor=0.6)
    assert np.isnan(out["slope"]) and out["window"] is None
