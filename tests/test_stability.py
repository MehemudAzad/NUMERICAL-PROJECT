"""Tests for the stability envelope (guide step 8).

``max_stable_h``'s bisection logic is verified against a synthetic linear
testbed with a known analytic instability threshold, independent of whether
any real Tier-1 testbed actually destabilizes (see notebooks/06_stability.ipynb
for that empirical question). ``diverged`` is checked directly.
"""

from __future__ import annotations

import numpy as np

from src.stability import diverged, max_stable_h
from src.testbeds import GaussianTier1


class _LinearTestbed:
    """dx/dt = J*x, constant J. Explicit Euler on a grid from T to t_end=0,
    n steps, step size h=T/n: the amplification factor per step is
    1 - h*J (h taken as |b-a|), so the exact stability boundary is
    h = 2/J -- a closed-form target to bisect against."""

    def __init__(self, J: float):
        self.J = J

    def rhs_t(self, x, t):
        return self.J * x

    def rhs_lambda(self, x, lam):
        return self.J * x

    def x_T(self, n: int = 1):
        return np.ones((n, 1))


def test_diverged_flags_non_finite():
    x_T = np.array([[1.0, 1.0]])
    assert diverged(np.array([[np.nan, 0.0]]), x_T)
    assert diverged(np.array([[np.inf, 0.0]]), x_T)


def test_diverged_flags_blow_up_but_not_bounded_growth():
    x_T = np.array([[1.0, 0.0]])
    assert diverged(np.array([[20.0, 0.0]]), x_T, factor=10.0)  # 20x growth
    assert not diverged(np.array([[5.0, 0.0]]), x_T, factor=10.0)  # 5x, under factor


def test_max_stable_h_matches_analytic_threshold():
    """Bisection must land near h = 2/J, the exact Euler stability boundary
    for dx/dt = J*x."""
    J = 1000.0
    tb = _LinearTestbed(J)
    h = max_stable_h(tb, "A", "euler", T=1.0, t_end=0.0, n_lo=2, n_hi=4096)
    assert abs(h - 2.0 / J) / (2.0 / J) < 0.05


def test_max_stable_h_is_nan_when_unstable_even_at_n_hi():
    """If the coarsest allowed grid (n_hi steps) still diverges, there is no
    stable h within the searched range -- must report nan, not a wrong number."""
    J = 1000.0
    tb = _LinearTestbed(J)
    h = max_stable_h(tb, "A", "euler", T=1.0, t_end=0.0, n_lo=2, n_hi=100)
    assert np.isnan(h)


def test_max_stable_h_is_n_lo_grid_when_already_stable_there():
    """If even the coarsest grid (n_lo steps) is stable, that's the answer --
    bisection must not search further than necessary."""
    J = 1.0  # tiny J: even n_lo=2 is comfortably stable
    tb = _LinearTestbed(J)
    h = max_stable_h(tb, "A", "euler", T=1.0, t_end=0.0, n_lo=2, n_hi=4096)
    assert h == 0.5  # (T - t_end) / n_lo


def test_max_stable_h_on_tier1_is_finite_for_both_arms():
    """On Tier 1 with these schedule constants, arm A does not destabilize
    within the swept kappa range (see notebooks/06_stability.ipynb for the
    full sweep and the analytic explanation) -- so this just checks the
    function returns a sane, finite h on the real testbed, for both arms."""
    tb = GaussianTier1(kappa=1000.0, d=3, seed=0)
    hA = max_stable_h(tb, "A", "euler", T=1.0, t_end=1e-3, n_lo=2, n_hi=512)
    hB = max_stable_h(tb, "B", "euler", T=1.0, t_end=1e-3, n_lo=2, n_hi=512)
    assert np.isfinite(hA) and hA > 0
    assert np.isfinite(hB) and hB > 0
