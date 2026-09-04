"""Tests for arm A/B solvers, AB2, and the shared integrator (guide step 3).

Each stepper is checked against its textbook order on the closed-form Tier-1
testbed. Step counts are chosen per solver so the fitted slope sits in the
asymptotic regime: too few steps and pre-asymptotic terms dominate; too many
and, for the higher-order solvers, the float64 round-off floor (~1e-13) would.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import grid_lambda, grid_t
from src.schedule import lmbda
from src.solvers import NFE_PER_STEP, STEPPERS, integrate
from src.testbeds import GaussianTier1

T, T_END = 1.0, 0.2
KAPPA = 10.0


def _slope(ns, errs) -> float:
    """Least-squares log-log slope of err vs h=1/n."""
    hs = 1.0 / np.asarray(ns, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    A = np.vstack([np.log(hs), np.ones_like(hs)]).T
    slope, _ = np.linalg.lstsq(A, np.log(errs), rcond=None)[0]
    return float(slope)


def _l2(x, ref) -> float:
    return float(np.sqrt(np.mean((x - ref) ** 2)))


# n's chosen so the fitted window is asymptotic for that solver's order.
ARM_A_CASES = {
    "euler": (1, [8, 16, 32, 64]),
    "midpoint": (2, [8, 16, 32, 64]),
    "heun3": (3, [16, 32, 64, 128]),
    "rk4": (4, [32, 64, 128, 256]),
}


@pytest.mark.parametrize("name", sorted(ARM_A_CASES))
def test_arm_a_stepper_hits_textbook_order(name):
    order, ns = ARM_A_CASES[name]
    tb = GaussianTier1(kappa=KAPPA, d=3, seed=0)
    xT = tb.x_T(5)
    ref = tb.exact(xT, T, T_END)

    errs = []
    for n in ns:
        grid = grid_t(T, T_END, n)
        xf, nfe = integrate(tb.rhs_t, xT, grid, name)
        assert nfe == NFE_PER_STEP[name] * n
        errs.append(_l2(xf, ref))

    assert abs(_slope(ns, errs) - order) < 0.15, (name, errs)


def test_ab2_hits_order_2_and_nfe_is_n_plus_1():
    tb = GaussianTier1(kappa=KAPPA, d=3, seed=0)
    xT = tb.x_T(5)
    ref = tb.exact(xT, T, T_END)

    ns = [16, 32, 64, 128]
    errs = []
    for n in ns:
        grid = grid_t(T, T_END, n)
        xf, nfe = integrate(tb.rhs_t, xT, grid, "ab2")
        assert nfe == n + 1  # measured, not assumed
        errs.append(_l2(xf, ref))

    assert abs(_slope(ns, errs) - 2.0) < 0.15


def test_arm_b_grid_lambda_reaches_the_same_answer_as_arm_a():
    """Arm B (rhs_lambda on grid_lambda) must converge to the same exact
    trajectory as arm A -- it's the same ODE, only reparameterised."""
    tb = GaussianTier1(kappa=KAPPA, d=3, seed=1)
    xT = tb.x_T(4)
    ref = tb.exact(xT, T, T_END)

    grid = grid_lambda(T, T_END, 200)
    xf, nfe = integrate(tb.rhs_lambda, xT, grid, "rk4")
    assert nfe == NFE_PER_STEP["rk4"] * 200
    assert _l2(xf, ref) < 1e-6


def test_grid_t_is_uniform_and_descending():
    n = 10
    g = grid_t(T, T_END, n)
    assert g.shape == (n + 1,)
    assert g[0] == T and g[-1] == pytest.approx(T_END)
    assert np.allclose(np.diff(g), np.diff(g)[0])
    assert np.all(np.diff(g) < 0)


def test_grid_lambda_is_uniform_and_increasing():
    n = 10
    g = grid_lambda(T, T_END, n)
    assert g.shape == (n + 1,)
    assert g[0] == pytest.approx(lmbda(T))
    assert g[-1] == pytest.approx(lmbda(T_END))
    assert np.allclose(np.diff(g), np.diff(g)[0])
    assert np.all(np.diff(g) > 0)  # lambda increases as t decreases


@pytest.mark.parametrize("stepper", ["euler", "midpoint", "heun3", "rk4", "ab2"])
def test_integrate_flags_divergence(stepper):
    """A rhs that blows up under float64 must be caught: nfe = nan, not a
    partial count -- the divergence detector, not the physics, is under test."""
    def explosive_rhs(x, t):
        return 1e100 * x

    grid = np.linspace(0.0, 1.0, 5)
    x0 = np.array([[1.0, -1.0]])
    xf, nfe = integrate(explosive_rhs, x0, grid, stepper)
    assert np.isnan(nfe)
    assert not np.all(np.isfinite(xf))


def test_integrate_returns_x0_shape_and_dtype():
    tb = GaussianTier1(kappa=KAPPA, d=4, seed=2)
    xT = tb.x_T(6)
    grid = grid_t(T, T_END, 5)
    for name in list(STEPPERS) + ["ab2"]:
        xf, nfe = integrate(tb.rhs_t, xT, grid, name)
        assert xf.shape == xT.shape
        assert xf.dtype == np.float64
        assert nfe == (5 + 1 if name == "ab2" else NFE_PER_STEP[name] * 5)
