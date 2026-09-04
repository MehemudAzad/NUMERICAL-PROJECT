"""Tests for the Tier-1 anisotropic-Gaussian testbed (guide, step 2).

``test_exact_satisfies_ode`` is the gate before any solver work: it is the
foundation of every error number in the report. If the closed-form trajectory
does not satisfy the ODE, nothing downstream is worth running.
"""

import numpy as np
import pytest

from src.schedule import alpha, lmbda, sigma, t_of_lmbda
from src.testbeds import GaussianTier1


# --- the gate: closed form must satisfy the ODE -----------------------------

def test_exact_satisfies_ode():
    """d/dt of the closed form must equal the PF-ODE right-hand side (arm A)."""
    tb = GaussianTier1(kappa=100.0, d=4, seed=0)
    xT = tb.x_T(1)
    T = 1.0
    dt = 1e-6
    for t in [0.9, 0.7, 0.5, 0.2, 0.05, 5e-3]:
        num = (tb.exact(xT, T, t + dt) - tb.exact(xT, T, t - dt)) / (2 * dt)
        ana = tb.rhs_t(tb.exact(xT, T, t), t)
        assert np.allclose(num, ana, rtol=1e-6, atol=1e-9), t


def test_exact_satisfies_ode_lambda():
    """Same, reparameterised: d/dlambda of the closed form == rhs_lambda (arm B)."""
    tb = GaussianTier1(kappa=100.0, d=4, seed=0)
    xT = tb.x_T(1)
    T = 1.0
    dl = 1e-6

    def X(lam):
        return tb.exact(xT, T, t_of_lmbda(lam))

    for lam in np.linspace(lmbda(1.0), lmbda(1e-3), 8):
        num = (X(lam + dl) - X(lam - dl)) / (2 * dl)
        ana = tb.rhs_lambda(X(lam), lam)
        assert np.allclose(num, ana, rtol=1e-6, atol=1e-9), lam


def test_exact_matches_scipy_integration():
    """An independent stiff integrator on rhs_t must reproduce the closed form."""
    from scipy.integrate import solve_ivp

    tb = GaussianTier1(kappa=50.0, d=3, seed=1)
    xT = tb.x_T(1)
    T, t_end = 1.0, 1e-3
    ref = tb.exact(xT, T, t_end).ravel()

    sol = solve_ivp(
        lambda t, y: tb.rhs_t(y.reshape(1, -1), t).ravel(),
        (T, t_end),
        xT.ravel(),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert sol.success
    assert np.allclose(sol.y[:, -1], ref, rtol=1e-8)


# --- properties of the exact noise oracle ---------------------------------

def test_eps_matches_posterior_mean_form():
    """eps* == (x - alpha x0_hat)/sigma with Gaussian posterior mean x0_hat = alpha s/v x.

    This is the same form Tier 2 uses, specialised to a single Gaussian.
    """
    tb = GaussianTier1(kappa=30.0, d=5, seed=2)
    x = tb.x_T(7)
    for t in [0.85, 0.4, 0.05]:
        a, sg = alpha(t), sigma(t)
        x0_hat = (a * tb.s / tb.var(t)) * x
        assert np.allclose(tb.eps(x, t), (x - a * x0_hat) / sg, rtol=1e-12)


def test_eps_linear_in_x():
    """The Gaussian score is linear: eps(c x, t) == c eps(x, t)."""
    tb = GaussianTier1(kappa=100.0, d=4, seed=3)
    x = tb.x_T(4)
    for t in [0.9, 0.3]:
        assert np.allclose(tb.eps(3.0 * x, t), 3.0 * tb.eps(x, t), rtol=1e-13)


def test_eps_is_conditional_expectation_of_noise():
    """Monte Carlo: the per-coordinate OLS slope of the true noise e on x_t equals
    the eps coefficient sigma(t)/v(t), i.e. eps* = E[e | x_t]."""
    tb = GaussianTier1(kappa=8.0, d=2, seed=4)
    rng = np.random.default_rng(5)
    n = 1_500_000
    for t in [0.6, 0.3]:
        a, sg = alpha(t), sigma(t)
        x0 = rng.standard_normal((n, tb.d)) * np.sqrt(tb.s)
        e = rng.standard_normal((n, tb.d))
        xt = a * x0 + sg * e
        slope = (xt * e).mean(0) / (xt**2).mean(0)
        assert np.allclose(slope, sg / tb.var(t), rtol=3e-2)


def test_forward_marginal_variance():
    """Monte Carlo: forward samples x_t = alpha x0 + sigma e have variance v(t)."""
    tb = GaussianTier1(kappa=10.0, d=3, seed=1)
    rng = np.random.default_rng(2)
    n = 400_000
    for t in [0.7, 0.4]:
        a, sg = alpha(t), sigma(t)
        x0 = rng.standard_normal((n, 3)) * np.sqrt(tb.s)
        e = rng.standard_normal((n, 3))
        xt = a * x0 + sg * e
        assert np.allclose(xt.var(axis=0), tb.var(t), rtol=2e-2)


# --- misc ----------------------------------------------------------------

def test_stiffness_knob():
    for kappa in [1.0, 10.0, 100.0, 1e3, 1e4]:
        tb = GaussianTier1(kappa=kappa, d=50)
        assert tb.s.max() == pytest.approx(1.0)
        assert tb.s.max() / tb.s.min() == pytest.approx(kappa, rel=1e-9)


def test_shapes_and_dtype():
    tb = GaussianTier1(kappa=100.0, d=6, seed=0)
    xT = tb.x_T(9)
    assert xT.shape == (9, 6) and xT.dtype == np.float64
    assert tb.exact(xT, 1.0, 1e-3).shape == (9, 6)
    assert tb.rhs_t(xT, 0.5).shape == (9, 6)
    assert tb.rhs_lambda(xT, 0.0).shape == (9, 6)
    assert tb.var(0.3).shape == (6,)
