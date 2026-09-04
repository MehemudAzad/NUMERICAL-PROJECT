"""Tests for the VP-linear noise schedule (guide, step 1).

The four core tests are the guide's; the rest guard the lambda-direction bug and
the agreement with the vendored DPM-Solver noise schedule (arm C depends on it).
"""

import numpy as np
import pytest

from src.schedule import (
    BETA_0,
    BETA_1,
    alpha,
    alpha_hat,
    f,
    g2,
    lmbda,
    log_alpha,
    sigma,
    sigma_hat,
    t_of_lmbda,
)


# --- the guide's four ---------------------------------------------------------

def test_roundtrip():
    """t -> lambda -> t is the identity (pure algebra, should be ~machine)."""
    t = np.logspace(-4, 0, 200)
    assert np.allclose(t_of_lmbda(lmbda(t)), t, rtol=1e-10)


def test_variance_preserving():
    t = np.linspace(1e-4, 1.0, 100)
    assert np.allclose(alpha(t) ** 2 + sigma(t) ** 2, 1.0, atol=1e-12)


def test_f_matches_derivative():
    t = np.linspace(0.05, 0.95, 40)
    dt = 1e-6
    fd = (log_alpha(t + dt) - log_alpha(t - dt)) / (2 * dt)
    assert np.allclose(fd, f(t), rtol=1e-6)


def test_g2_identity():
    """g^2(t) = -2 sigma(t)^2 dlambda/dt   (paper eq. 3.2)."""
    t = np.linspace(0.05, 0.95, 40)
    dt = 1e-6
    dl = (lmbda(t + dt) - lmbda(t - dt)) / (2 * dt)
    assert np.allclose(-2 * sigma(t) ** 2 * dl, g2(t), rtol=1e-6)


# --- the lambda direction (the project's most common bug) --------------------

def test_lambda_strictly_decreasing_in_t():
    t = np.linspace(1e-3, 1.0, 500)
    lam = lmbda(t)
    assert np.all(np.diff(lam) < 0.0)
    assert lmbda(1.0) < lmbda(1e-3)          # T end is the most negative lambda
    assert lmbda(1e-4) > lmbda(1e-2) > 0.0   # lambda -> +inf as t -> 0


# --- lambda-domain coefficients ---------------------------------------------

def test_alpha_hat_sigma_hat_consistency():
    t = np.linspace(1e-3, 1.0, 100)
    lam = lmbda(t)
    assert np.allclose(alpha_hat(lam), alpha(t), rtol=1e-10)
    assert np.allclose(sigma_hat(lam), sigma(t), rtol=1e-9)
    assert np.allclose(alpha_hat(lam) ** 2 + sigma_hat(lam) ** 2, 1.0, atol=1e-12)


# --- agreement with the vendored DPM-Solver schedule (arm C) -----------------

def test_matches_vendored_noise_schedule():
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party"))
    import torch
    from dpm_solver_pytorch import NoiseScheduleVP

    ns = NoiseScheduleVP(
        schedule="linear",
        continuous_beta_0=BETA_0,
        continuous_beta_1=BETA_1,
        dtype=torch.float64,
    )
    t_np = np.linspace(1e-3, 1.0, 50)
    t_th = torch.tensor(t_np, dtype=torch.float64)

    assert np.allclose(ns.marginal_alpha(t_th).numpy(), alpha(t_np), rtol=1e-11)
    assert np.allclose(ns.marginal_std(t_th).numpy(), sigma(t_np), rtol=1e-9)
    assert np.allclose(ns.marginal_lambda(t_th).numpy(), lmbda(t_np), rtol=1e-11)

    lam_np = lmbda(t_np)
    lam_th = torch.tensor(lam_np, dtype=torch.float64)
    assert np.allclose(ns.inverse_lambda(lam_th).numpy(), t_of_lmbda(lam_np), rtol=1e-11)
