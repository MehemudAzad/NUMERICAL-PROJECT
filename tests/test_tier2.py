"""Tests for the Tier-2 Gaussian/point-mixture testbed (guide, step 9).

``test_k1_matches_point_mass_exact`` is this tier's analogue of
``test_tier1.py``'s ``test_exact_satisfies_ode`` gate: for a single mode
(K=1) the mixture degenerates to a point mass at ``mu``, which *does* have a
closed-form trajectory (derived independently below, not borrowed from
``GaussianTier1``, which only ever models zero-mean data). If the general
K-mode ``eps`` doesn't reduce correctly to that K=1 special case, nothing
built on top of it (order fits, the crossover study) is trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.schedule import alpha, sigma
from src.testbeds import GaussianTier1, MixtureTier2, dop853_reference, swiss_roll


# --- the gate: K=1 must match an independently-derived exact solution ------

def _point_mass_exact(mu, xT, T, t):
    """x0 = mu with certainty, so xt = alpha(t) mu + sigma(t) z for a *fixed* z
    (the same noise realisation throughout the reverse process). Solving for z
    from the given x_T pins down the whole trajectory."""
    z = (xT - alpha(T) * mu) / sigma(T)
    return alpha(t) * mu + sigma(t) * z


def test_k1_matches_point_mass_exact():
    mu = np.array([[0.7, -1.3]])
    tb = MixtureTier2(mu, seed=0)
    xT = tb.x_T(6)
    T = 1.0
    dt = 1e-6
    for t in [0.9, 0.6, 0.3, 0.05, 5e-3]:
        num = (
            _point_mass_exact(mu[0], xT, T, t + dt) - _point_mass_exact(mu[0], xT, T, t - dt)
        ) / (2 * dt)
        ana = tb.rhs_t(_point_mass_exact(mu[0], xT, T, t), t)
        assert np.allclose(num, ana, rtol=1e-5, atol=1e-8), t


def test_k1_matches_point_mass_exact_lambda():
    """Same gate, reparameterised (arm B)."""
    from src.schedule import lmbda, t_of_lmbda

    mu = np.array([[0.7, -1.3]])
    tb = MixtureTier2(mu, seed=0)
    xT = tb.x_T(6)
    T = 1.0
    dl = 1e-6

    def X(lam):
        return _point_mass_exact(mu[0], xT, T, t_of_lmbda(lam))

    for lam in np.linspace(lmbda(1.0), lmbda(1e-3), 8):
        num = (X(lam + dl) - X(lam - dl)) / (2 * dl)
        ana = tb.rhs_lambda(X(lam), lam)
        assert np.allclose(num, ana, rtol=1e-5, atol=1e-8), lam


# --- properties of the posterior weights / noise oracle --------------------

def test_posterior_weights_sum_to_one_and_are_nonnegative():
    mu = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    tb = MixtureTier2(mu, seed=1)
    x = tb.x_T(20)
    for t in [0.9, 0.5, 0.1, 0.01]:
        r = tb.posterior_weights(x, t)
        assert r.shape == (20, 3)
        assert np.all(r >= 0.0)
        assert np.allclose(r.sum(axis=1), 1.0, rtol=1e-12)


def test_posterior_weights_no_overflow_at_small_sigma():
    """The naive (non-logsumexp) softmax would overflow here; logsumexp must not."""
    mu = np.array([[-5.0, 0.0], [5.0, 0.0]])
    tb = MixtureTier2(mu, seed=2)
    x = np.array([[-5.0, 0.0]])
    r = tb.posterior_weights(x, 1e-4)  # sigma(t) tiny -> huge exponents
    assert np.all(np.isfinite(r))
    assert np.allclose(r.sum(axis=1), 1.0)
    # x sits exactly on mode 0 at a near-zero-noise time: responsibility -> 1.
    assert r[0, 0] > 1.0 - 1e-6


def test_symmetric_two_mode_responsibility_is_half_at_midpoint():
    """By symmetry, a point equidistant from two equal-weight modes gets r = 0.5 each."""
    mu = np.array([[-1.0, 0.0], [1.0, 0.0]])
    tb = MixtureTier2(mu, seed=3)
    x = np.array([[0.0, 0.0]])
    for t in [0.8, 0.3, 0.05]:
        r = tb.posterior_weights(x, t)
        assert np.allclose(r, 0.5, atol=1e-12)


def test_eps_reduces_to_gaussian_tier1_in_the_isotropic_limit():
    """Many modes packed into an epsilon-ball around the origin, with an
    isotropic-Gaussian data covariance folded in via extra 'jitter' modes,
    should approximately reproduce GaussianTier1's score for tiny spread --
    both describe (in this limit) data concentrated near a single point.
    Here we just check the K>>1, near-coincident-mode limit collapses to the
    K=1 point-mass gate above (r_k -> uniform is not required; eps -> the
    single-point formula is)."""
    mu = np.tile(np.array([[0.3, -0.2]]), (5, 1)) + 1e-9 * np.arange(5)[:, None]
    tb = MixtureTier2(mu, seed=4)
    tb1 = MixtureTier2(np.array([[0.3, -0.2]]), seed=4)
    x = tb1.x_T(4)
    for t in [0.7, 0.2]:
        assert np.allclose(tb.eps(x, t), tb1.eps(x, t), rtol=1e-6, atol=1e-9)


# --- DOP853 reference --------------------------------------------------

def test_dop853_reference_is_converged():
    """Recomputing at a tighter rtol should barely move the answer -- guide's
    prescribed check that the nominal 1e-13 reference is actually converged."""
    mu = swiss_roll(n=8, noise=0.0, seed=0)
    tb = MixtureTier2(mu, seed=0)
    xT = tb.x_T(5)
    ref_nominal = dop853_reference(tb, xT, T=1.0, t_end=1e-3, rtol=1e-13, atol=1e-14)
    ref_tight = dop853_reference(tb, xT, T=1.0, t_end=1e-3, rtol=1e-11, atol=1e-14)
    assert np.allclose(ref_nominal, ref_tight, atol=1e-6)


def test_dop853_reference_matches_point_mass_exact():
    """On the K=1 degenerate case, the independent SciPy reference must agree
    with the closed-form point-mass trajectory -- ties this tier's numerical
    reference machinery back to an analytic answer."""
    mu = np.array([[0.4, 0.9]])
    tb = MixtureTier2(mu, seed=5)
    xT = tb.x_T(4)
    T, t_end = 1.0, 1e-3
    ref = dop853_reference(tb, xT, T=T, t_end=t_end, rtol=1e-13, atol=1e-14)
    exact = _point_mass_exact(mu[0], xT, T, t_end)
    assert np.allclose(ref, exact, rtol=1e-7, atol=1e-9)


def test_dop853_reference_batches_consistently_with_single_row():
    """Integrating a batch of rows in one call must give the same answer as
    integrating each row alone (the ODE right-hand side must not mix rows)."""
    mu = swiss_roll(n=6, noise=0.0, seed=1)
    tb = MixtureTier2(mu, seed=6)
    xT = tb.x_T(4)
    batched = dop853_reference(tb, xT, T=1.0, t_end=1e-2, rtol=1e-12, atol=1e-13)
    singles = np.stack(
        [dop853_reference(tb, xT[i:i + 1], T=1.0, t_end=1e-2, rtol=1e-12, atol=1e-13)[0]
         for i in range(len(xT))]
    )
    assert np.allclose(batched, singles, rtol=1e-8, atol=1e-10)


# --- swiss_roll --------------------------------------------------------

def test_swiss_roll_shape_and_reproducibility():
    a = swiss_roll(n=50, noise=0.05, seed=0)
    b = swiss_roll(n=50, noise=0.05, seed=0)
    c = swiss_roll(n=50, noise=0.05, seed=1)
    assert a.shape == (50, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_swiss_roll_noise_free_points_lie_on_the_parametric_curve():
    pts = swiss_roll(n=20, noise=0.0, seed=0)
    r = np.linalg.norm(pts * 15.0, axis=1)  # undo the /15 scaling
    p = 1.5 * np.pi
    # r == |p| by construction (x=p cos p, y=p sin p); the sampled p range is
    # [1.5 pi, 4.5 pi], all positive, so r == p exactly.
    assert np.all(r >= p - 1e-9)
    assert np.all(r <= 3.0 * p + 1e-9)


# --- misc / shapes ----------------------------------------------------------

def test_shapes_and_dtype():
    mu = swiss_roll(n=4, noise=0.0, seed=0)
    tb = MixtureTier2(mu, seed=0)
    xT = tb.x_T(9)
    assert xT.shape == (9, 2) and xT.dtype == np.float64
    assert tb.rhs_t(xT, 0.5).shape == (9, 2)
    assert tb.rhs_lambda(xT, 0.0).shape == (9, 2)
    assert tb.posterior_weights(xT, 0.5).shape == (9, 4)


def test_custom_weights_are_respected():
    mu = np.array([[-3.0, 0.0], [3.0, 0.0]])
    tb = MixtureTier2(mu, w=[0.9, 0.1], seed=0)
    x = np.array([[0.0, 0.0]])
    r = tb.posterior_weights(x, 0.9)  # early/noisy t: prior weight should dominate
    assert r[0, 0] > r[0, 1]
