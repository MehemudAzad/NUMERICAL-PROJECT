"""M9 / Tier 3 (guide step 10) -- the parts that can be checked without a GPU.

The Kaggle notebook needs a T4, a checkpoint download and ~20 minutes; none of
that belongs in a suite that has to stay runnable on a laptop in seconds. What
*can* be pinned down here is everything the notebook's numbers rest on:

* :class:`src.tier3.DiscreteSchedule` reads the checkpoint's own schedule
  correctly, and its one finite difference is right;
* arms A and B describe the *same* ODE under that schedule -- the end-to-end
  check that ``f`` and ``g2`` are right. A sign error or a missing ``sigma**2``
  shows up as O(1) disagreement here, not as a subtly wrong slope three hours
  into a GPU run;
* the backend-agnostic :func:`src.solvers.integrate` marches torch tensors
  exactly as it marches numpy arrays, preserving dtype and NFE accounting.

The schedule is built from *synthetic* betas -- ``linspace(1e-4, 0.02, 1000)``,
which is exactly what ``DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")``
produces -- so there is no network access and no diffusers dependency.
"""

import numpy as np
import pytest
import torch

from src import schedule as cont
from src.grids import grid_t
from src.metrics import l2
from src.solvers import integrate
from src.testbeds import GaussianTier1, pf_rhs_lambda, pf_rhs_t
from src.tier3 import DiscreteSchedule, make_noise_schedule

# The CIFAR-10 DDPM checkpoint's betas: linear, 1e-4 -> 0.02 over 1000 steps.
BETAS = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)

# Times deliberately offset by 5e-4 from the interpolant's knots (spaced 1/N),
# so the central difference sits inside a single segment.
OFF_KNOT = np.array([0.0155, 0.1005, 0.3005, 0.5005, 0.7005, 0.9005])


@pytest.fixture(scope="module")
def ds() -> DiscreteSchedule:
    return DiscreteSchedule(make_noise_schedule(BETAS, dtype=torch.float64))


# --- the schedule adapter ---------------------------------------------------


def test_floor_and_extent(ds):
    """t_min = 1/N is the floor below which the interpolant extrapolates."""
    assert ds.total_N == 1000
    assert ds.t_min == pytest.approx(1e-3)
    assert ds.T == pytest.approx(1.0)


def test_rejects_a_float32_schedule():
    """A float32 schedule silently ruins lambda'; the constructor must refuse it."""
    ns32 = make_noise_schedule(BETAS, dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        DiscreteSchedule(ns32)


def test_discrete_and_continuous_schedules_are_close_but_not_equal(ds):
    """The checkpoint's schedule is the *discretised* VP-linear one.

    The two agree to O(1/N) and no better. That gap is the whole reason arms A
    and B read their coefficients off ``ds`` rather than off ``src.schedule``:
    at ~1% in f, mixing them would put a systematic error under every Tier-3
    error curve.
    """
    t = np.linspace(1e-3, 1.0, 400)
    d_lam = np.abs(ds.lmbda(t) - cont.lmbda(t))
    assert d_lam.max() < 6e-2, "the two should still be recognisably one schedule"
    assert d_lam.max() > 1e-3, "...but not identical -- if they were, this test is wrong"
    assert np.allclose(ds.alpha(t) ** 2 + ds.sigma(t) ** 2, 1.0, atol=1e-12)


def test_lambda_roundtrip(ds):
    """t_of_lmbda inverts lmbda over the tabulated range."""
    t = np.linspace(1e-3, 1.0, 200)
    assert np.allclose(ds.t_of_lmbda(ds.lmbda(t)), t, atol=1e-9)


def test_lambda_is_decreasing_in_t(ds):
    """The project's most common bug, checked on the Tier-3 schedule too."""
    t = np.linspace(1e-3, 1.0, 500)
    assert np.all(np.diff(ds.lmbda(t)) < 0.0)


def test_f_over_dlambda_is_sigma_squared(ds):
    """f = sigma^2 lambda' holds *identically*, not approximately.

    This is what makes the arm-A and arm-B right-hand sides the same ODE (see
    the derivation in src/tier3.py), so it must be exact in floating point --
    it is one multiplication, not two independent finite differences.
    """
    for t in OFF_KNOT:
        assert ds.f(t) / ds.dlmbda_dt(t) == pytest.approx(ds.sigma(t) ** 2, abs=1e-15)


def test_g2_is_minus_two_f(ds):
    """g^2 = -2 f, as for the continuous schedule (f = -beta/2, g^2 = beta)."""
    for t in OFF_KNOT:
        assert ds.g2(t) == pytest.approx(-2.0 * ds.f(t), rel=1e-15)


def test_f_matches_a_direct_finite_difference_of_log_alpha(ds):
    """f really is d log alpha / dt, checked against an independent stencil.

    ``f`` is computed as sigma^2 lambda'; this differentiates log alpha
    directly, at a different step size. Agreement means the identity above was
    applied in the right direction.
    """
    dt = 1e-6
    for t in OFF_KNOT:
        direct = (np.log(ds.alpha(t + dt)) - np.log(ds.alpha(t - dt))) / (2.0 * dt)
        assert ds.f(t) == pytest.approx(direct, rel=1e-5)


# --- the end-to-end check: arm A and arm B are the same ODE ----------------


def test_arm_a_equals_arm_b_under_the_discrete_schedule(ds):
    """Marching in t and marching in lambda must reach the same endpoint.

    Arm A uses f and g2, the finite-differenced coefficients; arm B uses only
    sigma_hat(lambda) and t(lambda), which are exact. So this is the real test
    of the finite difference: if f or g2 were wrong by a factor or a sign, the
    two arms would disagree at O(1) and the gap would not shrink with n.

    Convergence is asserted, not just a tolerance. The discrete schedule's
    log alpha is piecewise linear, so its f is a *staircase* and the ODE's
    right-hand side is genuinely discontinuous at the 1000 knots -- that caps
    how cleanly RK4 can close the gap, but the gap must still go to zero.
    """
    tb = GaussianTier1(kappa=10.0, d=4, seed=0)
    xT = tb.x_T(3)
    T, t_end = 1.0, 1e-3

    def rhs_a(x, t):
        return pf_rhs_t(tb.eps, x, t, sched=ds)

    def rhs_b(x, lam):
        return pf_rhs_lambda(tb.eps, x, lam, sched=ds)

    gaps = []
    for n in [200, 400, 800, 1600]:
        xa, _ = integrate(rhs_a, xT, grid_t(T, t_end, n), "rk4")
        lam_grid = np.linspace(ds.lmbda(T), ds.lmbda(t_end), n + 1)
        xb, _ = integrate(rhs_b, xT, lam_grid, "rk4")
        gaps.append(l2(xa, xb) / np.linalg.norm(xa))

    assert gaps[-1] < 1e-5, f"arms A and B disagree at the finest grid: {gaps}"
    assert gaps[0] > 50 * gaps[-1], f"the gap should shrink with n, got {gaps}"
    assert np.all(np.diff(gaps) < 0), f"non-monotone convergence: {gaps}"


# --- integrate() on torch tensors ------------------------------------------
#
# Tier 3 marches a float32 CUDA tensor through the same integrator that marched
# Tiers 1-2. These run on CPU, but exercise the exact code path.


def _decay_rhs(x, t):
    """dx/dt = -2x, exact solution x(t) = x(0) e^{-2t}. Works on both backends."""
    return -2.0 * x


@pytest.mark.parametrize("stepper", ["euler", "midpoint", "heun3", "rk4", "ab2"])
def test_integrate_matches_between_backends(stepper):
    """The torch path and the numpy path give the same answer and the same NFE."""
    x0 = np.array([[1.0, -2.0, 0.5]])
    grid = np.linspace(0.0, 1.0, 33)

    xn, nfe_n = integrate(_decay_rhs, x0, grid, stepper)
    xt, nfe_t = integrate(_decay_rhs, torch.tensor(x0, dtype=torch.float64), grid, stepper)

    assert nfe_t == nfe_n
    assert np.allclose(xt.numpy(), xn, rtol=0.0, atol=1e-14)


def test_integrate_preserves_torch_dtype_and_does_not_promote():
    """A float32 state must stay float32 -- it is Tier 3's measurement floor."""
    x0 = torch.randn(2, 3, dtype=torch.float32)
    xf, _ = integrate(_decay_rhs, x0, np.linspace(0.0, 1.0, 17), "rk4")
    assert xf.dtype == torch.float32
    assert torch.allclose(xf, x0 * float(np.exp(-2.0)), atol=1e-5)


def test_integrate_does_not_alias_the_torch_initial_state():
    """integrate() copies x0; the caller's x_T is reused across every sweep point."""
    x0 = torch.ones(1, 4, dtype=torch.float64)
    before = x0.clone()
    integrate(_decay_rhs, x0, np.linspace(0.0, 1.0, 9), "euler")
    assert torch.equal(x0, before)


def test_torch_ab2_nfe_accounting():
    """AB2 still spends N+1 calls (1/step after a 2-call RK2 startup) on tensors."""
    n = 24
    _, nfe = integrate(
        _decay_rhs, torch.ones(1, 2, dtype=torch.float64), np.linspace(0.0, 1.0, n + 1), "ab2"
    )
    assert nfe == n + 1


def test_torch_divergence_is_flagged():
    """A blow-up returns nan NFE on the torch path, same as on numpy."""

    def blow_up(x, t):
        return x * 1e300

    _, nfe = integrate(
        blow_up, torch.ones(1, 2, dtype=torch.float64), np.linspace(0.0, 1.0, 11), "euler"
    )
    assert np.isnan(nfe)
