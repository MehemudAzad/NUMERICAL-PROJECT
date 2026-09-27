"""M12.4: when does DPM-Solver's exact-linear-part split help?

Finding 6 (review, 2026-09-24): on Tier 1 at the data's own variance (s = 1,
i.e. kappa = 1), the arm-A right-hand side has an exact cancellation that
DPM-Solver's split throws away. This is the one point on the M12.4 sweep with
a closed-form answer, so it is pinned down as a unit test independent of the
notebook: ``kappa=1.0`` gives every coordinate ``s = 1`` (GaussianTier1.s is
log-spaced over ``[1/kappa, 1]``, which collapses to all-ones at kappa=1), so
``v(t) = alpha(t)^2 + sigma(t)^2 = 1`` identically and the arm-A right-hand
side ``f(t) x + g^2(t)/(2 sigma(t)) * eps(x,t)`` reduces to
``[-beta/2 + beta/(2 sigma^2) * sigma] x = [-beta/2 + beta/2] x = 0`` -- Euler
is then exact to machine precision at *any* step size, while DPM-Solver-1
still spends its expm1(h) term on a linear part that needed no discretization
in the first place.
"""

from __future__ import annotations

import numpy as np

from src.arm_c import sample_dpm_solver
from src.grids import grid_t
from src.solvers import integrate
from src.testbeds import GaussianTier1

T, T_END = 1.0, 1e-3


def test_at_s_equals_1_euler_in_t_is_exact_but_dpm1_is_not():
    tb = GaussianTier1(kappa=1.0, d=2, seed=0)
    assert np.allclose(tb.s, 1.0), "kappa=1 must give every coordinate s=1"

    xT = tb.x_T(4)
    ref = tb.exact(xT, T, T_END)

    xf_euler, _ = integrate(tb.rhs_t, xT, grid_t(T, T_END, 20), "euler")
    xf_dpm1, _ = sample_dpm_solver(tb.eps, xT, T, T_END, order=1, steps=20)

    assert np.max(np.abs(xf_euler - ref)) < 1e-12
    assert np.max(np.abs(xf_dpm1 - ref)) > 1e-4, (
        "DPM-Solver-1 should NOT be near-exact here -- if it is, the s=1 "
        "cancellation finding has vanished and M12.4's premise is wrong"
    )


def test_rhs_t_is_identically_zero_at_s_equals_1():
    """The mechanism directly: the arm-A right-hand side itself, not just the
    resulting Euler error, must vanish at every t when s = 1."""
    tb = GaussianTier1(kappa=1.0, d=1, seed=0)
    x = np.array([[0.37]])
    for t in (0.9, 0.5, 0.2, 0.05, 1e-3):
        assert abs(tb.rhs_t(x, t)[0, 0]) < 1e-13
