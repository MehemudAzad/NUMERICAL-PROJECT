"""Arm C, hand-written: DPM-Solver-1 and DDIM (guide step 5).

Both are exact on the linear part of the PF-ODE, marched on a grid uniform in
``lambda`` (paper section 4). Algebraically identical (paper eq 4.1) -- Gate G1
checks that identity holds to machine precision, which is the first thing to
verify before trusting any arm-C error number.
"""

from __future__ import annotations

import numpy as np

from .schedule import alpha, lmbda, sigma

__all__ = ["dpm_solver_1", "ddim_step", "march"]


def dpm_solver_1(eps_fn, x, t_prev, t_cur):
    """DPM-Solver-1: exact on the linear part, first-order in the nonlinear term.

    x_i = (alpha_i/alpha_{i-1}) x_{i-1} - sigma_i (e^h - 1) eps(x_{i-1}, t_{i-1}),
    h = lambda_i - lambda_{i-1}   (paper eq 3.7)

    Uses expm1(h), not exp(h)-1 (Appendix D.6): the small-h regime this project
    measures is exactly where exp(h)-1 loses precision to cancellation.
    """
    h = lmbda(t_cur) - lmbda(t_prev)
    return (alpha(t_cur) / alpha(t_prev)) * x - sigma(t_cur) * np.expm1(h) * eps_fn(x, t_prev)


def ddim_step(eps_fn, x, t_prev, t_cur):
    """DDIM, original form (paper eq 4.1) -- algebraically identical to dpm_solver_1."""
    a_p, a_c = alpha(t_prev), alpha(t_cur)
    s_p, s_c = sigma(t_prev), sigma(t_cur)
    return (a_c / a_p) * x - a_c * (s_p / a_p - s_c / a_c) * eps_fn(x, t_prev)


def march(step_fn, eps_fn, x0, t_grid):
    """March a one-step arm-C update (`dpm_solver_1` or `ddim_step`) over `t_grid`."""
    x = np.array(x0, dtype=np.float64, copy=True)
    for i in range(len(t_grid) - 1):
        x = step_fn(eps_fn, x, t_grid[i], t_grid[i + 1])
    return x
