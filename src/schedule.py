"""Variance-preserving (VP) linear noise schedule.

This is the continuous-time forward process every arm of the project shares.
Hyperparameters match the base paper (Lu et al., DPM-Solver) and Song et al.'s
ScoreSDE default: ``beta_0 = 0.1``, ``beta_1 = 20``.

Conventions
-----------
* Time ``t`` runs from ``T = 1`` (pure noise) down to a small ``t_end`` (~1e-3).
* ``alpha(t)`` is the signal coefficient, ``sigma(t)`` the noise std, with
  ``alpha**2 + sigma**2 == 1`` (variance preserving).
* ``lambda(t) = log alpha(t) - log sigma(t)`` is the half log-SNR. It is
  **strictly decreasing in t**: ``t = 1`` gives the most negative lambda,
  ``t -> 0`` gives ``lambda -> +inf``. Getting this direction backwards is the
  single most common bug in this project.

All functions take array-likes, coerce to float64, and return float64. Everything
downstream runs in float64 so that order fits are not limited by a ~1e-7 float32
error floor.

Formulas (guide, Part 2)
------------------------
    log alpha(t) = -(b1 - b0)/4 * t**2  -  b0/2 * t
    sigma(t)     = sqrt(1 - alpha(t)**2)
    beta(t)      = b0 + t (b1 - b0)
    f(t)         = d log alpha / dt = -0.5 beta(t)          (PF-ODE linear drift)
    g2(t)        = beta(t)                                   (diffusion, squared)
    lambda(t)    = log alpha(t) - log sigma(t)
    t(lambda)    = 2 u / (sqrt(b0**2 + 2 (b1 - b0) u) + b0),  u = log(e^{-2 lambda} + 1)

    alpha_hat(lambda) = 1 / sqrt(1 + e^{-2 lambda})   ( == alpha(t(lambda)) )
    sigma_hat(lambda) = 1 / sqrt(1 + e^{ 2 lambda})   ( == sigma(t(lambda)) )
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BETA_0",
    "BETA_1",
    "T_MAX",
    "log_alpha",
    "alpha",
    "beta",
    "f",
    "g2",
    "sigma",
    "lmbda",
    "t_of_lmbda",
    "alpha_hat",
    "sigma_hat",
]

BETA_0: float = 0.1
BETA_1: float = 20.0
T_MAX: float = 1.0  # T: the end time of the forward process


def _asarray(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def log_alpha(t) -> np.ndarray:
    """log alpha(t). Closed form; the integral of f(t) = -beta(t)/2."""
    t = _asarray(t)
    return -(BETA_1 - BETA_0) / 4.0 * t**2 - BETA_0 / 2.0 * t


def alpha(t) -> np.ndarray:
    """Signal coefficient alpha(t) in (0, 1]."""
    return np.exp(log_alpha(t))


def beta(t) -> np.ndarray:
    """Instantaneous noise rate beta(t), linear in t."""
    return BETA_0 + _asarray(t) * (BETA_1 - BETA_0)


def f(t) -> np.ndarray:
    """PF-ODE linear drift coefficient, f(t) = d log alpha / dt = -beta(t)/2."""
    return -0.5 * beta(t)


def g2(t) -> np.ndarray:
    """Squared diffusion coefficient, g^2(t) = beta(t)."""
    return beta(t)


def sigma(t) -> np.ndarray:
    """Noise std sigma(t) = sqrt(1 - alpha(t)^2).

    Uses ``-expm1(2 log alpha)`` rather than ``1 - alpha**2`` so the result stays
    accurate when alpha is very close to 1 (small t).
    """
    la = log_alpha(t)
    return np.sqrt(-np.expm1(2.0 * la))


def lmbda(t) -> np.ndarray:
    """Half log-SNR, lambda(t) = log alpha(t) - log sigma(t). Decreasing in t."""
    la = log_alpha(t)
    return la - 0.5 * np.log(-np.expm1(2.0 * la))


def t_of_lmbda(lam) -> np.ndarray:
    """Inverse of :func:`lmbda`: recover t from lambda.

    Solve ``-2 log alpha(t) = log(1 + e^{-2 lambda}) =: u`` (a quadratic in t) and
    take the positive root in its rationalised, cancellation-free form. Matches
    ``NoiseScheduleVP('linear').inverse_lambda`` in the vendored DPM-Solver.
    """
    lam = _asarray(lam)
    u = np.logaddexp(0.0, -2.0 * lam)  # log(e^{-2 lambda} + 1)
    return 2.0 * u / (np.sqrt(BETA_0**2 + 2.0 * (BETA_1 - BETA_0) * u) + BETA_0)


def alpha_hat(lam) -> np.ndarray:
    """alpha as a direct function of lambda: 1 / sqrt(1 + e^{-2 lambda})."""
    return 1.0 / np.sqrt(1.0 + np.exp(-2.0 * _asarray(lam)))


def sigma_hat(lam) -> np.ndarray:
    """sigma as a direct function of lambda: 1 / sqrt(1 + e^{2 lambda})."""
    return 1.0 / np.sqrt(1.0 + np.exp(2.0 * _asarray(lam)))
