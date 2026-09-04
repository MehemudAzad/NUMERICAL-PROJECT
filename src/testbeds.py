"""Testbeds: toy problems with a known-correct answer, so solver error is measurable.

Tier 1 (this milestone): data ~ N(0, diag(s)). The probability-flow ODE decouples
into ``d`` independent scalar linear ODEs, so both the score and the whole
trajectory are closed form. The stiffness knob is the condition number
``kappa = s_max / s_min``.

Tier 2 (added later): a Gaussian / point mixture — closed-form score, real curvature.

The probability-flow ODE right-hand side is defined **once** here, as
:func:`pf_rhs_t` (arm A, integrates in ``t``) and :func:`pf_rhs_lambda` (arm B,
integrates in ``lambda``). A testbed only has to supply its exact noise oracle
``eps(x, t)``; everything else is shared.
"""

from __future__ import annotations

import numpy as np

from .schedule import alpha, f, g2, sigma, sigma_hat, t_of_lmbda

__all__ = ["pf_rhs_t", "pf_rhs_lambda", "GaussianTier1"]


# --- the probability-flow ODE, written once ---------------------------------

def pf_rhs_t(eps_fn, x, t):
    """PF-ODE in t (arm A):  dx/dt = f(t) x + g^2(t)/(2 sigma(t)) * eps(x, t)."""
    return f(t) * x + g2(t) / (2.0 * sigma(t)) * eps_fn(x, t)


def pf_rhs_lambda(eps_fn, x, lam):
    """PF-ODE in lambda (arm B):  dx/dlambda = sigma_hat(l)^2 x - sigma_hat(l) eps(x, t(l)).

    Same trajectory as :func:`pf_rhs_t`, reparameterised by the half log-SNR
    (paper eq. E.1). ``eps_fn`` still takes ``t``, so we invert the schedule here.
    """
    t = t_of_lmbda(lam)
    sh = sigma_hat(lam)
    return sh**2 * x - sh * eps_fn(x, t)


# --- Tier 1: anisotropic Gaussian ------------------------------------------

class GaussianTier1:
    """Data ~ N(0, diag(s)) with condition number ``kappa``. Exact score, exact flow.

    Parameters
    ----------
    kappa : float
        Condition number s_max / s_min. The eigenvalues are log-spaced over
        ``[1/kappa, 1]`` (so ``s_max = 1``).
    d : int
        Dimension. The linear problem decouples per coordinate, so ``d`` only
        changes the vector length, never the dynamics — sweep ``kappa``, not ``d``.
    seed : int
        RNG seed for :meth:`x_T`.

    Notes
    -----
    ``x`` has shape ``(n, d)`` (a batch of ``n`` trajectories); ``t`` / ``lam`` are
    scalars (the solvers march every trajectory on one shared grid).
    """

    def __init__(self, kappa: float, d: int = 2, seed: int = 0):
        if d > 1:
            s = np.logspace(0.0, np.log10(kappa), d)
        else:
            s = np.array([1.0])
        self.s = (s / s.max()).astype(np.float64)   # s_max = 1, s_min = 1/kappa
        self.d = int(d)
        self.kappa = float(kappa)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    # -- exact quantities --------------------------------------------------

    def var(self, t) -> np.ndarray:
        """Per-coordinate marginal variance at time t:  v(t) = alpha(t)^2 s + sigma(t)^2."""
        return alpha(t) ** 2 * self.s + sigma(t) ** 2

    def eps(self, x, t) -> np.ndarray:
        """Exact noise-prediction oracle. Same signature as a network.

        eps*(x, t) = sigma(t) x / v(t), and equivalently (x - alpha(t) x0_hat) / sigma(t)
        with the Gaussian posterior mean x0_hat = (alpha(t) s / v(t)) x.
        """
        return sigma(t) * x / self.var(t)

    def score(self, x, t) -> np.ndarray:
        """The score of the noised marginal, grad log p_t(x) = -eps(x, t) / sigma(t)."""
        return -self.eps(x, t) / sigma(t)

    def exact(self, x_T, T, t) -> np.ndarray:
        """Closed-form PF-ODE solution. No integration.

        Per coordinate the ODE is dx/dt = [f + g^2/(2v)] x, which is linear, so the
        flow is multiplication by a scalar: x_j(t) = x_j(T) sqrt(v_j(t) / v_j(T)).
        (Verified by differentiation: it needs v' = 2 f v + g^2, which the schedule
        satisfies identically.)
        """
        return np.asarray(x_T, dtype=np.float64) * np.sqrt(self.var(t) / self.var(T))

    # -- ODE right-hand sides (thin wrappers over the shared PF-ODE) ------

    def rhs_t(self, x, t) -> np.ndarray:
        return pf_rhs_t(self.eps, x, t)

    def rhs_lambda(self, x, lam) -> np.ndarray:
        return pf_rhs_lambda(self.eps, x, lam)

    # -- sampling --------------------------------------------------------

    def x_T(self, n: int = 1) -> np.ndarray:
        """Draw ``n`` initial conditions from the fully-noised marginal N(0, v(T))."""
        return self.rng.standard_normal((n, self.d)) * np.sqrt(self.var(1.0))
