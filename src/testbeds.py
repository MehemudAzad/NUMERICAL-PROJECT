"""Testbeds: toy problems with a known-correct answer, so solver error is measurable.

Tier 1: data ~ N(0, diag(s)). The probability-flow ODE decouples into ``d``
independent scalar linear ODEs, so both the score and the whole trajectory are
closed form. The stiffness knob is the condition number ``kappa = s_max / s_min``.

Tier 2: a Gaussian / point mixture (:class:`MixtureTier2`) — closed-form score,
but the posterior mean is now a genuine softmax over K modes, so the PF-ODE is
nonlinear in x and has real curvature. No algebraic trajectory exists, so its
reference comes from :func:`dop853_reference` instead of an ``exact()`` method.

The probability-flow ODE right-hand side is defined **once** here, as
:func:`pf_rhs_t` (arm A, integrates in ``t``) and :func:`pf_rhs_lambda` (arm B,
integrates in ``lambda``). A testbed only has to supply its exact noise oracle
``eps(x, t)``; everything else is shared.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .schedule import alpha, f, g2, sigma, sigma_hat, t_of_lmbda

__all__ = [
    "pf_rhs_t",
    "pf_rhs_lambda",
    "GaussianTier1",
    "MixtureTier2",
    "swiss_roll",
    "dop853_reference",
]


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


# --- Tier 2: Gaussian / point mixture --------------------------------------

class MixtureTier2:
    """Data = a weighted point cloud mu_1..mu_K (weights w_k). Closed-form score,
    real curvature (guide step 9).

    The noised density stays a K-component Gaussian mixture, so the score is
    still exact -- but unlike :class:`GaussianTier1`, the posterior mean below
    is a genuine softmax over the K modes, so the PF-ODE right-hand side is
    *nonlinear* in x. There is no algebraic closed-form trajectory here (Tier
    1's ``exact()`` has no Tier-2 analogue); the reference instead comes from
    :func:`dop853_reference`.

    Parameters
    ----------
    mu : array-like, shape (K, d)
        Mode locations.
    w : array-like, shape (K,), optional
        Mixture weights (default: uniform).
    seed : int
        RNG seed for :meth:`x_T`.
    """

    def __init__(self, mu, w=None, seed: int = 0):
        self.mu = np.asarray(mu, dtype=np.float64)  # (K, d)
        k = len(self.mu)
        self.logw = np.log(np.full(k, 1.0 / k) if w is None else np.asarray(w, dtype=np.float64))
        self.d = self.mu.shape[1]
        self.k = k
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def posterior_weights(self, x, t) -> np.ndarray:
        """Softmax responsibilities r_k(x, t) of each mode.

        Computed via ``logsumexp`` rather than a naive ``exp``-then-normalise:
        at small sigma(t) the exponent -||x - alpha mu_k||^2 / (2 sigma^2)
        overflows a plain ``np.exp`` long before the *ratios* r_k stop being
        well-defined.
        """
        x = np.atleast_2d(x)
        a, sg = alpha(t), sigma(t)
        diff = x[:, None, :] - a * self.mu[None, :, :]  # (n, k, d)
        logp = self.logw[None, :] - (diff**2).sum(-1) / (2.0 * sg**2)
        return np.exp(logp - logsumexp(logp, axis=1, keepdims=True))

    def eps(self, x, t) -> np.ndarray:
        """Exact noise-prediction oracle: eps*(x,t) = (x - alpha(t) x0_hat) / sigma(t),
        with the Gaussian-mixture posterior mean x0_hat = sum_k r_k(x,t) mu_k."""
        x = np.atleast_2d(x)
        a, sg = alpha(t), sigma(t)
        r = self.posterior_weights(x, t)
        x0_hat = np.einsum("nk,kd->nd", r, self.mu)
        return (x - a * x0_hat) / sg

    def rhs_t(self, x, t) -> np.ndarray:
        return pf_rhs_t(self.eps, x, t)

    def rhs_lambda(self, x, lam) -> np.ndarray:
        return pf_rhs_lambda(self.eps, x, lam)

    def x_T(self, n: int = 1) -> np.ndarray:
        """Draw ``n`` initial conditions from the fully-noised marginal.

        At T=1, alpha(T) is negligible (~1e-5), so every mode's noised
        marginal collapses to the same N(0, sigma(T)^2 I) regardless of mu --
        matching :meth:`GaussianTier1.x_T`'s ``sqrt(var(1.0))`` in that limit.
        """
        return self.rng.standard_normal((n, self.d)) * sigma(1.0)


def swiss_roll(n: int = 2000, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """``n`` points on a 2D Swiss-roll curve, meant as :class:`MixtureTier2` mode
    centers: a curved manifold with an exact score once discretised into a
    point mixture (a *continuous* Swiss-roll density has none -- see the
    module docstring / guide's Part 3, item 2)."""
    rng = np.random.default_rng(seed)
    p = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    xy = np.stack([p * np.cos(p), p * np.sin(p)], axis=1) / 15.0
    return xy + noise * rng.standard_normal(xy.shape)


def dop853_reference(
    tb, x_T, T: float = 1.0, t_end: float = 1e-3, rtol: float = 1e-13, atol: float = 1e-14
) -> np.ndarray:
    """High-accuracy reference trajectory for a testbed with no closed form
    (Tier 2), by independently integrating ``tb.rhs_t`` with SciPy's DOP853.

    All rows of the batch ``x_T`` (shape ``(n, d)``) are integrated in one
    ``solve_ivp`` call, on one shared flattened state vector.

    ``rtol=1e-13`` is deliberately not the guide's nominal ``1e-14``: SciPy
    rejects rtol below about 2.2e-14 (100x machine epsilon). Callers should
    check convergence by recomputing at a tighter ``rtol`` (e.g. ``1e-11``)
    and confirming the two agree well below the smallest error being measured.
    """
    from scipy.integrate import solve_ivp

    x_T = np.atleast_2d(np.asarray(x_T, dtype=np.float64))
    n, d = x_T.shape

    def rhs(t, y):
        return tb.rhs_t(y.reshape(n, d), t).ravel()

    sol = solve_ivp(rhs, (T, t_end), x_T.ravel(), method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"DOP853 reference integration failed: {sol.message}")
    return sol.y[:, -1].reshape(n, d)
