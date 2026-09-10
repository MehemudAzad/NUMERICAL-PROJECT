"""Tier 3 (guide step 10): the real CIFAR-10 network.

Tiers 1 and 2 drive the solvers with an *analytic* noise oracle, so their only
error is discretization error. Tier 3 swaps that oracle for a real network --
``google/ddpm-cifar10-32``, d = 3072 -- and asks whether the order, stability
and crossover results survive contact with it. The reference stops being
algebra and becomes a fine-grid run of the *same* checkpoint from the *same*
``x_T``, so what is measured is still discretization error, not model error.

This module is Tier 3's counterpart to :mod:`src.arm_c`: the schedule adapter
and the network oracle, no sweeps and no plotting (those live in
``notebooks/09_tier3_cifar10.ipynb``, which runs on a Kaggle T4).

The schedule question
---------------------
The checkpoint is a *discrete-time* DDPM: 1000 betas, and DPM-Solver reaches it
through ``NoiseScheduleVP('discrete', betas=...)``, which interpolates
``log alpha`` linearly between the 1000 tabulated times. That is the same
VP-linear schedule :mod:`src.schedule` implements, but discretised -- the two
disagree by O(1/N) (about 0.03 in lambda, on a span of ~10). Arms A and B need
``f``, ``g^2``, ``sigma``, ``lambda`` and ``t(lambda)`` too, and mixing the
continuous forms with the checkpoint's discrete one would inject that O(1/N)
schedule mismatch straight into their error curves. So :class:`DiscreteSchedule`
derives *everything* from the same ``NoiseScheduleVP`` object the solver uses,
and all three arms provably share one schedule -- the property Gate G1 and M4
established for Tiers 1-2, carried into Tier 3.

Only one derivative has to be approximated, and the identity below makes it a
single finite difference rather than two independent ones:

    lambda = log alpha - log sigma,  sigma^2 = 1 - alpha^2
    => sigma' = -alpha^2 f / sigma            (differentiate sigma^2)
    => lambda' = f - sigma'/sigma = f (1 + alpha^2/sigma^2) = f / sigma^2

so, for **any** variance-preserving schedule,

    f(t)  = sigma(t)^2 * lambda'(t)
    g^2(t) = -2 sigma(t)^2 lambda'(t) = -2 f(t)

(check against the continuous form: f = -beta/2, g^2 = beta = -2f). Both come
from one central difference of ``lambda``, which means ``f / lambda' == sigma^2``
holds *identically* rather than approximately -- and that identity is exactly
what makes the arm-A and arm-B right-hand sides describe the same ODE. Arms A
and B therefore agree by construction, not by luck; ``tests/test_tier3.py``
checks it end to end.

Two dtypes, on purpose
----------------------
``DPM_Solver`` multiplies schedule coefficients into the state, so a float64
schedule would silently promote a float32 CUDA batch to float64 and then fail
in the float32 UNet. Arm C therefore gets a **float32** schedule. The finite
difference above, on the other hand, is hopeless in float32 (lambda ~ 5,
dt = 1e-5, so float32 rounding alone costs ~0.03 in the derivative), so
:class:`DiscreteSchedule` wraps a **float64, CPU** schedule and returns plain
Python floats. Both are built from the same betas; they differ only by float32
rounding (~1e-7), which is four orders of magnitude below Tier 3's ~1e-3 floor.

.. warning::
   Never import :mod:`src.arm_c` from Tier-3 code. It sets
   ``torch.set_default_dtype(torch.float64)`` as a module-level side effect (the
   only way to fix the vendored solver's float32 timesteps without editing it --
   see CLAUDE.md section 8). On Tier 3 that would build the UNet in float64 and
   make ``get_time_steps`` emit float64 grids. ``sample_dpm_solver_t3`` asserts
   the ambient default is still float32 so this fails loudly, not at 2 a.m.
"""

from __future__ import annotations

import numpy as np
import torch

from third_party.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper

__all__ = [
    "make_noise_schedule",
    "DiscreteSchedule",
    "make_model_fn",
    "make_eps_fn",
    "sample_dpm_solver_t3",
]


def make_noise_schedule(betas, dtype=torch.float32, device=None) -> NoiseScheduleVP:
    """A ``NoiseScheduleVP('discrete')`` from the checkpoint's own betas.

    Call it twice: once at ``torch.float32`` on the GPU for ``DPM_Solver`` and
    ``model_wrapper`` (arm C), once at ``torch.float64`` on the CPU to hand to
    :class:`DiscreteSchedule` (arms A and B). See the module docstring for why.
    """
    b = torch.as_tensor(betas, dtype=torch.float64)
    if device is not None:
        b = b.to(device)
    return NoiseScheduleVP("discrete", betas=b, dtype=dtype)


class DiscreteSchedule:
    """Arm-A/B schedule coefficients read off a discrete ``NoiseScheduleVP``.

    Exposes the same names as :mod:`src.schedule` (``alpha``, ``sigma``,
    ``lmbda``, ``t_of_lmbda``, ``f``, ``g2``) so it can be passed as the
    ``sched=`` argument of :func:`src.testbeds.pf_rhs_t` and
    :func:`src.testbeds.pf_rhs_lambda` with nothing else changing.

    Every method takes a scalar (or array) of times and returns plain Python
    floats (or a numpy array) -- never tensors. The solvers multiply these into
    the state, so keeping them as scalars is what preserves the state's float32
    dtype and CUDA device.

    Parameters
    ----------
    ns : NoiseScheduleVP
        Must be the ``'discrete'`` schedule, **built at float64** -- the central
        difference in :meth:`dlmbda_dt` is meaningless otherwise.
    dt : float
        Central-difference step for ``lambda'``. The default 1e-5 sits well
        inside one segment of the interpolant, whose knots are ``1/total_N``
        (1e-3) apart, so the stencil almost never straddles a knot.
    """

    def __init__(self, ns: NoiseScheduleVP, dt: float = 1e-5):
        if ns.schedule != "discrete":
            raise ValueError("DiscreteSchedule wraps a 'discrete' NoiseScheduleVP")
        if ns.log_alpha_array.dtype != torch.float64:
            raise ValueError(
                "the schedule handed to DiscreteSchedule must be float64; a float32 "
                "one makes lambda'(t) useless (see the module docstring)"
            )
        self.ns = ns
        self.dt = float(dt)
        self.total_N = int(ns.total_N)
        # The discrete schedule tabulates t on linspace(0, 1, N+1)[1:], so 1/N is
        # its *floor*: below it, `interpolate_fn` extrapolates off the table.
        # t_end can approach this value but must not go under it.
        self.t_min = 1.0 / self.total_N
        self.T = float(ns.T)

    # -- plumbing ---------------------------------------------------------

    @staticmethod
    def _scalar_in(t):
        return np.ndim(t) == 0

    def _tensor(self, t) -> torch.Tensor:
        return torch.as_tensor(
            np.atleast_1d(np.asarray(t, dtype=np.float64)), dtype=torch.float64
        )

    @staticmethod
    def _out(v: torch.Tensor, scalar: bool):
        arr = v.detach().cpu().numpy().astype(np.float64).reshape(-1)
        return float(arr[0]) if scalar else arr

    # -- exact quantities, straight off the interpolant --------------------

    def alpha(self, t):
        """Signal coefficient alpha(t)."""
        return self._out(self.ns.marginal_alpha(self._tensor(t)), self._scalar_in(t))

    def sigma(self, t):
        """Noise std sigma(t) = sqrt(1 - alpha(t)^2)."""
        return self._out(self.ns.marginal_std(self._tensor(t)), self._scalar_in(t))

    def lmbda(self, t):
        """Half log-SNR lambda(t) = log alpha - log sigma. Decreasing in t."""
        return self._out(self.ns.marginal_lambda(self._tensor(t)), self._scalar_in(t))

    def t_of_lmbda(self, lam):
        """Inverse of :meth:`lmbda`, by the schedule's own table inversion."""
        return self._out(self.ns.inverse_lambda(self._tensor(lam)), self._scalar_in(lam))

    # -- the one finite difference, and the two coefficients it gives ------

    def dlmbda_dt(self, t):
        """lambda'(t) by central difference, clipped to the tabulated range.

        Clipping to ``[t_min, T]`` (and dividing by the *clipped* width, so the
        formula degrades to a one-sided difference at the ends) keeps the
        stencil from stepping off the interpolation table at the stiff boundary,
        which is precisely where arms A and B are being stressed.
        """
        scalar = self._scalar_in(t)
        tv = np.atleast_1d(np.asarray(t, dtype=np.float64))
        lo = np.clip(tv - self.dt, self.t_min, self.T)
        hi = np.clip(tv + self.dt, self.t_min, self.T)
        out = (np.atleast_1d(self.lmbda(hi)) - np.atleast_1d(self.lmbda(lo))) / (hi - lo)
        return float(out[0]) if scalar else out

    def f(self, t):
        """PF-ODE linear drift f(t) = d log alpha / dt = sigma(t)^2 lambda'(t)."""
        s = self.sigma(t)
        return s**2 * self.dlmbda_dt(t)

    def g2(self, t):
        """Squared diffusion g^2(t) = -2 sigma(t)^2 lambda'(t) = -2 f(t)."""
        return -2.0 * self.f(t)


# --- the network as a noise oracle -----------------------------------------


def make_model_fn(unet, ns: NoiseScheduleVP, model_kwargs: dict | None = None):
    """Wrap a diffusers ``UNet2DModel`` into the ``model_fn(x, t) -> eps`` DPM-Solver wants.

    ``UNet2DModel`` returns a dataclass, not a tensor -- hence the ``.sample``.
    Forgetting it is the guide's first "costs you a day" gotcha.

    ``ns`` must be the **float32** schedule: ``model_wrapper`` uses it to map
    continuous ``t`` to the discrete-time label the checkpoint was trained on,
    ``(t - 1/N) * 1000``.
    """
    return model_wrapper(
        lambda x, t: unet(x, t).sample,
        ns,
        model_type="noise",
        model_kwargs=model_kwargs or {},
    )


def make_eps_fn(model_fn):
    """Adapt ``model_fn(x, t_tensor)`` to the project's ``eps_fn(x, t_scalar)`` oracle.

    This is the signature :class:`src.testbeds.GaussianTier1.eps` has, so arms A
    and B march the network through :func:`src.solvers.integrate` with exactly
    the code that marched the analytic tiers. The scalar node supplied by the
    grid is broadcast to one time per batch element, and gradients are off --
    nothing here is ever backpropagated.
    """

    def eps_fn(x, t):
        t_t = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        with torch.no_grad():
            return model_fn(x, t_t)

    return eps_fn


def sample_dpm_solver_t3(model_fn, ns: NoiseScheduleVP, x_T, t_start, t_end, order, steps):
    """Arm C on Tier 3: fixed-order singlestep DPM-Solver-`order`, uniform in lambda.

    Same settings as :func:`src.arm_c.sample_dpm_solver` (guide step 5) --
    ``algorithm_type="dpmsolver"``, ``method="singlestep_fixed"`` so orders are
    not mixed, ``skip_type="logSNR"`` so h is constant, ``denoise_to_zero=False``
    so the endpoint does not move -- but driven by a network instead of an
    analytic oracle, and with no global dtype side effect.

    Returns ``(x_final, nfe)``. As on Tiers 1-2, ``singlestep_fixed``'s real cost
    is ``(steps // order) * order``, which is what gets recorded; pass a `steps`
    that is a multiple of `order` to spend the whole budget.
    """
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError(
            f"torch default dtype is {torch.get_default_dtype()}, expected float32. "
            "Something imported src.arm_c, which sets float64 globally; on Tier 3 "
            "that promotes the solver's time grids and breaks the float32 UNet. "
            "Restart the kernel and do not import src.arm_c here."
        )
    solver = DPM_Solver(model_fn, ns, algorithm_type="dpmsolver")
    with torch.no_grad():
        x_final = solver.sample(
            x_T,
            steps=steps,
            t_start=t_start,
            t_end=t_end,
            order=order,
            skip_type="logSNR",
            method="singlestep_fixed",
            denoise_to_zero=False,
        )
    return x_final, (steps // order) * order
