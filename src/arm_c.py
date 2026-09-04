"""Arm C via the authors' own code (guide step 6): a thin wrapper around
``third_party/dpm_solver_pytorch.py``.

``DPM_Solver`` only ever calls ``model_fn(x, t) -> noise``, so an analytic
oracle (``GaussianTier1.eps``, later ``MixtureTier2.eps``) drops in unmodified
where a neural network would go -- this is what lets Tiers 1-2 test the
*published* implementation with zero model error.

Critical settings, fixed here rather than left to the caller (guide step 6):
``algorithm_type="dpmsolver"`` (the paper's original parameterisation --
matches :func:`src.dpm.dpm_solver_1` exactly, not the data-prediction
``"dpmsolver++"`` default), ``method="singlestep_fixed"`` (``singlestep`` and
``multistep`` mix orders 1/2/3, so a convergence slope on them is meaningless,
Appendix D.3), ``skip_type="logSNR"`` (uniform in lambda, matching arm B and
:func:`src.grids.grid_lambda`), ``denoise_to_zero=False`` (adds an NFE and
moves the endpoint), float64 throughout.
"""

from __future__ import annotations

import numpy as np
import torch

from third_party.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP

__all__ = ["make_noise_schedule", "sample_dpm_solver"]

# `NoiseScheduleVP(..., dtype=...)` only reaches the 'discrete' schedule path;
# for 'linear' (what this project uses), get_time_steps() builds its own
# `torch.tensor(t_T)` / `torch.linspace(...)` with torch's *ambient* default
# dtype, silently dropping to float32 (~1e-7 error) regardless of the dtype
# passed here. Fixing it means patching vendored code we've committed to
# leaving untouched (third_party/README.md), so we fix the ambient default
# instead -- this project is float64-everywhere (guide, Convention 1) and
# touches no other torch code, so this is a safe, one-line global setting.
torch.set_default_dtype(torch.float64)


def make_noise_schedule() -> NoiseScheduleVP:
    """The one VP-linear schedule every arm shares (guide Part 2), in float64."""
    return NoiseScheduleVP(
        "linear", continuous_beta_0=0.1, continuous_beta_1=20.0, dtype=torch.float64
    )


def sample_dpm_solver(eps_fn, x0, t_start: float, t_end: float, order: int, steps: int):
    """Run fixed-order singlestep DPM-Solver-`order` from `t_start` to `t_end`.

    ``eps_fn(x, t) -> eps`` is the analytic (or network) noise oracle, called
    with numpy arrays and a python-float time; wrapped here to the tensor
    interface ``DPM_Solver`` expects.

    Returns ``(x_final, nfe)`` as ``(np.ndarray, int)``. NFE is the real cost
    of ``singlestep_fixed``, ``(steps // order) * order`` -- not `steps`.
    """
    ns = make_noise_schedule()

    def model_fn(x_t: torch.Tensor, t_t: torch.Tensor) -> torch.Tensor:
        t_scalar = float(t_t.reshape(-1)[0])
        return torch.from_numpy(np.asarray(eps_fn(x_t.numpy(), t_scalar), dtype=np.float64))

    solver = DPM_Solver(model_fn, ns, algorithm_type="dpmsolver")
    x = torch.from_numpy(np.asarray(x0, dtype=np.float64))
    x_final = solver.sample(
        x,
        steps=steps,
        t_start=t_start,
        t_end=t_end,
        order=order,
        skip_type="logSNR",
        method="singlestep_fixed",
        denoise_to_zero=False,
    )
    nfe = (steps // order) * order
    return x_final.numpy(), nfe
