"""Time grids for arms A and B (guide step 4).

Arm A marches in ``t`` on a uniform-``t`` grid; arm B marches in ``lambda`` on a
uniform-``lambda`` grid (same grid arm C's DPM-Solver uses, ``skip_type='logSNR'``).
Both run from ``T`` (most noise) down to ``t_end`` (least noise); since
``lambda(t)`` is strictly decreasing in ``t`` (schedule.py), :func:`grid_lambda`
is strictly *increasing*.
"""

from __future__ import annotations

import numpy as np

from .schedule import lmbda

__all__ = ["grid_t", "grid_lambda"]


def grid_t(T, t_end, n: int) -> np.ndarray:
    """Uniform grid in t: n steps, n+1 nodes, descending from T to t_end."""
    return np.linspace(float(T), float(t_end), n + 1)


def grid_lambda(T, t_end, n: int) -> np.ndarray:
    """Uniform grid in lambda: n steps, n+1 nodes, from lambda(T) to lambda(t_end)."""
    return np.linspace(lmbda(T), lmbda(t_end), n + 1)
