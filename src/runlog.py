"""The one results schema for the whole project.

Every experiment appends rows to a CSV with exactly these columns, in this order.
Agreed at step 0 of the guide and not renegotiated later, so that any teammate's
notebook produces tables the others can concatenate and plot without adapters.

    tier      1 | 2 | 3
    testbed   free-form label, e.g. "gaussian", "mog8", "cifar10"
    arm       "A" (raw ODE in t) | "B" (lambda-ODE) | "C" (DPM-Solver)
    solver    "euler" | "midpoint" | "heun3" | "rk4" | "dpm1" | "dpm2" | "dpm3" | "ddim" ...
    order     theoretical convergence order of the solver (int)
    kappa     condition number s_max/s_min for Tier 1; NaN when not applicable
    h         step size actually used (in the arm's own time variable)
    nfe       number of function (score) evaluations actually spent; NaN if diverged
    err_l2    L2 distance of the final state to the reference; NaN/inf if diverged
    diverged  bool
    seed      RNG seed for the run

Usage
-----
>>> from src.runlog import append_row, load
>>> append_row("results/tier1_order.csv", tier=1, testbed="gaussian", arm="C",
...            solver="dpm3", order=3, kappa=100.0, h=0.05, nfe=60,
...            err_l2=1.2e-6, diverged=False, seed=0)
>>> df = load("results/tier1_order.csv")
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any

COLUMNS: tuple[str, ...] = (
    "tier",
    "testbed",
    "arm",
    "solver",
    "order",
    "kappa",
    "h",
    "nfe",
    "err_l2",
    "diverged",
    "seed",
)


def _fmt(value: Any) -> Any:
    """Render NaN/inf consistently so pandas reads them back as floats."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, bool):
        return int(value)
    return value


def append_row(csv_path: str, **fields: Any) -> None:
    """Append one row to ``csv_path``, writing the header first if the file is new.

    Raises if ``fields`` is not exactly ``COLUMNS`` — a missing or misspelled key
    is a bug we want to hear about immediately, not a silent blank cell.
    """
    missing = set(COLUMNS) - fields.keys()
    extra = fields.keys() - set(COLUMNS)
    if missing or extra:
        raise ValueError(
            f"row keys must be exactly {COLUMNS}; missing={sorted(missing)} extra={sorted(extra)}"
        )

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(COLUMNS)
        writer.writerow([_fmt(fields[col]) for col in COLUMNS])


def load(csv_path: str):
    """Read a results CSV into a pandas DataFrame with the canonical column order."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df[list(COLUMNS)]
