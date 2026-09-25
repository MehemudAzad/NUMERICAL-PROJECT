"""The crossover, guide step 11 / M8 -- this project's headline result.

Table 6 of the paper reports, without explanation, that DPM-Solver-3 is far
*worse* than DPM-Solver-1 at ~10 NFE on CIFAR-10. That is textbook
pre-asymptotic behaviour: at large step size h the order-3 Taylor expansion
isn't valid yet, so its higher-order correction terms actively hurt. Tiers 1
and 2 have exact answers and a continuous h to sweep, so the crossover step
size h* -- below which order 3 beats order 1, above which it loses -- can be
located directly, without guessing at a network's behaviour.

Two error curves rarely cross exactly once in practice: DPM-Solver-3's own
sub-node structure can make its error non-monotonic at very coarse grids (the
guide's "pre-asymptotic" regime, quite literally), so :func:`crossover_h`
reports *every* sign change of ``log(err3) - log(err1)`` within the overlap
of the two swept ranges, rather than assuming there is exactly one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["crossover_h", "crossover_nfe", "crossover_x"]


def _crossover_x(x1, err1, x3, err3, n_grid: int) -> np.ndarray:
    """Axis-agnostic core: every point on ``x`` where the order-3 error curve
    crosses the order-1 curve. See :func:`crossover_h` for the full contract;
    ``x`` is step size ``h`` there and measured NFE in :func:`crossover_nfe`.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    err1 = np.asarray(err1, dtype=np.float64)
    x3 = np.asarray(x3, dtype=np.float64)
    err3 = np.asarray(err3, dtype=np.float64)

    ok1 = np.isfinite(err1) & (err1 > 0) & np.isfinite(x1) & (x1 > 0)
    ok3 = np.isfinite(err3) & (err3 > 0) & np.isfinite(x3) & (x3 > 0)
    x1, err1 = x1[ok1], err1[ok1]
    x3, err3 = x3[ok3], err3[ok3]
    if len(x1) < 2 or len(x3) < 2:
        return np.array([], dtype=np.float64)

    i1, i3 = np.argsort(x1), np.argsort(x3)
    lx1, le1 = np.log(x1[i1]), np.log(err1[i1])
    lx3, le3 = np.log(x3[i3]), np.log(err3[i3])

    lo = max(lx1[0], lx3[0])
    hi = min(lx1[-1], lx3[-1])
    if lo >= hi:
        return np.array([], dtype=np.float64)

    grid = np.linspace(lo, hi, n_grid)
    diff = np.interp(grid, lx3, le3) - np.interp(grid, lx1, le1)

    sign = np.sign(diff)
    flips = np.nonzero(np.diff(sign) != 0)[0]
    if len(flips) == 0:
        return np.array([], dtype=np.float64)

    # linear interpolation of `diff` to the exact zero within each flip's bracket
    crossings = []
    for i in flips:
        x0, x1_ = grid[i], grid[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        t = -y0 / (y1 - y0)
        crossings.append(np.exp(x0 + t * (x1_ - x0)))
    return np.array(crossings, dtype=np.float64)


def crossover_h(h1, err1, h3, err3, n_grid: int = 4000) -> np.ndarray:
    """All step sizes where the order-3 error curve crosses the order-1 curve.

    Both curves are compared at **matched step size** ``h`` -- DPM-Solver-3
    spends 3x the NFE of DPM-Solver-1 per macro step here, so this is not the
    same comparison the paper's Table 6 makes (matched NFE): see
    :func:`crossover_nfe` for that.

    Both curves are linearly interpolated in log-log space (piecewise-linear
    between the actual swept points -- no smoothing, no power-law
    extrapolation) onto a common fine log-spaced grid covering the *overlap*
    of ``[min(h), max(h)]`` for the two inputs. Returns every ``h`` where the
    sign of ``log(err3) - log(err1)`` flips, ascending in ``h``; an empty
    array if the ranges don't overlap or the curves never cross there.

    A crossing at index ``i`` is order 3 losing (large h, above h*) turning
    into order 3 winning (small h, below h*) if
    ``err3(h) < err1(h)`` just below the returned ``h``, matching the guide's
    "h* below which order 3 beats order 1, above which it loses" -- callers
    can check this directly by re-evaluating the interpolants, since a
    non-monotonic curve can flip the other way at a later crossing too.
    """
    return _crossover_x(h1, err1, h3, err3, n_grid)


def crossover_nfe(nfe1, err1, nfe3, err3, n_grid: int = 4000) -> np.ndarray:
    """All **measured NFE** where the order-3 error curve crosses the order-1
    curve -- the paper's own comparison (Table 6 plots FID against NFE, not h).

    Same interpolate-and-find-sign-flips method as :func:`crossover_h`, on the
    NFE axis instead of h (M12.3, guide step 11 revisited): DPM-Solver-1 and
    DPM-Solver-3 are compared at equal *cost*, not equal step count, so a
    crossing here is where spending one more network call starts favouring the
    higher-order solver. Pass the *measured* ``nfe`` column (``src.runlog``'s
    schema), not an assumed ``steps * order`` -- `singlestep_fixed`'s real cost
    is ``(steps // order) * order``.
    """
    return _crossover_x(nfe1, err1, nfe3, err3, n_grid)
