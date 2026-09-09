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

__all__ = ["crossover_h"]


def crossover_h(h1, err1, h3, err3, n_grid: int = 4000) -> np.ndarray:
    """All step sizes where the order-3 error curve crosses the order-1 curve.

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
    h1 = np.asarray(h1, dtype=np.float64)
    err1 = np.asarray(err1, dtype=np.float64)
    h3 = np.asarray(h3, dtype=np.float64)
    err3 = np.asarray(err3, dtype=np.float64)

    ok1 = np.isfinite(err1) & (err1 > 0) & np.isfinite(h1) & (h1 > 0)
    ok3 = np.isfinite(err3) & (err3 > 0) & np.isfinite(h3) & (h3 > 0)
    h1, err1 = h1[ok1], err1[ok1]
    h3, err3 = h3[ok3], err3[ok3]
    if len(h1) < 2 or len(h3) < 2:
        return np.array([], dtype=np.float64)

    i1, i3 = np.argsort(h1), np.argsort(h3)
    lh1, le1 = np.log(h1[i1]), np.log(err1[i1])
    lh3, le3 = np.log(h3[i3]), np.log(err3[i3])

    lo = max(lh1[0], lh3[0])
    hi = min(lh1[-1], lh3[-1])
    if lo >= hi:
        return np.array([], dtype=np.float64)

    grid = np.linspace(lo, hi, n_grid)
    diff = np.interp(grid, lh3, le3) - np.interp(grid, lh1, le1)

    sign = np.sign(diff)
    flips = np.nonzero(np.diff(sign) != 0)[0]
    if len(flips) == 0:
        return np.array([], dtype=np.float64)

    # linear interpolation of `diff` to the exact zero within each flip's bracket
    crossings = []
    for i in flips:
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        t = -y0 / (y1 - y0)
        crossings.append(np.exp(x0 + t * (x1 - x0)))
    return np.array(crossings, dtype=np.float64)
