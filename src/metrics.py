"""Order fitting (guide step 7): turn an error-vs-h sweep into a measured slope.

A real convergence curve has three regions: pre-asymptotic at large h (theory
doesn't apply yet), the true slope in the middle, and a round-off floor at tiny
h where float64 noise (~1e-13 for these problems) takes over and the curve
flattens or turns up. Fitting a single line through all points averages across
regions and reports the wrong order. :func:`fit_order` instead searches every
contiguous log-log subwindow and keeps the one that is both straight (best R^2)
and wide, which finds the asymptotic region without being told where it is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_order", "l2"]


def fit_order(hs, errs, min_pts: int = 4, floor: float | None = None) -> dict:
    """Sliding-window log-log fit of ``errs`` against ``hs``.

    Returns ``dict(slope, r2, window, n)`` for the straightest sufficiently-wide
    window, which auto-excludes the pre-asymptotic and round-off regions. Non-
    finite or non-positive errors (a diverged run, or an exact 0.0) are dropped
    before fitting. ``slope``/``r2``/``window`` are ``nan``/``nan``/``None`` if
    fewer than ``min_pts`` usable points remain.

    ``floor`` drops every point at or below a *known* error floor before
    fitting. The window search alone is not always enough: it scores windows as
    ``r2 + 0.01 * width``, so a wide window that runs into a flat floor can beat
    a narrower clean one, and the reported slope is then biased toward zero.
    Tier 3 is exactly this case -- float32 plus the network's own non-smoothness
    put a hard floor under every curve, and the guide's step 10 is explicit
    about it: *"Do not fit slopes through the flat part."* Pass a few times the
    measured floor (``3 * floor`` is what this project uses) when one is known;
    leave it ``None`` when it is not, as on Tiers 1-2 where the round-off floor
    sits far below every measured point.
    """
    hs = np.asarray(hs, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    ok = np.isfinite(errs) & (errs > 0)
    if floor is not None:
        ok &= errs > float(floor)
    hs, errs = hs[ok], errs[ok]
    if len(hs) < min_pts:
        return dict(slope=np.nan, r2=np.nan, window=None, n=len(hs))

    lx, ly = np.log(hs), np.log(errs)
    best = None
    n = len(lx)
    for i in range(n - min_pts + 1):
        for j in range(i + min_pts, n + 1):
            x, y = lx[i:j], ly[i:j]
            m, c = np.polyfit(x, y, 1)
            resid = y - (m * x + c)
            r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else 0.0
            # Prefer wide windows, but only if they stay straight.
            score = r2 + 0.01 * (j - i)
            if best is None or score > best["score"]:
                best = dict(
                    score=score,
                    slope=float(m),
                    r2=float(r2),
                    window=(float(hs[i]), float(hs[j - 1])),
                    n=j - i,
                )
    best.pop("score")
    return best


def l2(x, ref) -> float:
    """Euclidean distance between a final state and its reference."""
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64) - np.asarray(ref, dtype=np.float64)))
