"""Turning a sampler's output into something you can look at (M11, guide step 10
revisited). The project has never rendered a single image -- every earlier
milestone judged solvers purely on L2-to-reference. No plotting or GPU code
belongs here; this is the one pure array transform notebook 11 needs, kept
testable on the laptop like the rest of ``src/``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["to_uint8"]


def to_uint8(x) -> np.ndarray:
    """Map a ``[-1, 1]``-normalised NCHW batch to uint8 NHWC images.

    ``x`` is a torch tensor or numpy array of shape ``(n, c, h, w)`` -- the
    DDPM checkpoint's own convention (the same normalisation
    ``DDPMScheduler``/``diffusers`` trains and samples in). Clipped to
    ``[-1, 1]`` before scaling: a non-converged or diverged sample can land
    outside that range, and clipping (not wrapping) is what a display or an
    FID feature extractor does too, so the array shown is the array scored.
    """
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().astype(np.uint8)
    return np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
