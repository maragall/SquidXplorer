"""``stdev`` — a standard-deviation z-projection. The worked example.

A per-pixel standard deviation across the z-stack highlights pixels whose intensity varies with
depth and flattens pixels that are uniformly bright at every depth. It exercises every part of
the declaration at once:

    consumes = {"z"}          it collapses the z axis  -> output (T, C, 1, Y, X)
    produces = "intensity"    the pixels still measure light -> a napari Image layer
    params   = smooth_sigma, ddof     one entry, runnable at different settings
    requires = ("scipy",)     it imports scipy.ndimage, so it declares scipy

Every operator has one callable shape: ``operator(planes: Iterable[np.ndarray]) -> np.ndarray``,
where *planes* is an iterable of same-shape, same-dtype 2-D ``(Y, X)`` arrays and the return is
one 2-D array. ``consumes={"z"}`` hands you every z of one (t, c); ``consumes=set()`` hands you
exactly one plane (use ``squidxplorer.plane_op`` to lift a plane -> plane function).

Stream it: never ``list(planes)``. Welford's algorithm keeps this operator's footprint at two
float64 planes regardless of stack depth, and that is the contract, not an optimisation.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

#: Gaussian pre-smoothing in px; a raw per-pixel stdev is dominated by shot noise. 0.0 disables.
DEFAULT_SMOOTH_SIGMA: float = 1.0

#: Delta degrees of freedom. 0 = population standard deviation: the z-stack IS the population.
DEFAULT_DDOF: int = 0


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float result back to the acquisition's dtype, ROUNDING and CLIPPING.

    A bare ``astype`` truncates (dimming every pixel by half a count) and an unsigned wrap turns
    the dimmest pixels into the brightest. Return the acquisition's dtype unless you have a reason
    not to: viewer contrast rules are calibrated on the camera's range.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype, copy=False)


def stdev_op(smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
             ddof: int = DEFAULT_DDOF) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build the operator callable, bound to *smooth_sigma* and *ddof*. THE FACTORY.

    ``params=`` on the registration is what makes the registered object a factory: it is called
    with the declared defaults, and again with a run's ``operator_kwargs``, so one entry covers
    every setting. The closure holds the parameters (a module-level global would race across the
    worker pool). Bad parameters raise ValueError at BUILD time, before the run starts.
    """
    sigma = float(smooth_sigma)
    dof = int(ddof)
    if sigma < 0:
        raise ValueError(f"smooth_sigma must be >= 0 px, got {smooth_sigma!r} (0 disables it)")
    if dof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof!r}")

    def _stdev(planes: Iterable[np.ndarray]) -> np.ndarray:
        """Per-pixel standard deviation across the stack, in ONE streaming pass (Welford)."""
        count = 0
        mean = None
        m2 = None
        dtype = None

        for plane in planes:
            plane = np.asarray(plane)
            if dtype is None:
                dtype = plane.dtype
                mean = np.zeros(plane.shape, dtype=np.float64)
                m2 = np.zeros(plane.shape, dtype=np.float64)
            elif plane.shape != mean.shape:
                # A seam bug, and it must be loud: silently reducing over ragged planes produces a
                # plausible-looking image computed from the wrong data.
                raise ValueError(
                    f"stdev: plane {count} has shape {plane.shape}, expected {mean.shape}; "
                    "every plane of one (t, c) stack must be the same size")

            values = plane.astype(np.float64, copy=False)
            if sigma > 0:
                from scipy.ndimage import gaussian_filter   # lazy: see `requires` in __init__.py

                values = gaussian_filter(values, sigma=sigma)

            count += 1
            delta = values - mean
            mean += delta / count
            m2 += delta * (values - mean)

        if count == 0:
            raise ValueError(
                "stdev received an empty stack. A z-reducer is handed every z of one (t, c); "
                "zero planes means the reader produced nothing, which is never a valid input.")
        if count - dof <= 0:
            raise ValueError(
                f"stdev: ddof={dof} needs more than {dof} plane(s), and this stack has {count}. "
                f"A {count}-plane acquisition wants ddof=0 (the population standard deviation).")

        return _cast_like(np.sqrt(m2 / (count - dof)), dtype)

    return _stdev
