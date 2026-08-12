"""``stdev`` — a standard-deviation z-projection. The worked example.

A real operator, not a stub: a per-pixel standard deviation across the z-stack highlights pixels
whose intensity VARIES with depth (edges, structure, anything in focus somewhere) and flattens
pixels that are uniformly bright at every depth (haze, hot pixels, an evenly illuminated
background). It is a genuine complement to ``mip``, which cannot tell a bright plane from a bright
object, and it is the shape of thing a lab actually adds.

It is also chosen because it exercises every part of the declaration at once:

    consumes = {"z"}          it collapses the z axis  -> output (T, C, 1, Y, X)
    produces = "intensity"    the pixels still measure light -> a napari Image layer
    params   = smooth_sigma, ddof     one entry, runnable at different settings
    requires = ("scipy",)     it imports scipy.ndimage, so it declares scipy

WHAT THE ENGINE HANDS YOU, AND WHAT YOU MUST HAND BACK
------------------------------------------------------
One callable shape, for every operator in this system::

    operator(planes: Iterable[np.ndarray]) -> np.ndarray

*planes* is an iterable of 2-D ``(Y, X)`` arrays, all the same shape and the same dtype, and the
return is ONE 2-D ``(Y, X)`` array. ``consumes`` alone decides what the engine puts in that
iterable:

    consumes={"z"}    every z of one (t, c) -- the whole stack. You return one plane; the engine
                      files it at z=0 and the result is (T, C, 1, Y, X).
    consumes=set()    exactly one plane. You return one plane; the engine keeps z at full depth
                      and the result is (T, C, Nz, Y, X). Use `squidxplorer.plane_op` to lift a
                      natural plane -> plane function into this shape.

STREAM IT. The engine runs several wells at once and its peak memory is (wells in flight) x (one
well's footprint). An operator that calls ``list(planes)`` materialises a whole z-stack per worker
and multiplies that number by the stack depth. Welford's algorithm computes an exact variance in
one pass with two accumulators, so this operator's footprint is two float64 planes regardless of
how deep the stack is. That is not an optimisation, it is the contract: ``mip`` is a running
maximum for the same reason.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

#: Default Gaussian pre-smoothing, in pixels. A per-pixel standard deviation over ~10 planes is
#: dominated by shot noise if it is taken raw -- the noise varies plane to plane, which is exactly
#: what this operator measures. A mild low-pass first is what makes the output show structure
#: rather than sensor noise. 0.0 turns it off (and then scipy is never imported, though it is
#: still DECLARED -- see the note on `requires` in __init__.py).
DEFAULT_SMOOTH_SIGMA: float = 1.0

#: Default delta degrees of freedom. 0 = the population standard deviation (divide by N), which is
#: what you want here: the z-stack IS the population, not a sample drawn from a larger one.
DEFAULT_DDOF: int = 0


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float result back to the acquisition's dtype, ROUNDING and CLIPPING.

    Copied deliberately from ``squidxplorer._background._cast_like`` rather than imported, because it
    is part of what a contributor has to know and hiding it behind an import would not teach it:

    * ``astype`` truncates toward zero, biasing every pixel down by half a count -- a systematic
      dimming of the entire dataset.
    * an unsigned WRAP turns the dimmest pixels of a frame into the brightest ones.

    Return the acquisition's dtype unless you have a reason not to. The writer accepts other
    dtypes, but every viewer contrast rule and every downstream comparison is calibrated on the
    camera's range.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype, copy=False)


def stdev_op(smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
             ddof: int = DEFAULT_DDOF) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build the operator callable, bound to *smooth_sigma* and *ddof*. THE FACTORY.

    This is what ``add_projector(..., params=...)`` registers. The presence of ``params=`` is what
    makes the registered object a factory: SquidXplorer calls it once with the declared defaults to
    build the entry's default binding, and again with a caller's ``operator_kwargs`` for a run that
    names different values. So ONE registry entry covers every setting, instead of one entry per
    frozen parameter set.

    The returned closure holds the parameters, which is why the engine can pass it around, call it
    from several threads at once, and never think about your settings. Keep it that way: a module
    level global that the closure reads is a race across the worker pool and a lie in the recipe.

    Raises
    ------
    ValueError
        On a negative sigma or a ddof that cannot be satisfied. At BUILD time, i.e. before the run
        starts, because a parameter refused per-plane is refused once per (t, c) per well and the
        console fills with the same sentence a thousand times.
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
