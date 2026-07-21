"""Richardson-Lucy deconvolution as a PLANE-OP (IMA-223).

One ``add_projector`` call, zero engine edits: this module declares ``consumes=frozenset()``
via :func:`squidmip.plane_op`, so ``project_well``'s existing group-by-then-reduce loop hands
it ONE plane at a time and z survives at full depth (IMA-210). Nothing in ``_engine.py`` knows
this operator exists.

IMA-236 is a separate ticket (standing decon up in Hailing's lab). This is the operator only.

Prior art surveyed before writing
---------------------------------
* **scikit-image** ``restoration.richardson_lucy`` — INSTALLED (0.26) and used here as the
  cross-check reference in ``tests/test_decon.py``, not as the runtime path. Two reasons it is
  not the runtime path: (a) it requires an explicit PSF **array** and convolves it with
  ``scipy.signal.convolve(..., mode="same")``, i.e. **zero padding** at the border, which on a
  fluorescence tile darkens the rim and then RL amplifies that dark rim into a visible frame;
  (b) for a Gaussian PSF the array and the general convolution are both unnecessary — see the
  shortcut below. We match its update rule exactly (pinned by a test on the image interior).
* **flowdec** (TensorFlow) and **RedLionfish** (CUDA/PyOpenCL) — NOT installed, and neither is
  worth a GPU-stack dependency for a CPU-side per-plane operator that already runs at ~1 s for
  10 iterations on a 2000px plane. Both are FFT-based and would bring the same wrap-around
  boundary problem as (a) unless padded manually. Revisit under IMA-236 if lab throughput
  needs it.
* **DeconvolutionLab2 / ImageJ RL** — the reference for the *boundary* handling: mirror-pad
  before the transform. That is what ``mode="reflect"`` gives us for free here.

The Gaussian shortcut: no kernel array at all
---------------------------------------------
The RL update is

    x_{k+1} = x_k * H^T( y / (H x_k) )

For a Gaussian PSF, ``H`` is a Gaussian blur — and a Gaussian is **symmetric**, so ``H^T`` is
the *same* blur. One iteration is therefore literally TWO calls to
``scipy.ndimage.gaussian_filter``: no PSF array is ever built, no FFT is taken, and the cost is
separable ``O(N·sigma)`` instead of ``O(N log N)`` over a padded array. It also makes flux
conservation exact: with reflect padding ``H`` is doubly stochastic, so ``H^T 1 = 1`` and
``sum(x_{k+1}) = sum(y)`` at every iteration (pinned by a test).

THE BOUNDARY TRAP — numpy's "reflect" is NOT scipy's "reflect"
--------------------------------------------------------------
=================  ===========================  ==================
padding of a b c d  scipy.ndimage name           numpy.pad name
=================  ===========================  ==================
d c b a | a b c d   ``reflect``  (the default)   ``symmetric``
d c b   | a b c d   ``mirror``                   ``reflect``
=================  ===========================  ==================

The default here is scipy's ``reflect`` (= numpy's ``symmetric``), which duplicates the edge
sample. Getting this wrong does not crash — it produces a bright or dark rim that RL then
amplifies over the iterations into something that looks exactly like real membrane signal at
the FOV edge. ``tests/test_decon.py`` pins the numpy/scipy equivalence AND pins that a constant
image stays constant to its outermost row, which is the sharpest available boundary check.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from squidmip._engine import add_projector
from squidmip.projection import plane_op

# Defaults for the registered ``decon`` operator. A Gaussian sigma of ~1.5 px is the widefield
# PSF at the 10x/0.752 um-per-px working point (Abbe lateral resolution ~0.5 um at 500 nm /
# NA 0.4 => sigma ~1.3 px, rounded up because the measured PSF is always wider than the
# diffraction limit). 10 iterations is the usual widefield stopping point: RL amplifies noise
# monotonically past roughly 20-30, so the default sits well short of that.
DEFAULT_SIGMA_PX: float = 1.5
DEFAULT_ITERATIONS: int = 10

# scipy's name for "duplicate the edge sample when padding" — numpy calls this same padding
# "symmetric". See the module docstring's table; do not "fix" this to "mirror".
DEFAULT_BOUNDARY_MODE: str = "reflect"

_EPS = 1e-9   # keeps the y/(Hx) ratio finite where the blurred estimate underflows to 0


def richardson_lucy_gaussian(
    plane: np.ndarray,
    sigma: float = DEFAULT_SIGMA_PX,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    mode: str = DEFAULT_BOUNDARY_MODE,
    init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Deconvolve ONE plane with a Gaussian PSF by Richardson-Lucy, kernel-free.

    Parameters
    ----------
    plane:
        2-D image, any dtype. The caller's array is never mutated.
    sigma:
        Gaussian PSF standard deviation in **pixels**. Must be > 0.
    iterations:
        RL iterations. ``0`` is the identity (a plain copy) so "no deconvolution" has an
        unambiguous spelling and a benchmark has a zero point.
    mode:
        Boundary handling, passed to ``scipy.ndimage.gaussian_filter``. Default ``"reflect"``
        — scipy's spelling of numpy's ``"symmetric"``. READ the module docstring before
        changing this.
    init:
        Starting estimate. ``None`` (default) starts from *plane* itself, which converges in
        fewer iterations than a flat seed on sparse fluorescence. Pass an array to match another
        implementation's seed (scikit-image starts from a constant 0.5).

    Returns
    -------
    np.ndarray
        Same shape and dtype as *plane*. Integer dtypes are **clipped** to the dtype's range
        before the cast, never wrapped — an RL overshoot on a saturated punctum would otherwise
        turn the brightest pixel in the frame into a black one.
    """
    from scipy.ndimage import gaussian_filter   # lazy: keeps the headless import graph light

    if sigma <= 0:
        raise ValueError(f"sigma must be > 0 pixels, got {sigma}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if plane.ndim != 2:
        raise ValueError(f"richardson_lucy_gaussian takes ONE 2-D plane; got shape {plane.shape}")
    if iterations == 0:
        return np.array(plane, copy=True)

    observed = plane.astype(np.float32, copy=True)
    estimate = (observed.copy() if init is None
                else np.array(init, dtype=np.float32, copy=True))
    if estimate.shape != observed.shape:
        raise ValueError(f"init shape {estimate.shape} != plane shape {observed.shape}")

    for _ in range(iterations):
        blurred = gaussian_filter(estimate, sigma, mode=mode)
        ratio = observed / (blurred + _EPS)
        # H^T == H for a symmetric kernel with symmetric padding: the SAME blur, no PSF array.
        estimate *= gaussian_filter(ratio, sigma, mode=mode)

    return _cast_like(estimate, plane.dtype)


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast back to the acquisition dtype, clipping integers instead of wrapping them."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(values, info.min, info.max)
    return values.astype(dtype, copy=False)


def deconvolve(plane: np.ndarray) -> np.ndarray:
    """Deconvolve one plane with the module defaults — the function behind the ``decon`` name."""
    return richardson_lucy_gaussian(plane, DEFAULT_SIGMA_PX, DEFAULT_ITERATIONS)


def decon_op(
    sigma: float = DEFAULT_SIGMA_PX,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    mode: str = DEFAULT_BOUNDARY_MODE,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a parameterised deconvolution **plane-op**, ready for ``add_projector``::

        add_projector("decon_sharp", decon_op(sigma=1.0, iterations=25))

    The returned callable carries ``consumes = frozenset()`` (stamped by
    :func:`squidmip.plane_op`), so the registry infers the declaration and z survives.
    """
    def _decon(p: np.ndarray) -> np.ndarray:
        return richardson_lucy_gaussian(p, sigma, iterations, mode=mode)

    _decon.__name__ = f"decon(sigma={sigma},iterations={iterations})"
    return plane_op(_decon)


# The whole registration. No engine edit — that is the IMA-210 seam working as designed.
add_projector("decon", plane_op(deconvolve))
