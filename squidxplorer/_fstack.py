"""All-in-focus z fusion: noise-robust selective focus stacking (Pertuz SAF).

Port of ``fstack.m`` (S. Pertuz, Jan/2016) — the algorithm of Pertuz et al., "Generation
of all-in-focus images by noise-robust selective fusion of limited depth-of-field
images", IEEE Trans. Image Process. 22(3):1242-1251, 2013. Registered as ``fstack``, a
z-reducer: per (FOV, channel, t) it consumes ALL z planes and returns ONE fused plane.

Faithful to the MATLAB math — the local-variance focus measure, the 3-point log-domain
Gaussian interpolation across z (STEP=2, the end clamps AND the Index1/Index3 end-swaps,
which make the third log sample equal the first at EVERY pixel, so the fitted Gaussian
always peaks at the clamped argmax frame), the PSNR-based selectivity and the tanh
weighting. The fit is carried in ``s^2`` (:func:`gauss3P` returns it), which is the only
form the MATLAB ever uses ``s`` in: where the curvature is positive (an end-of-stack argmax
over a non-monotone profile: 10.6% of a real G7 488 field, measured 2026-08-26) MATLAB's
``sqrt`` goes complex and its arithmetic continues in a real, NEGATIVE ``s.^2``, giving an
inverted Gaussian model whose error is finite and large. Taking ``sqrt`` for real made
those pixels NaN and handed them the whole image's minimum selectivity (6.4 dB, phi 0.33,
neither the MATLAB's value nor the mean), coupling every window to the field it sits in.
Three deliberate divergences, decided with Julio (2026-08-25):

- native dtype preserved: the MATLAB casts the fusion to uint8; here the blend runs in
  float64 and rounds+clips back to the input dtype (``cast_like``). The weighted fusion
  is a convex combination, so values stay inside each pixel's own z range — no wrap.
- PER-CHANNEL application: each fluorescence channel computes its OWN focus, selectivity
  and weights from its own planes. The MATLAB's one-graymap-for-RGB is for color
  photographs of one scene; our channels image different structures.
- model-free: no PSF, no optics, no wavelength — it runs on broadband and brightfield
  channels unchanged.

Two interpretations where the MATLAB is degenerate rather than defined:

- a stack of fewer than ``2*STEP + 1 = 5`` planes is refused BY NAME (the MATLAB indexes
  out of bounds there);
- a pixel with zero focus measure in every frame (or a selectivity with no finite value)
  fuses as the plain mean of its planes, where the MATLAB casts NaN to uint8 0.

Boundary handling: ``uniform_filter(mode="nearest")`` is imfilter 'replicate'; the 3x3
median's mode="nearest" is value-identical to scipy's default reflect at size 3 (MATLAB's
medfilt2 zero-pads instead, which zeroes selectivity in the outermost corner pixels — a
boundary artifact deliberately not imported). The selectivity window is a DIRECT separable
convolution (:func:`_windowed_mean`), imfilter's own arithmetic: a NaN, an inf or a huge
error reaches exactly the windows containing it. ``uniform_filter`` is a running sum, and
its rounding residual after a tainted value leaves the window never returns to zero, so a
mask built from it (``> 0``) stayed set along the rest of the line. That was the
horizontal banding Julio saw on G7 (2026-08-26): 830,866 px of over-reach in one 488
field, in runs up to 2004 px along rows (``uniform_filter``'s last axis) against 517 along
columns, every over-reached pixel fused as the plain mean of its planes beside neighbours
that were not.

The operator DECLARES its lateral support as ``halo_px`` (:func:`lateral_halo_px`, ruling
z): an ROI window solved with that halo equals the whole-field result inside the window.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import convolve1d, median_filter, uniform_filter

from squidxplorer._engine import Param, add_operator
from squidxplorer.projection import Z_REDUCER, cast_like

DEFAULT_NHSIZE = 9      #: focus-measure window, px (the MATLAB's 'nhsize')
DEFAULT_ALPHA = 0.2     #: selectivity sharpness, (0, 1] (the MATLAB's 'alpha')
DEFAULT_STH = 13.0      #: selectivity threshold, dB (the MATLAB's 'sth')

_STEP = 2               #: gauss3P's internal frame stride (the MATLAB's STEP)
MIN_PLANES = 2 * _STEP + 1


def gfocus(im: np.ndarray, nhsize: int) -> np.ndarray:
    """Gray-level local variance focus measure, float64, replicate borders."""
    im = np.asarray(im, dtype=np.float64)
    mean = uniform_filter(im, size=nhsize, mode="nearest")
    return uniform_filter((im - mean) ** 2, size=nhsize, mode="nearest")


def _take_z(fm: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``fm[z[y, x], y, x]`` — one value per pixel from the per-pixel z index map."""
    return np.take_along_axis(fm, z[None], axis=0)[0]


def gauss3P(focus: np.ndarray, fm: np.ndarray):
    """Per-pixel 3-point Gaussian interpolation across z; ``fm`` is (P, Y, X).

    Returns ``(u, s2, A, fmax)``: the fitted mean, the fitted width SQUARED, the amplitude
    and the per-pixel focus-measure maximum. The index clamps and the Index1/Index3
    end-swaps are the MATLAB's own, verbatim (0-based): after them the third log sample
    equals the first at every pixel. ``s2`` is the MATLAB's ``s.^2``, the only form it
    uses ``s`` in; it is NEGATIVE where the curvature is positive (the MATLAB's complex
    ``s``), and the model there is an inverted Gaussian with a finite, large error, as the
    MATLAB computes it. NaN only where the fit is undefined (a zero sample under the log,
    or a flat pair making the curvature 0); the caller's selectivity path absorbs it.
    """
    focus = np.asarray(focus, dtype=np.float64)
    P = fm.shape[0]
    if P < MIN_PLANES:
        raise ValueError(
            f"gauss3P needs at least {MIN_PLANES} frames (it samples {_STEP} frames each "
            f"side of the peak); got {P}")
    fmax = fm.max(axis=0)
    idx = np.argmax(fm, axis=0)
    ic = idx.copy()
    ic[ic < _STEP] = _STEP                      # MATLAB: Ic(Ic<=STEP)   = STEP+1
    ic[ic >= P - _STEP - 1] = P - _STEP - 1     # MATLAB: Ic(Ic>=P-STEP) = P-STEP
    z1, z2, z3 = ic - _STEP, ic, ic + _STEP
    # The end-swaps, in the MATLAB's own order — the second reads the UPDATED Index1:
    #   Index1(I<=STEP) = Index3(I<=STEP);  Index3(I>=STEP) = Index1(I>=STEP)
    low = idx < _STEP                           # 1-based I<=STEP
    high = idx >= _STEP - 1                     # 1-based I>=STEP
    iz1 = np.where(low, z3, z1)
    iz3 = np.where(high, iz1, z3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y1 = np.log(_take_z(fm, iz1))
        y2 = np.log(_take_z(fm, z2))
        y3 = np.log(_take_z(fm, iz3))
        x1, x2, x3 = focus[z1], focus[z2], focus[z3]    # x is never swapped
        c = (((y1 - y2) * (x2 - x3) - (y2 - y3) * (x1 - x2))
             / ((x1 ** 2 - x2 ** 2) * (x2 - x3) - (x2 ** 2 - x3 ** 2) * (x1 - x2)))
        b = ((y2 - y3) - c * (x2 - x3) * (x2 + x3)) / (x2 - x3)
        s2 = -1.0 / (2.0 * c)                   # MATLAB: s = sqrt(-1./(2*c)), used as s.^2
        u = b * s2
        a = y1 - b * x1 - c * x1 ** 2
        big_a = np.exp(a + u ** 2 / (2.0 * s2))
    return u, s2, big_a, fmax


def _windowed_mean(arr: np.ndarray, size: int) -> np.ndarray:
    """imfilter(fspecial('average', size), 'replicate'): a DIRECT separable convolution.

    Never ``uniform_filter`` here: it is a running sum, and the inverted-fit pixels carry
    errors up to 1e300 (or inf), which a running sum smears down the rest of the line as
    rounding residue, and whose taint mask it cannot state exactly. Direct arithmetic
    lets a NaN, an inf or a huge value reach exactly the windows containing it, the
    MATLAB's own behaviour (measured against an exact dilation: identical).
    """
    kernel = np.full(int(size), 1.0 / int(size))
    rows = convolve1d(np.asarray(arr, dtype=np.float64), kernel, axis=0, mode="nearest")
    return convolve1d(rows, kernel, axis=1, mode="nearest")


def lateral_halo_px(nhsize: int) -> int:
    """The pixels a fused pixel depends on laterally: two focus-measure windows, the
    selectivity window (``nhsize // 2`` each) and the 3x3 median (1)."""
    return 3 * (int(nhsize) // 2) + 1


def fuse_stack(planes, nhsize: int = DEFAULT_NHSIZE, alpha: float = DEFAULT_ALPHA,
               sth: float = DEFAULT_STH) -> np.ndarray:
    """Fuse a z stack of same-shape planes into one all-in-focus plane, native dtype."""
    planes = [np.asarray(p) for p in planes]
    if len(planes) < MIN_PLANES:
        raise ValueError(
            f"fstack needs at least {MIN_PLANES} z planes: its 3-point Gaussian "
            f"interpolation samples {_STEP} frames each side of the focus peak, and this "
            f"stack has {len(planes)}. Reduce fewer-plane stacks with 'mip' instead.")
    first = planes[0]
    for plane in planes[1:]:
        if plane.shape != first.shape:
            raise ValueError(f"plane shape {plane.shape} != first plane {first.shape}")
        if plane.dtype != first.dtype:
            raise ValueError(f"plane dtype {plane.dtype} != first plane {first.dtype}")

    dtype = first.dtype
    stack = np.stack(planes).astype(np.float64)         # (P, Y, X)
    # im2double's scaling for integer input; the fusion is a convex combination of the
    # planes, so working scaled and rescaling once at the end is exact.
    scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    if scale != 1.0:
        stack /= scale
    P = stack.shape[0]
    focus = np.arange(P, dtype=np.float64)

    fm = np.empty_like(stack)
    for p in range(P):
        fm[p] = gfocus(stack[p], nhsize)

    u, s2, big_a, fmax = gauss3P(focus, fm)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        err = np.zeros_like(fmax)
        for p in range(P):
            err += np.abs(fm[p] - big_a * np.exp(-((focus[p] - u) ** 2) / (2.0 * s2)))
        fm /= fmax                                       # normalise AFTER err, like the MATLAB
        inv_psnr = _windowed_mean(err / (P * fmax), nhsize)
        big_s = 20.0 * np.log10(1.0 / inv_psnr)
    nan = np.isnan(big_s)
    if nan.any():
        rest = big_s[~nan]                # MATLAB min() skips NaN but keeps -Inf; so does this
        big_s[nan] = rest.min() if rest.size else 0.0

    phi = 0.5 * (1.0 + np.tanh(alpha * (big_s - sth))) / alpha
    phi = median_filter(phi, size=3, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        weights = 0.5 + 0.5 * np.tanh(phi[None] * (fm - 1.0))
        fused = (stack * weights).sum(axis=0) / weights.sum(axis=0)
    bad = ~np.isfinite(fused)
    if bad.any():                          # zero focus measure in every frame: plain mean
        fused[bad] = stack.mean(axis=0)[bad]
    if scale != 1.0:
        fused *= scale
    return cast_like(fused, dtype, copy=False)


def fstack_op(nhsize: int = DEFAULT_NHSIZE, alpha: float = DEFAULT_ALPHA,
              sth: float = DEFAULT_STH) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build THE fstack operator: all z planes of one (FOV, channel, t) -> one fused plane."""
    if not (nhsize >= 1 and float(nhsize) == int(nhsize)):
        raise ValueError(f"nhsize must be a whole window size >= 1 px; got {nhsize!r}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1] (the MATLAB's own bound); got {alpha!r}")
    if not np.isfinite(sth):
        raise ValueError(f"sth must be a finite threshold in dB; got {sth!r}")

    def _fstack(planes: Iterable[np.ndarray]) -> np.ndarray:
        return fuse_stack(list(planes), int(nhsize), float(alpha), float(sth))

    _fstack.__name__ = f"fstack(nhsize={int(nhsize)},alpha={alpha},sth={sth})"
    _fstack.consumes = Z_REDUCER
    _fstack.halo_px = lateral_halo_px(int(nhsize))     # ruling z: an ROI window's halo
    return _fstack


# Advanced on purpose (Julio: the user should only tweak what can't be deduced from
# acquisition filenames; every other knob hides in an 'advanced' slot).
_FSTACK_PARAMS = (
    Param("nhsize", DEFAULT_NHSIZE,
          "Focus measure window size in px. Larger windows smooth the focus map; smaller "
          "ones follow finer structure.", advanced=True),
    Param("alpha", DEFAULT_ALPHA,
          "Selectivity sharpness in (0, 1]. Smaller values blend frames more; 1 commits "
          "harder to the sharpest frame.", advanced=True),
    Param("sth", DEFAULT_STH,
          "Selectivity threshold in dB. Pixels whose focus profile fits a Gaussian worse "
          "than this blend toward the plain mean.", advanced=True),
)

add_operator("fstack", fstack_op, consumes=frozenset({"z"}), params=_FSTACK_PARAMS)
