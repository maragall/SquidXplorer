"""Background subtraction as a PLANE-OP and, crucially, as a LAYER (IMA-224).

One ``add_projector`` call, zero engine edits: ``consumes=frozenset()`` via
:func:`squidmip.plane_op`, so ``project_well``'s existing loop hands this ONE plane at a time
and z survives at full depth (IMA-210).

WHY THIS IS A LAYER AND NOT A DESTRUCTIVE EDIT
----------------------------------------------
Julio's constraint, verbatim: *"Each transform is a layer, something like CellProfiler does
this."* Background subtraction is the operator where that matters most, because it is the one
that throws photons away. Four separate mechanisms keep the raw recoverable, and each is
pinned by a test in ``tests/test_background.py``:

1. **The source is never written.** The operator is a pure function of a plane the reader
   handed us; ``estimate_background``/``subtract_background`` copy before they touch anything,
   so the caller's buffer — and therefore the raw TIFF behind it — is byte-identical after a
   run. (``test_the_reader_is_read_only_so_the_source_tiffs_survive_a_run``.)
2. **The background is a first-class artefact, not a side effect.**
   :func:`estimate_background` is public and returns the operand itself. The transform is
   ``corrected = raw - background`` with ``background`` in hand, which makes it *invertible*:
   :func:`restore` gives ``raw`` back EXACTLY (``np.array_equal``, not ``allclose``) wherever
   the result did not clip.
3. **Where it is NOT invertible, it says so.** On an unsigned dtype, pixels whose background
   estimate exceeds the raw value clip at 0 and that information is genuinely gone.
   :func:`subtract_background` can report the clipped fraction (:func:`clipped_fraction`)
   rather than presenting a lossy transform as a lossless one.
4. **The UI composes it as a stack entry.** The result is a new layer in
   :class:`squidmip._layers.OperationStack` (key ``bgsub@<tab>``), so switching back to the raw
   is a toggle on the layer, not an inverse transform and not a re-read.

That is the CellProfiler model — a module reads a named image and *provides* a new one, leaving
the input image in the image set untouched — expressed in the shape this codebase already has.

Prior art surveyed before writing
---------------------------------
* **scikit-image** ``restoration.rolling_ball`` — INSTALLED (0.26); this is Sternberg's 1983
  rolling-ball algorithm, i.e. the same algorithm as ImageJ's *Process > Subtract Background*,
  which is what everyone in the lab means by "background subtraction". USED as the default
  method. Not reimplemented.
* **ImageJ's own implementation** — the reference for the *speed* trick, not the algorithm:
  ImageJ shrinks the image before rolling the ball and enlarges the resulting background back.
  The background is by definition low-frequency, so this costs nothing in accuracy and turns a
  minutes-long call on a 2000px plane into a sub-second one. TAKEN (see ``downsample``);
  measured, not assumed — the estimate is checked against a planted background to 15%.
* **CellProfiler** ``CorrectIlluminationCalculate`` / ``...Apply`` — TAKEN: the two-step split
  (compute the correction as its own image, then apply it) is exactly why
  ``estimate_background`` is public, and the "subtract vs divide" distinction is why this
  operator is *additive* only; the multiplicative case is IMA-225's flat-field, a different
  operator with a different physical meaning.
* **BaSiC** — the right tool for the multiplicative shading field, and it is already reused in
  IMA-225. NOT used here: a per-plane additive haze is not what BaSiC estimates.

IMA-247: Julio's ``bgsub`` (sep) is wired in as a METHOD, and is NOT the default
-------------------------------------------------------------------------------
``https://github.com/maragall/background_subtraction`` (the ``bgsub`` package, now cloned to
``/Users/julioamaragall/CEPHLA/projects/background_subtraction``) estimates the background with
**sep** — the Source Extractor sky estimator from astronomy — over a box grid. It is his code,
it is 6.5x faster than rolling_ball on a real 2084x2084 plane (0.04 s vs 0.26 s), and
:func:`estimate_background` now calls it directly for ``method="sep"``.

It is deliberately **not** the default, and the reason is measured, not aesthetic. sep's
background is a *central* estimator: the sigma-clipped mean of each box, splined. By
construction roughly half of a plane's pixels sit BELOW it, so on an unsigned acquisition dtype
half the frame subtracts to a negative number and clips at zero. Measured on real 10x uint16
data (``manual0_0``, mid-plane):

    ============  =====================  ==================
    method        clipped pixel fraction  min residual
    ============  =====================  ==================
    rolling_ball  0.0384                 -181 counts
    sep           **0.5068**             -298 counts
    ============  =====================  ==================

Clipping is the one place this operator destroys information (see point 3 below), so switching
the default to sep would take the layer from ~4% lossy to ~51% lossy — it would quietly break
the "the raw is recoverable" guarantee over half of every image. His own CLI has the same
property, because ``bgsub.core._process_frame_worker`` clips the result back to the input dtype
before writing; that is fine for his tool, whose output is an analysis product, and not fine
for a LAYER whose whole contract is that the raw survives.

So: sep is available, it is his implementation verbatim (``bgsub.core._run_sep``), and
:func:`clipped_fraction` reports what it costs. What sep is genuinely better at is a plane
whose background is *above* the noise floor everywhere and where negative excursions are
meaningful — which is photometry, not this MIP pipeline's default.

Note what is reused and what is NOT: ``bgsub.core.BackgroundSubtractor`` is a file-level
orchestrator that walks an acquisition and WRITES new TIFFs into an output directory. That
class is not used here at all, because writing files is exactly what a layer must not do. Only
the pure array-level estimator is called, which is what keeps property 1 below true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from squidmip._engine import add_projector
from squidmip.projection import plane_op

# The methods this operator knows, in one greppable place (the error message quotes it).
METHODS: tuple[str, ...] = ("rolling_ball", "gaussian", "sep")

_SEP_MISSING = (
    "background method 'sep' needs Julio's bgsub package, which is not importable.\n"
    "  repo:    https://github.com/maragall/background_subtraction\n"
    "           (cloned to /Users/julioamaragall/CEPHLA/projects/background_subtraction)\n"
    "  install: pip install --no-deps -e "
    "/Users/julioamaragall/CEPHLA/projects/background_subtraction\n"
    "           pip install sep\n"
    "There is deliberately NO fallback to rolling_ball: the two estimators disagree by design "
    "(sep is a central estimator, rolling_ball a lower envelope) and silently swapping them "
    "would change how much of the image clips, from ~4% to ~51% (IMA-247)."
)

# Default ball radius in pixels. At the 10x working point (0.752 um/px) 50 px is ~38 um — a few
# cell diameters, i.e. comfortably larger than any object we want to KEEP and comfortably
# smaller than the illumination-scale structure we want to remove. A radius smaller than the
# objects eats the sample; that failure is pinned by test_foreground_puncta_survive_subtraction.
DEFAULT_RADIUS_PX: int = 50


@dataclass(frozen=True)
class BackgroundParams:
    """The layer's parameters — a frozen record, so a layer is fully described by this value.

    method:
        ``"rolling_ball"`` (default) is Sternberg's algorithm via scikit-image: it follows
        structure the sample does not have and is the method the lab already reads as
        "background subtraction" (ImageJ). ``"gaussian"`` is a heavy low-pass, ~50x faster and
        adequate when the haze is genuinely smooth, but it is pulled up by bright objects
        (they leak into their own background estimate). ``"sep"`` is Julio's ``bgsub``
        (Source-Extractor box estimator): the fastest of the three and his own code, but a
        CENTRAL estimator, so about half the pixels of an unsigned plane clip at zero — see the
        module docstring's table before selecting it.
    radius_px:
        Ball radius / low-pass scale in pixels — and, for ``"sep"``, the box size ``bw=bh``.
        MUST exceed the largest object to be kept.
    downsample:
        Estimate the background on a ``1/downsample`` image and enlarge it back — ImageJ's own
        trick, and what makes rolling_ball usable on a 2000px plane. ``0`` (default) picks a
        factor from *radius_px* so the ball is ~8 px on the shrunken image. ``1`` disables it.
    """
    method: str = "rolling_ball"
    radius_px: int = DEFAULT_RADIUS_PX
    downsample: int = 0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"unknown background method {self.method!r}; available: {list(METHODS)}"
            )
        if self.radius_px < 1:
            raise ValueError(f"radius_px must be >= 1, got {self.radius_px}")
        if self.downsample < 0:
            raise ValueError(f"downsample must be >= 0 (0 = auto, 1 = off), got {self.downsample}")

    def scale(self) -> int:
        """The concrete downsample factor, resolving the ``0`` = auto case."""
        if self.downsample:
            return self.downsample
        return max(1, int(self.radius_px // 8))


DEFAULT_PARAMS = BackgroundParams()


def estimate_background(plane: np.ndarray, params: Optional[BackgroundParams] = None) -> np.ndarray:
    """Return the estimated background of ONE plane as float32 — **the layer's operand**.

    Public on purpose (see the module docstring, point 2): with the background in hand the
    subtraction is invertible via :func:`restore`, which is what makes this a layer rather than
    a destructive edit. Never mutates *plane*.
    """
    from scipy.ndimage import gaussian_filter, zoom   # lazy: headless import graph stays light

    params = params or DEFAULT_PARAMS
    if plane.ndim != 2:
        raise ValueError(f"estimate_background takes ONE 2-D plane; got shape {plane.shape}")
    # Re-validate: a dataclass built with object.__setattr__ or unpickled could bypass __post_init__.
    if params.method not in METHODS:
        raise ValueError(f"unknown background method {params.method!r}; available: {list(METHODS)}")

    img = plane.astype(np.float32, copy=True)

    if params.method == "sep":
        # Julio's implementation, called at the ARRAY level: bgsub.core._run_sep is a pure
        # function of one plane and returns (foreground, background), so the background comes
        # back in hand and nothing is written to disk. Imported lazily and failing loud, the
        # same shape _flatfield.py uses for tilefusion.
        try:
            from bgsub.core import _run_sep
        except ImportError as exc:
            raise ImportError(_SEP_MISSING) from exc
        _, bg = _run_sep(img, params.radius_px)
        return np.asarray(bg, dtype=np.float32)

    if params.method == "gaussian":
        # A heavy low-pass. sigma = radius/2 puts the -3 dB point at about the ball scale.
        return gaussian_filter(img, params.radius_px / 2.0, mode="reflect")

    from skimage.restoration import rolling_ball

    scale = params.scale()
    if scale <= 1:
        return rolling_ball(img, radius=params.radius_px).astype(np.float32)

    # ImageJ's shrink -> roll -> enlarge. The background is low-frequency by construction, so
    # the resampling costs accuracy that is below the estimator's own error (tested).
    small = zoom(img, 1.0 / scale, order=1)
    bg_small = rolling_ball(small, radius=max(1.0, params.radius_px / scale))
    bg = zoom(bg_small.astype(np.float32),
              (img.shape[0] / bg_small.shape[0], img.shape[1] / bg_small.shape[1]), order=1)
    # zoom's output shape is rounded, so trim/pad-by-edge to land exactly on the plane shape.
    if bg.shape != img.shape:
        bg = np.pad(bg[:img.shape[0], :img.shape[1]],
                    ((0, max(0, img.shape[0] - bg.shape[0])),
                     (0, max(0, img.shape[1] - bg.shape[1]))), mode="edge")
    # The background can never exceed the smoothed signal it came from by more than rounding;
    # clamping to the plane's own range keeps a resampling overshoot from creating negatives.
    return np.minimum(bg, float(img.max())).astype(np.float32)


def subtract_background(
    plane: np.ndarray, params: Optional[BackgroundParams] = None
) -> np.ndarray:
    """Subtract the estimated background from ONE plane. Same shape and dtype; input untouched.

    Integer dtypes are **clipped** to the dtype range, never wrapped — an unsigned wrap would
    turn the dimmest pixels of the frame into the brightest ones. Clipping at 0 is the only
    place this transform loses information; :func:`clipped_fraction` measures how much.
    """
    background = estimate_background(plane, params)
    return _cast_like(plane.astype(np.float32, copy=False) - background, plane.dtype)


def restore(corrected: np.ndarray, background: np.ndarray, dtype=None) -> np.ndarray:
    """Rebuild the raw plane from a corrected plane and its background: the layer's INVERSE.

    ``restore(subtract_background(raw, p), estimate_background(raw, p)) == raw`` exactly, for
    every pixel that did not clip (``np.array_equal``, no tolerance). This is the mechanical
    statement of "the raw is recoverable"; :func:`clipped_fraction` names the exception.
    """
    dtype = corrected.dtype if dtype is None else np.dtype(dtype)
    return _cast_like(corrected.astype(np.float32, copy=False) + background, dtype)


def clipped_fraction(plane: np.ndarray, params: Optional[BackgroundParams] = None) -> float:
    """Fraction of pixels whose subtraction would clip at the dtype floor — i.e. where the raw
    is NOT exactly recoverable. Reported rather than hidden: a lossy step must say it is lossy."""
    if not np.issubdtype(plane.dtype, np.integer):
        return 0.0
    background = estimate_background(plane, params)
    residual = np.rint(plane.astype(np.float32, copy=False) - background)
    info = np.iinfo(plane.dtype)
    return float(np.mean((residual < info.min) | (residual > info.max)))


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast to the acquisition dtype, ROUNDING and clipping integers rather than truncating
    and wrapping them.

    Both halves are load-bearing:

    * **round, not truncate.** ``astype`` truncates toward zero, which biases every pixel of
      every plane down by half a count — a systematic dimming applied to the whole dataset.
      Rounding is also what makes the layer invertible: with ``c = round(raw - bg)`` the
      residual error is < 0.5 count, so ``round(c + bg) == raw`` exactly (:func:`restore`).
      Truncation loses that, and the raw stops being recoverable by one count everywhere.
    * **clip, not wrap.** An unsigned wrap turns the dimmest pixels of the frame into the
      brightest ones.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    return values.astype(dtype, copy=False)


def bgsub_op(
    params: Optional[BackgroundParams] = None,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a parameterised background-subtraction **plane-op**, ready for ``add_projector``::

        add_projector("bgsub_tight", bgsub_op(BackgroundParams(radius_px=20)))

    The returned callable carries ``consumes = frozenset()``, so z survives at full depth.

    A second registry entry is no longer the only way to change ``radius_px``: an entry can declare
    its own ``params`` and be run at a different value (``_engine.Param``, ``Operator.bind``).
    ``bgsub`` has not been migrated -- doing so is a one-line change to its registration below and
    is deliberately left for whoever needs it, rather than done speculatively.
    """
    params = params or DEFAULT_PARAMS

    def _bgsub(p: np.ndarray) -> np.ndarray:
        return subtract_background(p, params)

    _bgsub.__name__ = f"bgsub({params.method},radius_px={params.radius_px})"
    return plane_op(_bgsub)


# The layer key the viewer uses for this operator's OperationStack entry (``bgsub@<tab>``), so
# the UI and the registry cannot drift apart on the spelling.
LAYER_KEY: str = "bgsub"
LAYER_LABEL: str = "background subtraction"

# The whole registration. No engine edit — the IMA-210 seam working as designed.
add_projector(LAYER_KEY, bgsub_op())
