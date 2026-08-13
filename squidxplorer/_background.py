"""Background subtraction as a plane-op and, crucially, as a LAYER.

The raw is recoverable: the source is never written, the background is a first-class
artefact (:func:`estimate_background`), :func:`restore` is the exact inverse where nothing
clipped, and :func:`clipped_fraction` reports where it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from squidxplorer._engine import add_operator
from squidxplorer.projection import cast_like, plane_op

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

# Default ball radius in pixels: larger than any object to keep, smaller than
# illumination-scale structure.
DEFAULT_RADIUS_PX: int = 50


@dataclass(frozen=True)
class BackgroundParams:
    """The layer's parameters — a frozen record, so a layer is fully described by this value.

    ``rolling_ball`` (default, ImageJ's algorithm), ``gaussian`` (fast heavy low-pass) or
    ``sep`` (Julio's bgsub; a central estimator, so ~half of an unsigned plane clips).
    ``radius_px`` must exceed the largest object to keep; ``downsample`` 0 = auto, 1 = off.
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
    """Return the estimated background of ONE plane as float32 — the layer's operand. Never mutates."""
    from scipy.ndimage import gaussian_filter, zoom   # lazy: headless import graph stays light

    params = params or DEFAULT_PARAMS
    if plane.ndim != 2:
        raise ValueError(f"estimate_background takes ONE 2-D plane; got shape {plane.shape}")
    # Re-validate: a dataclass built with object.__setattr__ or unpickled could bypass __post_init__.
    if params.method not in METHODS:
        raise ValueError(f"unknown background method {params.method!r}; available: {list(METHODS)}")

    img = plane.astype(np.float32, copy=True)

    if params.method == "sep":
        # bgsub.core._run_sep is a pure function of one plane; imported lazily, failing loud.
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

    # ImageJ's shrink -> roll -> enlarge: the background is low-frequency by construction.
    small = zoom(img, 1.0 / scale, order=1)
    bg_small = rolling_ball(small, radius=max(1.0, params.radius_px / scale))
    bg = zoom(bg_small.astype(np.float32),
              (img.shape[0] / bg_small.shape[0], img.shape[1] / bg_small.shape[1]), order=1)
    # zoom's output shape is rounded, so trim/pad-by-edge to land exactly on the plane shape.
    if bg.shape != img.shape:
        bg = np.pad(bg[:img.shape[0], :img.shape[1]],
                    ((0, max(0, img.shape[0] - bg.shape[0])),
                     (0, max(0, img.shape[1] - bg.shape[1]))), mode="edge")
    # Clamp to the plane's own range so a resampling overshoot cannot create negatives.
    return np.minimum(bg, float(img.max())).astype(np.float32)


def subtract_background(
    plane: np.ndarray, params: Optional[BackgroundParams] = None
) -> np.ndarray:
    """Subtract the estimated background from ONE plane. Same shape and dtype; input untouched.

    Integer dtypes are clipped to the dtype range, never wrapped.
    """
    background = estimate_background(plane, params)
    return cast_like(plane.astype(np.float32, copy=False) - background, plane.dtype)


def restore(corrected: np.ndarray, background: np.ndarray, dtype=None) -> np.ndarray:
    """Rebuild the raw plane from a corrected plane and its background: the layer's exact inverse where nothing clipped."""
    dtype = corrected.dtype if dtype is None else np.dtype(dtype)
    return cast_like(corrected.astype(np.float32, copy=False) + background, dtype)


def clipped_fraction(plane: np.ndarray, params: Optional[BackgroundParams] = None) -> float:
    """Fraction of pixels whose subtraction would clip at the dtype floor, i.e. where the raw is not recoverable."""
    if not np.issubdtype(plane.dtype, np.integer):
        return 0.0
    background = estimate_background(plane, params)
    residual = np.rint(plane.astype(np.float32, copy=False) - background)
    info = np.iinfo(plane.dtype)
    return float(np.mean((residual < info.min) | (residual > info.max)))


def bgsub_op(
    params: Optional[BackgroundParams] = None,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a parameterised background-subtraction plane-op, ready for ``add_operator``."""
    params = params or DEFAULT_PARAMS

    def _bgsub(p: np.ndarray) -> np.ndarray:
        return subtract_background(p, params)

    _bgsub.__name__ = f"bgsub({params.method},radius_px={params.radius_px})"
    return plane_op(_bgsub)


# The layer key the viewer uses for this operator's OperationStack entry (``bgsub@<tab>``).
LAYER_KEY: str = "bgsub"
LAYER_LABEL: str = "background subtraction"

add_operator(LAYER_KEY, bgsub_op())
