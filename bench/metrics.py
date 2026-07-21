"""Seam-residual quality metric -- one identical ruler for every stitcher.

The question this answers: *given the tile positions a stitcher solved for, how well do
neighbouring tiles actually line up?* Lower is better; a perfect solver leaves 0 px.

It is measured PRE-fusion, on the input tiles, because that is the only place it can be
measured at all::

    tile_i --+                                    fused mosaic
             +-- overlap -> strip_i, strip_j        +------------+
    tile_j --+   two INDEPENDENT views              | blended px | -> one view
                 -> correlatable                    +------------+  -> nothing to
                                                                       correlate against

Algorithm, per adjacent pair::

    1. place both tiles using the SOLVER'S positions
    2. cut the overlap rectangle out of each -> strip_i, strip_j (same shape)
    3. split the strips into blocks along their long axis
    4. per block: phase-correlate -> residual shift (dy, dx)
    5. keep the block only if normalized cross-correlation after shifting >= ncc_min
       (rejects blank/textureless blocks, which would otherwise report a fake 0)
    6. residual = |(dy, dx)| in full-resolution pixels

Implemented on numpy's FFT alone. SquidMIP depends on neither scipy nor scikit-image,
and importing ``tilefusion`` to borrow its ``_block_shifts`` would drag numba, GPU and
basicpy through its ``__init__`` -- the exact coupling ``pyproject.toml`` forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A block narrower than this has too little support for a stable correlation peak.
MIN_BLOCK_PX = 24
# Blocks whose post-shift NCC falls below this are discarded as untextured.
DEFAULT_NCC_MIN = 0.3
# Below this std a block is flat; correlating it produces noise, not a measurement.
MIN_STD = 1e-6
# A seam needs at least this many surviving blocks to report a residual.
MIN_BLOCKS = 2


@dataclass(frozen=True)
class SeamResult:
    """One measured seam between two tiles."""

    fov_i: int
    fov_j: int
    n_blocks_locked: int
    n_blocks_total: int
    shifts_px: np.ndarray  # magnitude per locked block

    @property
    def median_px(self) -> float:
        return float(np.median(self.shifts_px)) if self.shifts_px.size else float("nan")


def phase_correlate(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation of ``b`` relative to ``a``, via phase correlation.

    Returns ``(dy, dx)`` such that ``b`` is displaced from ``a`` by that amount --
    i.e. feature at row ``r`` in ``a`` appears near row ``r + dy`` in ``b``.

    Phase normalization (dividing the cross-power spectrum by its magnitude) makes the
    result depend on alignment alone, not on brightness or contrast -- so it survives
    the illumination differences between neighbouring FOVs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    # Mean-subtract so the DC term doesn't dominate the peak.
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12  # guard: a flat block gives an all-zero spectrum
    corr = np.fft.ifft2(cross / mag).real

    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = _subpixel(corr, peak, axis=0)
    dx = _subpixel(corr, peak, axis=1)

    # ifft2 output is periodic: an index past the halfway point is a negative shift.
    h, w = corr.shape
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)


def _subpixel(corr: np.ndarray, peak: tuple[int, int], axis: int) -> float:
    """Parabolic refinement of the correlation peak along one axis.

    Fits a parabola through the peak and its two neighbours; the vertex is the
    sub-pixel location. Wraps at the array edge because the correlation is periodic.
    """
    n = corr.shape[axis]
    i = peak[axis]
    lo = list(peak)
    hi = list(peak)
    lo[axis] = (i - 1) % n
    hi[axis] = (i + 1) % n
    c0 = corr[tuple(lo)]
    c1 = corr[peak]
    c2 = corr[tuple(hi)]
    denom = c0 - 2.0 * c1 + c2
    if abs(denom) < 1e-12:
        return float(i)
    return float(i + 0.5 * (c0 - c2) / denom)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two same-shaped blocks; NaN-safe."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < MIN_STD or b.std() < MIN_STD:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _integer_shift(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift by the nearest whole pixel using edge padding.

    Whole-pixel is deliberate: the NCC here is a *gate* on whether a block has enough
    texture to trust, not the measurement itself, so interpolation would add cost
    without changing any decision.
    """
    return np.roll(np.roll(arr, int(round(dy)), axis=0), int(round(dx)), axis=1)


def overlap_strips(
    tile_i: np.ndarray,
    tile_j: np.ndarray,
    offset_ij: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cut the shared region out of both tiles, given ``j``'s offset from ``i``.

    ``offset_ij`` is ``(dy, dx)`` in pixels, from the stitcher's solved positions.
    Returns ``(strip_i, strip_j)`` of identical shape, or ``None`` if they do not
    overlap enough to measure.

    Geometry, with ``i`` at the origin::

        i occupies rows [0, H),        cols [0, W)
        j occupies rows [dy, dy + H),  cols [dx, dx + W)
        overlap = the intersection, expressed in each tile's own local coordinates
    """
    h, w = tile_i.shape[:2]
    hj, wj = tile_j.shape[:2]
    dy, dx = int(round(offset_ij[0])), int(round(offset_ij[1]))

    r0, r1 = max(0, dy), min(h, dy + hj)
    c0, c1 = max(0, dx), min(w, dx + wj)
    if r1 - r0 < MIN_BLOCK_PX or c1 - c0 < MIN_BLOCK_PX:
        return None

    strip_i = tile_i[r0:r1, c0:c1]
    strip_j = tile_j[r0 - dy : r1 - dy, c0 - dx : c1 - dx]
    if strip_i.shape != strip_j.shape:  # pragma: no cover - defensive
        return None
    return strip_i, strip_j


def block_shifts(
    strip_i: np.ndarray,
    strip_j: np.ndarray,
    n_blocks: int = 8,
    ncc_min: float = DEFAULT_NCC_MIN,
) -> np.ndarray:
    """Per-block residual shift magnitudes along the strips' long axis.

    Blocks that fail the NCC gate are dropped rather than counted as zero -- a blank
    block correlates to nothing, and scoring it 0 px would flatter the stitcher.
    """
    h, w = strip_i.shape[:2]
    along_x = w >= h
    length = w if along_x else h
    block = length // max(1, n_blocks)
    if block < MIN_BLOCK_PX:
        n_blocks = max(1, length // MIN_BLOCK_PX)
        block = length // n_blocks
    if block < MIN_BLOCK_PX:
        return np.empty(0)

    out: list[float] = []
    for k in range(n_blocks):
        s = slice(k * block, (k + 1) * block)
        a = strip_i[:, s] if along_x else strip_i[s, :]
        b = strip_j[:, s] if along_x else strip_j[s, :]
        if a.size < MIN_BLOCK_PX or a.std() < MIN_STD or b.std() < MIN_STD:
            continue
        dy, dx = phase_correlate(a, b)
        if not np.isfinite(dy) or not np.isfinite(dx):
            continue
        if _ncc(a, _integer_shift(b, dy, dx)) < ncc_min:
            continue
        out.append(float(np.hypot(dy, dx)))
    return np.asarray(out, dtype=np.float64)


def adjacent_pairs(
    positions_px: dict[int, tuple[float, float]],
    frame_shape: tuple[int, int],
    min_overlap_fraction: float = 0.02,
) -> list[tuple[int, int]]:
    """FOV pairs whose frames overlap enough to carry a measurable seam."""
    h, w = frame_shape
    fovs = sorted(positions_px)
    pairs: list[tuple[int, int]] = []
    for a_idx, i in enumerate(fovs):
        yi, xi = positions_px[i]
        for j in fovs[a_idx + 1 :]:
            yj, xj = positions_px[j]
            oy = h - abs(yj - yi)
            ox = w - abs(xj - xi)
            if oy <= 0 or ox <= 0:
                continue
            if (oy * ox) / float(h * w) >= min_overlap_fraction:
                pairs.append((i, j))
    return pairs


def seam_residual(
    read_tile,
    positions_px: dict[int, tuple[float, float]],
    pairs: list[tuple[int, int]] | None = None,
    frame_shape: tuple[int, int] | None = None,
    n_blocks: int = 8,
    ncc_min: float = DEFAULT_NCC_MIN,
) -> dict[str, float | int]:
    """Aggregate seam residual over every adjacent pair.

    ``read_tile(fov) -> 2-D array`` is a callable so the caller controls caching and
    channel choice; ``positions_px`` are the STITCHER'S solved positions, which is what
    makes this a measurement of that stitcher rather than of the stage metadata.

    Returns median / mean / p90 in full-resolution pixels, plus the counts needed to
    tell "aligned well" from "we could not measure it".
    """
    if pairs is None:
        if frame_shape is None:
            raise ValueError("pass pairs or frame_shape")
        pairs = adjacent_pairs(positions_px, frame_shape)

    all_shifts: list[float] = []
    seams_measured = 0
    for i, j in pairs:
        if i not in positions_px or j not in positions_px:
            continue
        yi, xi = positions_px[i]
        yj, xj = positions_px[j]
        strips = overlap_strips(read_tile(i), read_tile(j), (yj - yi, xj - xi))
        if strips is None:
            continue
        shifts = block_shifts(*strips, n_blocks=n_blocks, ncc_min=ncc_min)
        if shifts.size < MIN_BLOCKS:
            continue
        seams_measured += 1
        all_shifts.extend(shifts.tolist())

    if not all_shifts:
        return {
            "resid_median_px": float("nan"),
            "resid_mean_px": float("nan"),
            "resid_p90_px": float("nan"),
            "n_seams_measured": 0,
            "n_blocks_locked": 0,
            "n_pairs_candidate": len(pairs),
        }

    arr = np.asarray(all_shifts, dtype=np.float64)
    return {
        "resid_median_px": float(np.median(arr)),
        "resid_mean_px": float(np.mean(arr)),
        "resid_p90_px": float(np.percentile(arr, 90)),
        "n_seams_measured": seams_measured,
        "n_blocks_locked": int(arr.size),
        "n_pairs_candidate": len(pairs),
    }
