"""Stitch acceptance oracle: cut a known image into perturbed tiles, grade a stitcher's
recovered positions against the ground truth. numpy only; contains no stitcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "Fixture",
    "Grade",
    "cut_fixture",
    "paste",
    "grade_positions",
    "seam_ratio",
    "overlap_texture",
]


@dataclass(frozen=True)
class Fixture:
    """A cut-up image plus the ground truth a stitcher is supposed to recover."""

    tiles: list
    nominal: np.ndarray
    truth: np.ndarray
    grid: tuple
    overlap_px: tuple
    source: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        """The injected error, ``truth - nominal``. What a stitcher must discover."""
        return self.truth - self.nominal


@dataclass(frozen=True)
class Grade:
    """Result of grading one stitcher run against a :class:`Fixture`."""

    max_error_px: float
    mean_error_px: float
    bit_exact: bool
    seam_ratio: float
    passed: bool
    reasons: tuple

    def __str__(self) -> str:  # pragma: no cover - display only
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}  max={self.max_error_px:.2f}px mean={self.mean_error_px:.2f}px "
            f"seam_ratio={self.seam_ratio:.2f} bit_exact={self.bit_exact}"
            + ("" if self.passed else "  reasons: " + "; ".join(self.reasons))
        )


def _as_positions(name: str, arr, n: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.shape != (n, 2):
        raise ValueError(f"{name} must have shape ({n}, 2), got {a.shape}")
    return a


def cut_fixture(
    source: np.ndarray,
    grid: tuple = (2, 2),
    overlap_frac: float = 0.09,
    max_offset_px: int = 8,
    seed: int = 0,
    offsets: Optional[np.ndarray] = None,
) -> Fixture:
    """Cut *source* into an overlapping tile grid at known, deliberately perturbed positions."""
    src = np.asarray(source)
    if src.ndim != 2:
        raise ValueError(f"source must be 2-D (Y, X), got shape {src.shape}")
    if not 0.0 <= overlap_frac < 1.0:
        raise ValueError(f"overlap_frac must be in [0, 1), got {overlap_frac}")
    ny, nx = int(grid[0]), int(grid[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"grid must be >= 1 in both axes, got {grid}")

    sh, sw = src.shape
    # Tile size derived so the grid spans the source, with a margin for the perturbation.
    margin = 2 * max_offset_px
    th = (sh - margin) // (1 + (ny - 1) * (1 - overlap_frac)) if ny > 1 else sh - margin
    tw = (sw - margin) // (1 + (nx - 1) * (1 - overlap_frac)) if nx > 1 else sw - margin
    th, tw = int(th), int(tw)
    if th < 8 or tw < 8:
        raise ValueError(
            f"source {src.shape} is too small for grid={grid} overlap={overlap_frac} "
            f"max_offset_px={max_offset_px}: derived tile size ({th}, {tw}) < 8 px"
        )

    # The rounded step can push the grid past the source; shrink the tile until it fits.
    for _ in range(64):
        step_y = int(round(th * (1.0 - overlap_frac)))
        if (ny - 1) * step_y + th <= sh - margin:
            break
        th -= 1
    for _ in range(64):
        step_x = int(round(tw * (1.0 - overlap_frac)))
        if (nx - 1) * step_x + tw <= sw - margin:
            break
        tw -= 1
    step_y = int(round(th * (1.0 - overlap_frac)))
    step_x = int(round(tw * (1.0 - overlap_frac)))
    if th < 8 or tw < 8:
        raise ValueError(
            f"source {src.shape} is too small for grid={grid} overlap={overlap_frac} "
            f"max_offset_px={max_offset_px}: derived tile size ({th}, {tw}) < 8 px"
        )
    oy, ox = th - step_y, tw - step_x

    n = ny * nx
    base = np.array(
        [[r * step_y, c * step_x] for r in range(ny) for c in range(nx)], dtype=np.int64
    )

    if offsets is None:
        rng = np.random.default_rng(seed)
        off = rng.integers(-max_offset_px, max_offset_px + 1, size=(n, 2), dtype=np.int64)
    else:
        off = _as_positions("offsets", offsets, n).astype(np.int64)
    bound = int(np.abs(off).max()) if off.size else 0

    # Offsets beyond half the overlap tear the mosaic (uncovered strips): refuse.
    if ny > 1 and 2 * bound > oy:
        raise ValueError(
            f"offset {bound}px tears the mosaic vertically: 2*offset ({2 * bound}) exceeds "
            f"the {oy}px row overlap. Raise overlap_frac (>= {2 * bound / th:.3f}) or lower "
            f"max_offset_px (<= {oy // 2})."
        )
    if nx > 1 and 2 * bound > ox:
        raise ValueError(
            f"offset {bound}px tears the mosaic horizontally: 2*offset ({2 * bound}) exceeds "
            f"the {ox}px column overlap. Raise overlap_frac (>= {2 * bound / tw:.3f}) or "
            f"lower max_offset_px (<= {ox // 2})."
        )

    # Shift into a non-negative frame; nominal and truth move together.
    nominal = base + max_offset_px
    truth = nominal + off
    if truth.min() < 0:
        raise ValueError("injected offsets pushed a tile off the source; raise the margin")

    need_y = int(truth[:, 0].max()) + th
    need_x = int(truth[:, 1].max()) + tw
    if need_y > sh or need_x > sw:
        raise ValueError(
            f"source {src.shape} too small: grid={grid} at overlap={overlap_frac} with "
            f"max_offset_px={max_offset_px} needs at least ({need_y}, {need_x})"
        )

    tiles = [
        np.array(src[ty : ty + th, tx : tx + tw], copy=True) for ty, tx in truth
    ]

    cov_y = int(truth[:, 0].max()) + th
    cov_x = int(truth[:, 1].max()) + tw
    covered = np.array(src[: cov_y, : cov_x], copy=True)

    return Fixture(
        tiles=tiles,
        nominal=nominal,
        truth=truth,
        grid=(ny, nx),
        overlap_px=(oy, ox),
        source=covered,
    )


def paste(tiles: Sequence[np.ndarray], positions, canvas_shape=None) -> np.ndarray:
    """Composite *tiles* at *positions* by hard overwrite, in order; dtype preserved."""
    if len(tiles) == 0:
        raise ValueError("no tiles to paste")
    pos = _as_positions("positions", positions, len(tiles)).astype(np.int64)
    if pos.min() < 0:
        raise ValueError("positions must be non-negative")

    dtype = tiles[0].dtype
    for i, t in enumerate(tiles):
        if t.ndim != 2:
            raise ValueError(f"tile {i} must be 2-D, got shape {t.shape}")
        if t.dtype != dtype:
            raise ValueError(f"tile {i} dtype {t.dtype} != tile 0 dtype {dtype}")

    if canvas_shape is None:
        h = int(max(p[0] + t.shape[0] for p, t in zip(pos, tiles)))
        w = int(max(p[1] + t.shape[1] for p, t in zip(pos, tiles)))
        canvas_shape = (h, w)

    out = np.zeros(canvas_shape, dtype=dtype)
    for (y, x), t in zip(pos, tiles):
        th, tw = t.shape
        y2, x2 = min(y + th, canvas_shape[0]), min(x + tw, canvas_shape[1])
        if y2 > y and x2 > x:
            out[y:y2, x:x2] = t[: y2 - y, : x2 - x]
    return out


def coverage(tiles: Sequence[np.ndarray], positions, canvas_shape=None) -> np.ndarray:
    """Boolean mask of canvas pixels that at least one tile actually wrote."""
    if len(tiles) == 0:
        raise ValueError("no tiles to cover")
    pos = _as_positions("positions", positions, len(tiles)).astype(np.int64)
    if canvas_shape is None:
        h = int(max(p[0] + t.shape[0] for p, t in zip(pos, tiles)))
        w = int(max(p[1] + t.shape[1] for p, t in zip(pos, tiles)))
        canvas_shape = (h, w)

    mask = np.zeros(canvas_shape, dtype=bool)
    for (y, x), t in zip(pos, tiles):
        th, tw = t.shape
        y2, x2 = min(y + th, canvas_shape[0]), min(x + tw, canvas_shape[1])
        if y2 > y and x2 > x:
            mask[y:y2, x:x2] = True
    return mask


def seam_ratio(composite: np.ndarray, positions, tile_shape, mask=None) -> float:
    """Discontinuity across seam lines, normalised by the image's own interior step.

    Near 1.0 for an aligned mosaic; ``inf`` when the interior step is zero. Needs spatial
    structure — a companion to the placement check, never a replacement for it.
    """
    comp = np.asarray(composite).astype(np.float64)
    pos = np.asarray(positions, dtype=np.int64)
    th, tw = int(tile_shape[0]), int(tile_shape[1])

    if mask is None:
        mask = np.ones(comp.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    # Only count a seam step where BOTH sides carry image data, not zero-padding.
    seam_steps = []
    for x in sorted({int(p) for p in pos[:, 1] if p > 0}):
        if 0 < x < comp.shape[1]:
            both = mask[:, x] & mask[:, x - 1]
            if both.any():
                seam_steps.append(np.abs(comp[both, x] - comp[both, x - 1]).mean())
    for y in sorted({int(p) for p in pos[:, 0] if p > 0}):
        if 0 < y < comp.shape[0]:
            both = mask[y, :] & mask[y - 1, :]
            if both.any():
                seam_steps.append(np.abs(comp[y, both] - comp[y - 1, both]).mean())

    if not seam_steps:
        return 1.0

    # Interior baseline, taken well inside the first tile, away from seams and corners.
    y0 = int(pos[:, 0].min()) + th // 4
    x0 = int(pos[:, 1].min()) + tw // 4
    y1, x1 = min(y0 + th // 2, comp.shape[0]), min(x0 + tw // 2, comp.shape[1])
    interior = comp[y0:y1, x0:x1]
    if interior.shape[0] < 2 or interior.shape[1] < 2:
        interior = comp
    base = float(
        np.mean(
            [
                np.abs(np.diff(interior, axis=0)).mean(),
                np.abs(np.diff(interior, axis=1)).mean(),
            ]
        )
    )
    if base == 0.0:
        return float("inf")
    return float(np.mean(seam_steps) / base)


def overlap_texture(fixture: Fixture) -> np.ndarray:
    """Per-tile standard deviation within the nominal overlap strips; near zero means a
    blank seam a stitcher must fall back to nominal placement on."""
    oy, ox = fixture.overlap_px
    out = []
    for t in fixture.tiles:
        strips = []
        if ox > 0 and t.shape[1] > ox:
            strips.append(t[:, :ox].ravel())
            strips.append(t[:, -ox:].ravel())
        if oy > 0 and t.shape[0] > oy:
            strips.append(t[:oy, :].ravel())
            strips.append(t[-oy:, :].ravel())
        out.append(float(np.concatenate(strips).std()) if strips else 0.0)
    return np.asarray(out, dtype=np.float64)


def grade_positions(
    estimated,
    fixture: Fixture,
    max_error_px: float = 1.0,
    max_seam_ratio: float = 1.5,
) -> Grade:
    """Grade a stitcher's recovered positions against the fixture's ground truth."""
    est = _as_positions("estimated", estimated, len(fixture.tiles)).astype(np.int64)
    err = np.abs(est - fixture.truth).astype(np.float64)
    max_err = float(err.max()) if err.size else 0.0
    mean_err = float(err.mean()) if err.size else 0.0

    composite = paste(fixture.tiles, est)
    mask = coverage(fixture.tiles, est)
    th, tw = fixture.tiles[0].shape
    ratio = seam_ratio(composite, est, (th, tw), mask=mask)

    # Bit-exactness is judged over the covered region only; uncovered corners are padding.
    ref = fixture.source
    if composite.shape == ref.shape:
        bit_exact = bool(np.array_equal(composite[mask], ref[mask]))
    else:
        h = min(composite.shape[0], ref.shape[0])
        w = min(composite.shape[1], ref.shape[1])
        m = mask[:h, :w]
        bit_exact = bool(np.array_equal(composite[:h, :w][m], ref[:h, :w][m]))

    reasons = []
    if max_err > max_error_px:
        reasons.append(f"placement error {max_err:.2f}px > {max_error_px}px")
    if not np.isfinite(ratio) or ratio > max_seam_ratio:
        reasons.append(f"seam ratio {ratio:.2f} > {max_seam_ratio}")

    return Grade(
        max_error_px=max_err,
        mean_error_px=mean_err,
        bit_exact=bit_exact,
        seam_ratio=ratio,
        passed=not reasons,
        reasons=tuple(reasons),
    )
