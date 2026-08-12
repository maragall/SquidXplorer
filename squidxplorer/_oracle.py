"""IMA-211 stitch acceptance oracle — a stitcher-agnostic quality gate.

IMA-211 originally asked to wrap four external stitchers as plate operators. The stitch
operator itself is IMA-222's; the consumes-axis registry is IMA-226's; FOV geometry is
IMA-187's. What no other ticket owns — and what IMA-222 lacks entirely — is an **acceptance
criterion**. "Each stitcher runs as a plate operator" is a smoke test: it cannot tell a correct
mosaic from a scrambled one.

This module is that criterion. It **grades** a stitcher; it does not contain one. There is no
registration here on purpose — the thing under test supplies the offsets, and duplicating
IMA-222's algorithm inside its own grader would make the gate circular.

The trick that makes this run **today**, with no microscope, no laser-AF fix, and no new
dependency: cut a known-good image into overlapping tiles at *known* positions, hand the tiles
to a stitcher, and check whether it puts them back where they came from.

Fixture construction::

    source image (one real plane, or synthetic)
         │
         │  grid=(ny,nx), overlap_frac  ──►  nominal[i] = (row*step_y, col*step_x)
         │                                    step = tile_size * (1 - overlap_frac)
         │
         │  inject integer offsets       ──►  truth[i] = nominal[i] + offset[i]
         │  (integer, so bit-exactness is well defined; subpixel would need
         │   interpolation and there would be no exact answer to compare against)
         ▼
    tiles[i] = source[ty : ty+h, tx : tx+w]        cut AT the true positions
         │
         ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  a stitcher under test sees ONLY (tiles, nominal, overlap_px)        │
    │  and must return estimated positions                                 │
    └──────────────────────────────────────────────────────────────────────┘
         │
         ▼
    grade(estimated, fixture)  ──►  max/mean placement error in px
    paste(tiles, truth)        ──►  bit-exact to source  (oracle #1)
    paste(tiles, nominal)      ──►  visible seams        (proves the metric discriminates)

Two gates, matching the plan's acceptance oracle:

  #1 placement — max |estimated - truth| <= 1 px, and pasting at the recovered positions
     reproduces the source **exactly** in the covered region.
  #2 seam continuity — ``seam_ratio`` <= 1.5. The mean absolute step across a seam line,
     divided by the mean absolute step between adjacent columns/rows *inside* a tile. A
     perfectly aligned mosaic has no discontinuity at the seam, so the ratio sits near 1;
     a misplaced tile spikes it. Normalising by the interior step is what makes the
     threshold hold across images of wildly different contrast.

  #5 negative case — a blank or uniform-noise overlap strip carries no registration signal.
     ``overlap_texture`` reports it so a stitcher can be *required* to fall back to nominal
     placement there rather than chase a noise peak and throw a tile tens of pixels off.

Nothing here imports scipy or scikit-image: numpy only, so the oracle costs the shipped
package nothing. See ``docs/ima-211-eng-review.md``.
"""

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
    """A cut-up image plus the ground truth a stitcher is supposed to recover.

    Attributes
    ----------
    tiles:
        The cut tiles, in row-major grid order. Native dtype of the source, never cast.
    nominal:
        ``(N, 2)`` int array of ``(y, x)`` positions the stage *intended* — i.e. what a
        stitcher gets for free from coordinates, before any registration.
    truth:
        ``(N, 2)`` int array of the positions the tiles were actually cut from. This is the
        answer; a stitcher never sees it.
    grid:
        ``(ny, nx)``.
    overlap_px:
        ``(oy, ox)`` nominal overlap in pixels between adjacent tiles.
    source:
        The image the tiles were cut from, cropped to the covered region — the bit-exact
        target for ``paste(tiles, truth)``.
    """

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
    """Cut *source* into an overlapping tile grid at known, deliberately perturbed positions.

    The perturbation is what makes this a test: a stitcher that ignores the pixels and simply
    trusts the nominal grid scores exactly ``max_offset_px`` of error, so it cannot pass by
    doing nothing.

    Parameters
    ----------
    source:
        2-D image. Must be large enough for the requested grid; the requirement is reported
        precisely if it is not.
    grid:
        ``(ny, nx)`` tile counts.
    overlap_frac:
        Fractional overlap between neighbours (0.09 ≈ the ~9% measured on the 20x scan).
        Must be in [0, 1).
    max_offset_px:
        Bound on the injected per-tile offset, in pixels. Offsets are integers so that
        "bit-exact" has an exact meaning. Must stay within half the overlap or the mosaic
        tears; the default (8) is consistent with the default 9% overlap. Raise
        ``overlap_frac`` to inject larger errors.
    seed:
        Seed for the offset draw, so a failing run is reproducible.
    offsets:
        Explicit ``(N, 2)`` integer offsets, overriding the random draw. Pass zeros to build
        a perfectly-aligned fixture.

    Raises
    ------
    ValueError
        Non-2D source, out-of-range ``overlap_frac``, a grid the source is too small for, or
        an ``offsets`` array of the wrong shape — each named loud, never silently clamped.
    """
    src = np.asarray(source)
    if src.ndim != 2:
        raise ValueError(f"source must be 2-D (Y, X), got shape {src.shape}")
    if not 0.0 <= overlap_frac < 1.0:
        raise ValueError(f"overlap_frac must be in [0, 1), got {overlap_frac}")
    ny, nx = int(grid[0]), int(grid[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"grid must be >= 1 in both axes, got {grid}")

    sh, sw = src.shape
    # Tile size is derived so the grid exactly spans the source at zero offset, leaving a
    # margin of max_offset_px on every side for the perturbation to move into.
    margin = 2 * max_offset_px
    th = (sh - margin) // (1 + (ny - 1) * (1 - overlap_frac)) if ny > 1 else sh - margin
    tw = (sw - margin) // (1 + (nx - 1) * (1 - overlap_frac)) if nx > 1 else sw - margin
    th, tw = int(th), int(tw)
    if th < 8 or tw < 8:
        raise ValueError(
            f"source {src.shape} is too small for grid={grid} overlap={overlap_frac} "
            f"max_offset_px={max_offset_px}: derived tile size ({th}, {tw}) < 8 px"
        )

    # The step is a ROUNDED fraction of the tile, so (ny-1)*step + tile can land a pixel or
    # two past the source even though the unrounded arithmetic fit. Shrink the tile until the
    # grid provably fits inside the margin rather than discovering it as a crash below.
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

    # Resolve offsets FIRST: an explicitly-supplied set is validated on its own merits, and
    # the tear check below then bounds on what was actually injected rather than on the
    # unused max_offset_px ceiling (passing offsets=zeros can never tear anything).
    if offsets is None:
        rng = np.random.default_rng(seed)
        off = rng.integers(-max_offset_px, max_offset_px + 1, size=(n, 2), dtype=np.int64)
    else:
        off = _as_positions("offsets", offsets, n).astype(np.int64)
    bound = int(np.abs(off).max()) if off.size else 0

    # A tile shifted by -d and its neighbour shifted by +d pull apart by 2d. If that exceeds
    # the overlap the mosaic TEARS: a strip of the plane is covered by no tile at all. Such a
    # fixture silently under-tests (the grader would be scoring holes, not alignment), so
    # refuse it rather than produce it. This is a real property of stitching, not a limitation
    # of the harness: you cannot perturb tiles by more than half the overlap and still have a
    # covering.
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

    # Shift everything into a non-negative frame; nominal and truth move together so the
    # error a stitcher must recover is exactly `off`.
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
    """Composite *tiles* at *positions* by hard overwrite, in order.

    This is deliberately the naive placement — no feather, no blending. It is both the
    baseline any real stitcher must beat and the fallback behaviour oracle #5 requires when
    an overlap strip carries no signal. Later tiles overwrite earlier ones in the overlap,
    which is what makes ``paste(tiles, truth)`` bit-exact against the source: every pixel,
    whoever writes it, came from the same place in the original image.

    dtype is preserved — the canvas takes the tiles' dtype, never a float upcast.
    """
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
    """Boolean mask of canvas pixels that at least one tile actually wrote.

    Offsetting tiles leaves the canvas corners uncovered — tile 0 nudged down-and-right
    exposes the top-left. Those pixels are zeros, not image data, and comparing or measuring
    across them would score the padding instead of the mosaic. Every metric here masks by
    this rather than quietly treating background as signal.
    """
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

    Returns ``mean|step across seams| / mean|step between adjacent interior lines|``. A
    perfectly aligned mosaic is continuous at the seam, so the ratio sits near 1.0; a
    misplaced tile creates a visible edge and the ratio climbs. Normalising by the interior
    step is what lets one threshold (1.5) hold for both a dim and a bright acquisition.

    Returns ``inf`` when the interior step is zero (a perfectly flat image carries no signal
    to normalise against) — reported honestly rather than papered over with a fudge factor.

    **Know the limitation:** this metric needs *spatial structure*. On uncorrelated noise a
    misplaced tile is statistically indistinguishable from a correct one, so the ratio stays
    near 1.0 and the gate silently passes a bad mosaic. Real microscopy fields are structured,
    so it discriminates in practice — but it is a companion to the placement check in
    :func:`grade_positions`, never a replacement for it. Use :func:`overlap_texture` to
    confirm there is signal to measure before trusting a seam result.
    """
    comp = np.asarray(composite).astype(np.float64)
    pos = np.asarray(positions, dtype=np.int64)
    th, tw = int(tile_shape[0]), int(tile_shape[1])

    if mask is None:
        mask = np.ones(comp.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    # A seam step is only meaningful where BOTH sides carry image data. Uncovered canvas is
    # zero-padding; including it would report the padding edge as a giant seam and the metric
    # would fail every mosaic, aligned or not.
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

    # Interior baseline: adjacent-column and adjacent-row steps taken well inside the first
    # tile, away from any seam and from the uncovered corners.
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
    """Per-tile texture in the nominal overlap strips — oracle #5's precondition.

    A stitcher can only register where there is structure to register on. This returns, per
    tile, the standard deviation within its overlap strips. A value at or near zero means a
    blank seam: correlation there is chasing noise, and the stitcher **must** fall back to
    nominal placement instead of shifting the tile tens of pixels.

    Use it to assert the negative case rather than hoping it never happens.
    """
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
    """Grade a stitcher's recovered positions against the fixture's ground truth.

    Applies the plan's oracle #1 (placement within ``max_error_px``, and a bit-exact
    reconstruction when the recovery is perfect) and #2 (``seam_ratio`` within
    ``max_seam_ratio``). Every failed gate is named in ``Grade.reasons`` — a bare False is
    useless when this runs in CI.
    """
    est = _as_positions("estimated", estimated, len(fixture.tiles)).astype(np.int64)
    err = np.abs(est - fixture.truth).astype(np.float64)
    max_err = float(err.max()) if err.size else 0.0
    mean_err = float(err.mean()) if err.size else 0.0

    composite = paste(fixture.tiles, est)
    mask = coverage(fixture.tiles, est)
    th, tw = fixture.tiles[0].shape
    ratio = seam_ratio(composite, est, (th, tw), mask=mask)

    # Bit-exactness is judged against the source the tiles were cut from, over the region a
    # tile actually wrote — the uncovered corners are padding, not a reconstruction failure.
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
