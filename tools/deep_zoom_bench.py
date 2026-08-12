"""Measure the deep-zoom tile path, and render a before/after on real data.

Uses the same placement maths as PlateOverview._paint_tiles, so a placement bug shows up here
rather than as a picture someone has to eyeball in the GUI. A fit-to-plate tile overlapping
every FOV measured 25s to build on a 9-well, 10-deep acquisition — re-run this before changing
the zoom threshold that decides when the overlay engages.

    python tools/deep_zoom_bench.py "D:/path/to/acquisition" --out before_after.png
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from squidxplorer._tiling import select_tiles
from squidxplorer._tilesource import ReaderTileSource, _paste_field, plate_ladder
from squidxplorer.reader import open_reader


def _render(src, ladder, geometry, channel, bbox, level, out_px):
    """Assemble one screen rectangle from a specific rung. Returns (image, n_tiles, seconds)."""
    scale = (bbox[2] - bbox[0]) / out_px
    dst = np.zeros((out_px, out_px), dtype=np.uint16)
    descs = [d for d in select_tiles(bbox, geometry.levels[level].scale_um_per_px, geometry,
                                     channels=(channel,)) if d.level == level]
    t0 = time.perf_counter()
    for d in descs:
        _paste_field(dst, bbox, scale, src.read_tile(d), d.bbox_um)
    return dst, len(descs), time.perf_counter() - t0


def _to8(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a[a > 0], (1, 99)) if (a > 0).any() else (0, 1)
    return np.clip((a.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acquisition", help="a RAW Squid acquisition folder")
    ap.add_argument("--out", default=None, help="write a before/after PNG here (needs Pillow)")
    ap.add_argument("--screen", type=int, default=700, help="px of the rendered rectangle")
    args = ap.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # the coordinates.csv salvage warning is expected
        reader = open_reader(args.acquisition)
    meta = reader.metadata
    channel = meta["channels"][0]["name"]
    ladder = plate_ladder(meta)
    g = ladder.geometry
    src = ReaderTileSource(reader, meta, ladder)
    w = ladder.world_bbox_um

    print(f"ladder: {len(g)} rungs, "
          f"{g.levels[0].scale_um_per_px:.3f}-{g.levels[-1].scale_um_per_px:.1f} um/px, "
          f"tiles/level {[len(lv) for lv in g.levels]}, worst_case={g.worst_case_tiles}")

    print("\n=== fetch cost per viewport (cold cache) ===")
    for label, frac in (("fit-to-plate", 1.0), ("1/4 plate", 0.25), ("one well", 0.06)):
        span = (w[2] - w[0]) * frac
        bbox = (w[0], w[1], w[0] + span, w[1] + span)
        um_per_px = span / args.screen
        lvl = g.pick_level(um_per_px)
        descs = [d for d in select_tiles(bbox, um_per_px, g, channels=(channel,)) if d.level == lvl]
        fresh = ReaderTileSource(reader, meta, ladder)      # cold, so the number means something
        t0 = time.perf_counter()
        for d in descs:
            fresh.read_tile(d)
        dt = time.perf_counter() - t0
        print(f"  {label:>13}: level {lvl}  {len(descs):>3} tiles  {dt:7.2f}s total  "
              f"{dt / max(len(descs), 1) * 1000:7.0f} ms/tile  ({um_per_px:8.3f} um/px)")

    span = (w[2] - w[0]) * 0.06
    roi = (w[0] + (w[2] - w[0]) * 0.02, w[1] + (w[3] - w[1]) * 0.02,
           w[0] + (w[2] - w[0]) * 0.02 + span, w[1] + (w[3] - w[1]) * 0.02 + span)
    best = g.pick_level(span / args.screen)

    print("\n=== before/after on the same rectangle ===")
    coarse, nc, tc = _render(src, ladder, g, channel, roi, len(g) - 1, args.screen)
    fine, nf, tf = _render(src, ladder, g, channel, roi, best, args.screen)
    print(f"  BEFORE (coarsest rung {len(g) - 1}, upscaled): {nc} tile(s), {tc:.2f}s")
    print(f"  AFTER  (ladder picks rung {best})            : {nf} tile(s), {tf:.2f}s")

    if args.out:
        from PIL import Image

        gap = np.full((args.screen, 6), 255, np.uint8)
        Image.fromarray(np.concatenate([_to8(coarse), gap, _to8(fine)], axis=1)).save(args.out)
        print(f"  wrote {args.out} (left = montage-era blur, right = deep zoom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
