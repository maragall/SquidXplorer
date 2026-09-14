#!/usr/bin/env python
"""Deconvolve a WHOLE acquisition: full z per solve, lateral tiling across solves.

Usage: python scripts/decon_whole.py SOURCE [--iterations N] [--dry-run]

Each solve is one whole (Z, y, x) window through the existing deconvolve_stack
(never tiled internally, z always whole); windows tile the frame exactly and each
is expanded by the channel's declared halo, so interiors match a whole-field solve
(within ~1 count at <= 2 iterations; at 3+ the Biggs-Andrews lambda is global per
solved volume, so tile interiors differ slightly from an un-runnable whole solve).
Output is a sibling acquisition in the source's own OME-TIFF shape.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

# This checkout's package, not an editable install pointing at another worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from squidxplorer import _decon, _decon_gpu
from squidxplorer.reader import open_reader

log = logging.getLogger("decon_whole")

# Per-solve working-set target, bytes: leaves headroom for the GUI beside this run.
TARGET_WORKING_SET = 2_000_000_000

SIDECARS = ("acquisition parameters.json", "acquisition.yaml",
            "acquisition_channels.yaml", "acquisition.log", "coordinates.csv")


def tile_edges(size: int, n: int) -> list[int]:
    """n+1 cut points tiling [0, size) exactly into n near-equal windows."""
    return [i * size // n for i in range(n + 1)]


def plan_grid(n_z: int, height: int, width: int, halo: int, psf_shape,
              target: float) -> tuple[int, int]:
    """(grid_n, working_set_bytes): the coarsest square grid whose worst window fits target."""
    for n in range(1, max(height, width) + 1):
        tile = max(-(-height // n), -(-width // n))
        shape = (n_z, tile + 2 * halo, tile + 2 * halo)
        device = _decon_gpu.select_device(shape, gpu=True, psf_shape=psf_shape)
        ws = _decon_gpu.working_set_bytes(shape, device, psf_shape)
        if ws <= target:
            return n, ws
    raise MemoryError(
        f"even one column of the frame at full z ({n_z} planes) does not fit the "
        f"{target / 1e9:.1f} GB per-solve target. The z axis is never split: free memory "
        "or deconvolve a z subset instead.")


def _release_gpu() -> None:
    """MPS's caching allocator holds freed blocks across solves; without this release the
    run drains the machine (measured 6.8 -> 1.8 GB inside one channel) and select_device
    falls to a CPU path that refuses."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:                        # noqa: BLE001 - no torch = nothing cached
        pass


def solve_tiled(stack: np.ndarray, optics, halo: int, grid: int,
                iterations: int, backends: Counter) -> np.ndarray:
    """Tile stack (Z, Y, X) laterally, solve each window whole in z, paste trimmed cores.

    *backends* tallies, per window, the device ``select_device`` answers for that window's
    exact expanded shape — the same call ``deconvolve_stack`` makes, so the tally IS the
    backend each window ran on (edge windows lose halo at the frame edge, so their shapes,
    and therefore the smoothness guard's answer, can differ from interior ones).
    """
    nz, height, width = stack.shape
    out = np.empty_like(stack)
    psf_shape = _decon.make_psf(optics).shape
    rows, cols = tile_edges(height, grid), tile_edges(width, grid)
    for i in range(grid):
        for j in range(grid):
            r0, r1, c0, c1 = rows[i], rows[i + 1], cols[j], cols[j + 1]
            er0, er1 = max(0, r0 - halo), min(height, r1 + halo)
            ec0, ec1 = max(0, c0 - halo), min(width, c1 + halo)
            window_shape = (nz, er1 - er0, ec1 - ec0)
            device = _decon_gpu.select_device(window_shape, gpu=True, psf_shape=psf_shape)
            backends[device or "cpu"] += 1
            solved = _decon.deconvolve_stack(
                stack[:, er0:er1, ec0:ec1], optics, iterations, project=False)
            out[:, r0:r1, c0:c1] = solved[:, r0 - er0:r1 - er0, c0 - ec0:c1 - ec0]
            _release_gpu()
    return out


def channel_plans(source: Path, meta: dict, overrides: dict | None = None) -> dict:
    """{channel: (optics, halo) or None for copy-through} with z bound to the stack depth.

    *overrides* maps OpticsParams field names (na, dxy_um, ni) to explicit values that
    replace the acquisition record's, for a record known to be wrong or incomplete.
    """
    plans = {}
    for ch in meta["channels"]:
        name = ch["name"]
        try:
            o = _decon.optics_for_channel(source, name)
        except _decon.NoEmissionLine as exc:
            log.info("%s: no emission wavelength, copied unchanged, not deconvolved (%s)",
                     name, exc)
            plans[name] = None
            continue
        changes = dict(overrides or {})
        if o.nz != meta["n_z"]:
            changes["nz"] = meta["n_z"]
        if changes:
            o = dataclasses.replace(o, **changes)
        plans[name] = (o, _decon.lateral_halo_px(o))
    return plans


def link_sidecars(source: Path, out_dir: Path) -> None:
    """Hardlink (copy on failure) the acquisition sidecars; nothing needs editing, Nz is unchanged."""
    def link(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    for name in SIDECARS:
        link(source / name, out_dir / name)
    link(source / "0" / "coordinates.csv", out_dir / "0" / "coordinates.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="Squid acquisition folder to deconvolve")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--suffix", default="",
                        help="output-name suffix, e.g. _i2 -> decon46_i2_<source>")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the tile plan and estimates, solve nothing")
    parser.add_argument("--na", type=float, default=None,
                        help="override the recorded numerical aperture")
    parser.add_argument("--dxy-um", type=float, default=None,
                        help="override the recorded lateral pixel size (um)")
    parser.add_argument("--ni", type=float, default=None,
                        help="override the immersion refractive index")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    source = args.source.resolve()
    out_dir = source.parent / f"decon46{args.suffix}_{source.name}"
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out_dir}")

    reader = open_reader(source)
    meta = reader.metadata
    n_z, (height, width) = meta["n_z"], meta["frame_shape"]
    names = [c["name"] for c in meta["channels"]]
    overrides = {k: v for k, v in
                 (("na", args.na), ("dxy_um", args.dxy_um), ("ni", args.ni))
                 if v is not None}
    if overrides:
        log.info("optics overrides: %s", ", ".join(f"{k}={v}" for k, v in overrides.items()))
    plans = channel_plans(source, meta, overrides)
    solved_plans = {n: p for n, p in plans.items() if p is not None}
    if not solved_plans:
        raise SystemExit("every channel copies through; nothing to deconvolve")

    # One grid for all channels, planned from the worst (halo, PSF) among them.
    max_halo = max(h for _o, h in solved_plans.values())
    worst_psf = max((_decon.make_psf(o).shape for o, _h in solved_plans.values()),
                    key=np.prod)
    device = _decon_gpu.select_device((n_z, height, width), gpu=True, psf_shape=worst_psf)
    # The solve runs beside two held uint16 stacks (input and assembled output); the
    # per-solve target leaves room for them inside the measured budget.
    held = 2 * n_z * height * width * np.dtype(meta["dtype"]).itemsize
    target = min(TARGET_WORKING_SET, _decon_gpu.budget_bytes(device) - held)
    grid, worst_ws = plan_grid(n_z, height, width, max_halo, worst_psf, target)
    tile = max(-(-height // grid), -(-width // grid))
    window_shape = (n_z, tile + 2 * max_halo, tile + 2 * max_halo)

    n_fields = sum(len(f) for f in meta["fovs_per_region"].values())
    log.info("plan: %d field(s) x %d channel(s), frame %dx%d, z %d (always whole)",
             n_fields, len(names), height, width, n_z)
    log.info("tile plan: %dx%d grid, %d window(s) of ~%d px per channel; halo per "
             "channel: %s", grid, grid, grid * grid, tile,
             ", ".join(f"{n}={h} px" for n, (_o, h) in solved_plans.items()))
    log.info("per-solve working set: ~%.2f GB for a %s window (measured %dx multiple), "
             "against a %.2f GB budget (%.0f%% of available), target %.1f GB",
             worst_ws / 1e9, window_shape,
             _decon_gpu.working_set_multiple(
                 _decon_gpu.select_device(window_shape, gpu=True, psf_shape=worst_psf)),
             _decon_gpu.budget_bytes(device) / 1e9,
             _decon_gpu.MEMORY_FRACTION * 100, target / 1e9)
    log.info(_decon_gpu.describe(window_shape, gpu=True, psf_shape=worst_psf))
    log.info("caveat: Biggs-Andrews lambda is global per solved volume, so at 3+ "
             "iterations tile interiors differ slightly from an un-runnable whole "
             "solve; within ~1 count at 2 or fewer. This run: %d iteration(s).",
             args.iterations)
    if args.dry_run:
        log.info("dry run: nothing solved, nothing written. Output would be %s", out_dir)
        return 0

    (out_dir / "ome_tiff").mkdir(parents=True)
    link_sidecars(source, out_dir)
    t0 = time.monotonic()
    for region, fovs in meta["fovs_per_region"].items():
        for fov in fovs:
            # A memmap-backed OME-TIFF: the (Z, C, Y, X) field never lives whole in RAM.
            field = tifffile.memmap(
                out_dir / "ome_tiff" / f"{region}_{fov}.ome.tiff",
                shape=(n_z, len(names), height, width), dtype=meta["dtype"],
                metadata={
                    "axes": "ZCYX",
                    "Channel": {"Name": names},
                    "PhysicalSizeX": meta["pixel_size_um"], "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": meta["pixel_size_um"], "PhysicalSizeYUnit": "µm",
                    "PhysicalSizeZ": meta["dz_um"], "PhysicalSizeZUnit": "µm",
                })
            for c, name in enumerate(names):
                stack = np.stack([reader.read(region, fov, name, z)
                                  for z in range(n_z)])
                if plans[name] is None:
                    field[:, c] = stack
                    continue
                optics, halo = plans[name]
                t1 = time.monotonic()
                backends: Counter = Counter()
                field[:, c] = solve_tiled(stack, optics, halo, grid, args.iterations,
                                          backends)
                field.flush()
                log.info("%s/%s %s: %d window(s) solved in %.1f s (backends: %s)",
                         region, fov, name, grid * grid, time.monotonic() - t1,
                         ", ".join(f"{n} {d}" for d, n in sorted(backends.items())))
            del field
    log.info("done: %d field(s) in %.1f s, output %s",
             n_fields, time.monotonic() - t0, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
