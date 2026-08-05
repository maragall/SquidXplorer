#!/usr/bin/env python
"""Write a synthetic 5-D Squid acquisition: regions x FOV x z x channel x TIMEPOINT.

WHY THIS EXISTS. Every acquisition on this workstation is ``n_t = 1`` -- the 10x tissue set, the
20x scan and the 1536 plate all have a single timepoint folder. So the T axis is exercised by
nothing except ``tests/conftest.py``'s in-memory ``multi_time_point_dataset``, and the features
that live on that axis cannot be driven by hand at all: the timepoint bar, playback, the ``.mp4``
recorder's T path, and ``fuse_region_pyramid``'s ``t`` argument (which, measured, is never passed
-- the region mosaic renders t=0 whatever the slider says, and nothing on disk could reveal it).

A SCRIPT, not a hand-built folder. The 1536 symlink farm was rebuilt by hand and dangled twice,
the second time at 24576 of 24576 links, because the thing that made it was never written down.
This is written down.

WHAT IT MAKES. Content varies along EVERY axis, so a defect that collapses one is visible rather
than merely plausible:

* **t** moves a bright blob across the frame, so a stuck t=0 shows a blob that never moves.
* **z** sweeps focus (sharpest at the middle plane), so a stuck z shows no focal sweep and a MIP
  is brighter than any single plane.
* **channel** changes the structure itself, so a channel mix-up is not just a brightness change.
* **fov** shifts the field by the stage step with real overlap, so stitching has something to
  register -- the texture is a function of ABSOLUTE stage position, so neighbouring FOVs genuinely
  share content in the seam rather than by coincidence.

Usage::

    python tools/make_5d_fixture.py ~/Downloads/sim_5d_2x2_t3
    python tools/make_5d_fixture.py OUT --regions A1,A2,B1,B2 --fovs 4 --nz 3 --nt 3 --size 256
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile

#: Squid's own naming. The region token may not contain "_": reader.py:160's _STEM_RE is
#: ``^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)$``, so "A1" is legal and
#: "well_A1" is not -- it would parse as region "well" and then fail on the fov field.
_CHANNELS = ["Fluorescence_405_nm_Ex", "Fluorescence_638_nm_Ex"]
_PIXEL_UM = 0.752
_DZ_UM = 1.5
#: Stage step as a FRACTION of the frame, so adjacent FOVs overlap by the remainder. 0.75 leaves
#: 25% overlap, comfortably above find_adjacent_pairs' 15 px floor at any frame size here.
_STEP_FRAC = 0.75


def _frame(size: int, fov: int, z: int, nz: int, c: int, t: int, nt: int) -> np.ndarray:
    """One plane. Every axis changes it, each in a way a human can name on sight."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # Texture keyed on ABSOLUTE stage position, so the overlap between neighbouring FOVs really
    # is the same tissue -- which is what makes registration meaningful here.
    ox = (fov % 2) * size * _STEP_FRAC
    oy = (fov // 2) * size * _STEP_FRAC
    gx, gy = xx + ox, yy + oy
    texture = (
        np.sin(gx / (7.0 + 3.0 * c)) * np.cos(gy / (11.0 - 2.0 * c))
        + 0.5 * np.sin((gx + gy) / 23.0)
    )

    # Focus sweep: sharpest at the middle plane, so a MIP is brighter than any one plane and a
    # reference-plane pick has a real answer instead of a tie.
    mid = (nz - 1) / 2.0
    sharp = 1.0 / (1.0 + 2.0 * abs(z - mid))

    # A blob that MOVES with t. This is the tell for a stuck timepoint.
    frac = t / max(nt - 1, 1)
    bx = size * (0.2 + 0.6 * frac)
    by = size * (0.5 + 0.25 * np.sin(2 * np.pi * frac))
    blob = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * (size / 12.0) ** 2)))

    img = 2000.0 + 12000.0 * sharp * (0.5 + 0.5 * texture) + 20000.0 * blob * sharp
    rng = np.random.default_rng(1000 * t + 100 * fov + 10 * z + c)   # deterministic per plane
    img = img + rng.normal(0.0, 60.0, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def build(out: Path, regions, n_fovs: int, nz: int, nt: int, size: int) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    per_side = int(np.ceil(np.sqrt(n_fovs)))
    step_mm = size * _STEP_FRAC * _PIXEL_UM / 1000.0
    t0 = datetime(2026, 8, 5, 9, 0, 0)

    for t in range(nt):
        tdir = out / str(t)
        tdir.mkdir(exist_ok=True)
        rows = ["region,fov,z_level,x (mm),y (mm),z (um),time"]
        for r_i, region in enumerate(regions):
            # Wells sit far apart so they read as distinct regions, not one mosaic.
            rx, ry = (r_i % 2) * 10.0, (r_i // 2) * 10.0
            for fov in range(n_fovs):
                x_mm = rx + (fov % per_side) * step_mm
                y_mm = ry + (fov // per_side) * step_mm
                for z in range(nz):
                    for c_i, ch in enumerate(_CHANNELS):
                        tifffile.imwrite(
                            tdir / f"{region}_{fov}_{z}_{ch}.tiff",
                            _frame(size, fov, z, nz, c_i, t, nt),
                        )
                    stamp = (t0 + timedelta(minutes=30 * t, seconds=fov + z)).strftime(
                        "%Y-%m-%d_%H-%M-%S.%f")
                    rows.append(f"{region},{fov},{z},{x_mm},{y_mm},{z * _DZ_UM},{stamp}")
        (tdir / "coordinates.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    (out / "acquisition.yaml").write_text(
        "# Synthetic 5-D acquisition written by tools/make_5d_fixture.py\n"
        "objective:\n"
        f"  pixel_size_um: {_PIXEL_UM}\n"
        "  magnification: 20.0\n"
        "  sensor_pixel_size_um: 15.04\n"
        "sample:\n"
        "  wellplate_format: glass slide\n"
        "z_stack:\n"
        f"  nz: {nz}\n"
        f"  delta_z_mm: {_DZ_UM / 1000.0}\n"
        "time_series:\n"
        f"  nt: {nt}\n", encoding="utf-8")

    (out / "acquisition parameters.json").write_text(json.dumps({
        "dx(mm)": step_mm, "Nx": per_side, "dy(mm)": step_mm, "Ny": per_side,
        "dz(um)": _DZ_UM, "Nz": nz, "dt(s)": 1800.0, "Nt": nt,
        "with AF": False, "with reflection AF": False, "with manual focus map": False,
        # NA is the ONLY source of aperture for decon's per-channel optics -- acquisition.yaml
        # carries none -- so a fixture without this cannot be deconvolved.
        "objective": {"magnification": 20.0, "NA": 0.8, "tube_lens_f_mm": 180.0, "name": "20x"},
        "sensor_pixel_size_um": 15.04, "tube_lens_mm": 180,
    }, indent=1), encoding="utf-8")

    modes = "\n".join(
        f'  <mode ID="{i}" Name="{ch}" ExposureTime="20.0" AnalogGain="0" '
        f'IlluminationSource="{11 + i}" IlluminationIntensity="30" CameraSN="" '
        f'ZOffset="0.0" EmissionFilterPosition="1" Selected="1" />'
        for i, ch in enumerate(_CHANNELS))
    (out / "configurations.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<modes>\n{modes}\n</modes>\n', encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--regions", default="A1,A2,B1,B2")
    ap.add_argument("--fovs", type=int, default=4)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--nt", type=int, default=3)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    regions = [r.strip() for r in a.regions.split(",") if r.strip()]
    bad = [r for r in regions if "_" in r]
    if bad:
        ap.error(f"region names may not contain '_' (reader.py:160 _STEM_RE): {bad}")
    out = build(a.out, regions, a.fovs, a.nz, a.nt, a.size)
    n = len(regions) * a.fovs * a.nz * len(_CHANNELS) * a.nt
    print(f"wrote {n} planes to {out}")
    print(f"  {len(regions)} regions x {a.fovs} FOV x {a.nz} z x {len(_CHANNELS)} ch x {a.nt} t")


if __name__ == "__main__":
    main()
