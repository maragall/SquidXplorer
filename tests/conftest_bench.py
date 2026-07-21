"""Synthetic Squid acquisitions with EXACTLY known geometry, for benchmark tests.

Tiles are cut from one large textured canvas at known pixel offsets, so ground truth
is not estimated -- it is constructed. That is what lets the tests assert the seam
metric recovers a deliberately injected misalignment, rather than merely asserting it
returns a number.

    canvas
    +---------------------------+
    | tile(0,0) | tile(0,1) |   |   step < tile width  =>  real overlap
    +-----------+-----------+   |   tile(i,j) origin = (i*step_y, j*step_x)
    | tile(1,0) | tile(1,1) |   |
    +---------------------------+
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

DEFAULT_PIXEL_UM = 0.3728571351101784  # 20x objective, matches the real 20x_scan
DEFAULT_CHANNEL = "Fluorescence_405_nm_Ex"


def make_canvas(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Broadband texture. Random noise correlates crisply and has no periodic
    ambiguity, which is what we want when testing the correlator itself."""
    rng = np.random.default_rng(seed)
    return rng.integers(200, 4000, size=(height, width), dtype=np.uint16)


def write_acquisition(
    root: Path,
    *,
    grid: tuple[int, int] = (2, 3),
    tile: tuple[int, int] = (128, 128),
    step: tuple[int, int] = (96, 96),
    region: str = "C5",
    channel: str = DEFAULT_CHANNEL,
    channels: list[str] | None = None,
    z_levels: int = 1,
    pixel_um: float = DEFAULT_PIXEL_UM,
    seed: int = 0,
    with_fov_column: bool = False,
    blank: bool = False,
) -> dict:
    """Write a complete Squid-layout acquisition. Returns the ground truth."""
    ny, nx = grid
    th, tw = tile
    sy, sx = step
    canvas = (
        np.zeros((th + (ny - 1) * sy, tw + (nx - 1) * sx), dtype=np.uint16)
        if blank
        else make_canvas(th + (ny - 1) * sy, tw + (nx - 1) * sx, seed=seed)
    )

    img_dir = root / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    chans = channels or [channel]

    truth: dict[int, tuple[float, float]] = {}
    fov = 0
    for i in range(ny):
        for j in range(nx):
            y0, x0 = i * sy, j * sx
            truth[fov] = (float(y0), float(x0))
            crop = canvas[y0 : y0 + th, x0 : x0 + tw]
            for z in range(z_levels):
                for ch in chans:
                    tifffile.imwrite(img_dir / f"{region}_{fov}_{z}_{ch}.tiff", crop)
            fov += 1

    # Stage coordinates in mm, derived from the same ground truth.
    lines = ["region,fov,x (mm),y (mm),z (mm)"] if with_fov_column else ["region,x (mm),y (mm),z (mm)"]
    for f in sorted(truth):
        y_px, x_px = truth[f]
        x_mm = x_px * pixel_um / 1000.0
        y_mm = y_px * pixel_um / 1000.0
        if with_fov_column:
            lines.append(f"{region},{f},{x_mm:.9f},{y_mm:.9f},")
        else:
            lines.append(f"{region},{x_mm:.9f},{y_mm:.9f},")
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")

    (root / "acquisition parameters.json").write_text(
        json.dumps(
            {
                "Nx": nx,
                "Ny": ny,
                "Nz": z_levels,
                "sensor_pixel_size_um": pixel_um * 20.0,
                "objective": {"magnification": 20.0, "NA": 0.8, "name": "20x"},
            }
        )
    )

    return {
        "root": root,
        "region": region,
        "channel": chans[0],
        "channels": chans,
        "positions_px": truth,
        "tile": tile,
        "step": step,
        "grid": grid,
        "pixel_um": pixel_um,
    }


def write_broken_symlink_farm(root: Path, n: int = 3, region: str = "C5") -> Path:
    """An acquisition whose TIFFs are all dangling links -- the sim_1536wp failure."""
    img = root / "0"
    img.mkdir(parents=True, exist_ok=True)
    missing = root / "gone" / "source.tiff"
    for i in range(n):
        (img / f"{region}_{i}_0_Fluorescence_405_nm_Ex.tiff").symlink_to(missing)
    (root / "coordinates.csv").write_text("region,x (mm),y (mm),z (mm)\nC5,0,0,\n")
    return root
