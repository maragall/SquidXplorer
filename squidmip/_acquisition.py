"""Scalar acquisition metadata: physical params and per-FOV stage positions.

These come from the sidecar files, not the image filenames:
    acquisition parameters.json  -> Nz, Nt, dz(um), pixel size (sensor / magnification)
    coordinates.csv              -> per-FOV (x, y) stage positions

`coordinates.csv` schema is version-dependent (verified against Cephla-Lab/Squid):
    current : region, fov, z_level, x (mm), y (mm), z (um), time [, z_piezo (um)]
    legacy  : region, x (mm), y (mm), z (mm)          # no fov, no z_level
Both are handled. When there is no `fov` column, FOV is the per-region row index —
identical to the filename fov token, which Squid assigns by enumerating each region's
coordinate list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_acquisition_params(root) -> dict:
    """Read 'acquisition parameters.json'. Missing file/keys yield None (never raises)."""
    path = Path(root) / "acquisition parameters.json"
    params: dict = {}
    if path.exists():
        params = json.loads(path.read_text())

    magnification = (params.get("objective") or {}).get("magnification")
    sensor_px = params.get("sensor_pixel_size_um")
    pixel_size_um = None
    if magnification and sensor_px:
        pixel_size_um = sensor_px / magnification

    return {
        "n_z_declared": params.get("Nz"),
        "n_t_declared": params.get("Nt"),
        "dz_um": params.get("dz(um)"),
        "pixel_size_um": pixel_size_um,
        "magnification": magnification,
        "sensor_pixel_size_um": sensor_px,
    }


def _find_col(df: pd.DataFrame, *candidates: str):
    lower = {c.lower().replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        hit = lower.get(cand.lower().replace(" ", ""))
        if hit is not None:
            return hit
    return None


def load_positions(coords_path) -> dict:
    """Return {(region, fov): (x_mm, y_mm)}. Empty dict if the file/columns are absent.

    Positions are best-effort auxiliary metadata (for the plate-view UI); a missing or
    partial coordinates.csv never blocks ingest.
    """
    path = Path(coords_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    x_col = _find_col(df, "x (mm)", "x(mm)", "x")
    y_col = _find_col(df, "y (mm)", "y(mm)", "y")
    if "region" not in df.columns or x_col is None or y_col is None:
        return {}

    positions: dict = {}
    if "fov" in df.columns:
        # current schema: fov is explicit; first row per (region, fov) wins (z_level 0)
        for _, row in df.iterrows():
            if pd.isna(row["fov"]):
                continue
            key = (str(row["region"]), int(row["fov"]))
            if key not in positions:
                positions[key] = (float(row[x_col]), float(row[y_col]))
    else:
        # legacy schema: one row per FOV; fov = per-region enumeration index
        counters: dict = {}
        for _, row in df.iterrows():
            region = str(row["region"])
            fov = counters.get(region, 0)
            positions[(region, fov)] = (float(row[x_col]), float(row[y_col]))
            counters[region] = fov + 1
    return positions
