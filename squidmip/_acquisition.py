"""Physical / scalar acquisition metadata.

Primary source is ``acquisition.yaml`` — Squid's authoritative rich metadata: the objective
pixel size ALREADY computed for the objective + camera binning (so no fragile sensor/mag
recompute), the wellplate format, and the z-stack / time-series parameters. Datasets that
predate it fall back to the flat legacy ``acquisition parameters.json``, where the FOV pixel
size is computed from sensor size, magnification AND the tube-lens ratio it carries (see
``_fov_pixel_size``) — best-effort, since the legacy sidecar has no binning field. If neither
sidecar exists, scalars are None (the MIP still runs; physical scale is simply absent).

``coordinates.csv`` is intentionally NOT read: for one-FOV-per-well (IMA-183) the plate layout
comes from the well ID + ``wellplate_format``; per-FOV stage positions are a stitching/mosaic
concern, deferred to the ticket that needs them.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _load_yaml(path: Path):
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or {}


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fov_pixel_size(sensor_px, magnification, tube_lens_used=None, tube_lens_ref=None):
    """Object-space (FOV) pixel size from the flat JSON.

    FOV pixel size = sensor_pixel_size / effective_magnification. The objective's stated
    magnification assumes its reference tube lens (objective.tube_lens_f_mm); a different
    tube lens (tube_lens_mm) scales it: effective_mag = magnification * used / reference.
    Applying that correction is the difference between sensor pixel size and image pixel size.

    Best-effort only: the legacy JSON has no camera-binning field, so a binned acquisition
    (binning != 1) would still be off by the binning factor. acquisition.yaml's stored
    objective.pixel_size_um already accounts for binning + tube lens, so it is preferred.
    Returns None when sensor size or magnification is missing.
    """
    if not (sensor_px and magnification):
        return None
    effective_mag = magnification
    if tube_lens_used and tube_lens_ref:
        effective_mag = magnification * (tube_lens_used / tube_lens_ref)
    return sensor_px / effective_mag


def load_acquisition_metadata(root) -> dict:
    """Return scalar acquisition metadata from the best available sidecar.

    Keys (any may be None if the sidecar lacks them):
        pixel_size_um    - object-space pixel size (µm)
        n_z_declared     - Nz as recorded (cross-checked against filenames by the reader)
        dz_um            - z-step (µm)
        n_t_declared     - Nt as recorded (cross-checked against timepoint folders)
        wellplate_format - e.g. "24 well plate" (plate-layout hint for the viewer)
        source           - which file the values came from
    """
    root = Path(root)

    rich = _load_yaml(root / "acquisition.yaml")
    if rich:
        objective = rich.get("objective") or {}
        z_stack = rich.get("z_stack") or {}
        time_series = rich.get("time_series") or {}
        sample = rich.get("sample") or {}
        delta_z_mm = z_stack.get("delta_z_mm")
        return {
            "pixel_size_um": objective.get("pixel_size_um"),  # authoritative, binning-aware
            "n_z_declared": z_stack.get("nz"),
            "dz_um": delta_z_mm * 1000 if delta_z_mm is not None else None,
            "n_t_declared": time_series.get("nt"),
            "wellplate_format": sample.get("wellplate_format"),
            "source": "acquisition.yaml",
        }

    # Legacy fallback: flat JSON. pixel size must be recomputed (no stored value); this
    # ignores camera binning and any non-design tube lens, so it is best-effort only.
    flat = _load_json(root / "acquisition parameters.json")
    if flat:
        objective = flat.get("objective") or {}
        pixel_size_um = _fov_pixel_size(
            flat.get("sensor_pixel_size_um"),
            objective.get("magnification"),
            tube_lens_used=flat.get("tube_lens_mm"),
            tube_lens_ref=objective.get("tube_lens_f_mm"),
        )
        return {
            "pixel_size_um": pixel_size_um,
            "n_z_declared": flat.get("Nz"),
            "dz_um": flat.get("dz(um)"),
            "n_t_declared": flat.get("Nt"),
            "wellplate_format": None,
            "source": "acquisition parameters.json",
        }

    return {
        "pixel_size_um": None,
        "n_z_declared": None,
        "dz_um": None,
        "n_t_declared": None,
        "wellplate_format": None,
        "source": None,
    }
