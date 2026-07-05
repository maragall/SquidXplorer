"""Physical / scalar acquisition metadata from ``acquisition.yaml`` (the single format).

``acquisition.yaml`` is Squid's authoritative metadata: the objective pixel size ALREADY
computed for the objective + camera binning (so no fragile sensor/mag recompute), the
wellplate format, and the z-stack / time-series parameters. It is **required**.

The legacy flat ``acquisition parameters.json`` is intentionally NOT supported: every current
Squid acquisition writes ``acquisition.yaml``, so a JSON fallback has no real input — it would
be dead code carrying a permanent second-format test burden. One format, required, loud on
absence. (If a genuinely pre-yaml dataset ever resurfaces, convert it to ``acquisition.yaml``
up front rather than adding a second read path here.)

``coordinates.csv`` is intentionally NOT read: for one-FOV-per-well (IMA-183) the plate layout
comes from the well ID + ``wellplate_format``; per-FOV stage positions are a stitching/mosaic
concern, deferred to the ticket that needs them.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_acquisition_metadata(root) -> dict:
    """Return scalar acquisition metadata from ``acquisition.yaml``.

    Keys (all from acquisition.yaml; the reader cross-checks n_z/n_t against the filenames /
    timepoint folders and warns on disagreement):
        pixel_size_um    - object-space pixel size (µm), binning-aware
        n_z_declared     - Nz as recorded
        dz_um            - z-step (µm)
        n_t_declared     - Nt as recorded
        wellplate_format - e.g. "24 well plate"

    Raises
    ------
    FileNotFoundError
        If ``acquisition.yaml`` is absent — it is the single supported metadata format.
    """
    root = Path(root)
    path = root / "acquisition.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"acquisition.yaml not found in {root} — it is required. The legacy flat "
            "'acquisition parameters.json' is no longer supported (convert a pre-yaml "
            "dataset to acquisition.yaml up front)."
        )

    rich = yaml.safe_load(path.read_text()) or {}
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
    }
