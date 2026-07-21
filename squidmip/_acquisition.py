"""Physical / scalar acquisition metadata from ``acquisition.yaml`` (the single format).

``acquisition.yaml`` is Squid's authoritative metadata: the objective pixel size ALREADY
computed for the objective + camera binning (so no fragile sensor/mag recompute), the
wellplate format, and the z-stack / time-series parameters. It is **required**.

The legacy flat ``acquisition parameters.json`` is intentionally NOT supported: every current
Squid acquisition writes ``acquisition.yaml``, so a JSON fallback has no real input — it would
be dead code carrying a permanent second-format test burden. One format, required, loud on
absence. (If a genuinely pre-yaml dataset ever resurfaces, convert it to ``acquisition.yaml``
up front rather than adding a second read path here.)

``coordinates.csv`` is NOT read here — the plate layout comes from the well ID +
``wellplate_format`` (IMA-183), and this module stays scalar-only. Per-FOV stage positions
are parsed by :mod:`squidmip._coordinates` (IMA-215) and surface as
``metadata["fov_positions_um"]``; that table is per-FOV rather than scalar, so it does not
fit this module's flat return shape.
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

    def _section(key):
        v = rich.get(key)
        return v if isinstance(v, dict) else {}   # a scalar/None section -> empty (never .get on a float)

    objective = _section("objective")
    z_stack = _section("z_stack")
    time_series = _section("time_series")
    sample = _section("sample")
    delta_z_mm = z_stack.get("delta_z_mm")
    return {
        "pixel_size_um": objective.get("pixel_size_um"),  # authoritative, binning-aware
        "n_z_declared": z_stack.get("nz"),
        "dz_um": delta_z_mm * 1000 if delta_z_mm is not None else None,
        "n_t_declared": time_series.get("nt"),
        "wellplate_format": sample.get("wellplate_format"),
    }
