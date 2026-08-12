"""Physical / scalar acquisition metadata from ``acquisition.yaml`` (the single format).

The legacy flat ``acquisition parameters.json`` is not supported as a metadata source; per-FOV
stage positions live in ``reader.load_fov_positions``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def load_acquisition_metadata(root) -> dict:
    """Return scalar acquisition metadata from ``acquisition.yaml``; raises when it is absent."""
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


def load_objective_na(root) -> Optional[float]:
    """Objective NA from ``acquisition parameters.json`` (the only place it is written), or None."""
    path = Path(root) / "acquisition parameters.json"
    try:
        params = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    objective = params.get("objective") if isinstance(params, dict) else None
    na = objective.get("NA") if isinstance(objective, dict) else None
    try:
        na = float(na)
    except (TypeError, ValueError):
        return None
    return na if na > 0 else None


class Channel(BaseModel):
    """One channel of the acquisition, keyed on its canonical (filename) name."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """Canonical filename form, e.g. ``Fluorescence_488_nm_Ex``. The key ``read()`` accepts."""

    display_name: str
    """Human label for the UI. Falls back to ``name`` when the channel YAML has none."""

    display_color: str
    """Hex colour, e.g. ``#1FFF00``. Never a placeholder — see :func:`resolve_channels`."""

    exposure_time_ms: Optional[float] = None
    """Camera exposure (ms) when the channel YAML records one."""

    excitation_nm: Optional[float] = None
    """Excitation wavelength (nm); ``None`` for a broadband channel, never a substitute."""

    # Mapping shim: `c["name"]` is written at many call sites; keep it working.
    def __getitem__(self, key: str):
        if key not in _CHANNEL_FIELDSET:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key) if key in _CHANNEL_FIELDSET else default

    def __contains__(self, key: object) -> bool:
        return key in _CHANNEL_FIELDSET

    def keys(self):
        return _CHANNEL_KEYS


class Acquisition(BaseModel):
    """The validated acquisition model — what ``reader.metadata`` returns.

    Also a Mapping, so the ~96 subscript call sites keep working while attribute access lands
    incrementally; genuinely-optional fields have loud ``require_*`` accessors.
    """

    # arbitrary_types_allowed: `dtype` is a real numpy dtype and must stay one.
    # extra="forbid": a typo'd key must be refused here, not stored and never read.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    regions: list[str]
    """Well / region ids, plate-ordered."""

    fovs_per_region: dict[str, list[int]]
    """``{region: [fov, ...]}``. Must cover every region — see the validator."""

    fov_positions_um: dict[tuple[str, int], tuple[float, float]]
    """``{(region, fov): (x_um, y_um)}`` — stage MICROMETRES; ``{}`` when unusable."""

    channels: list[Channel]
    """Acquisition channels, in C-axis order."""

    n_z: int = Field(ge=1)
    """Distinct z levels. Filename/page-derived, cross-checked against the declared Nz."""

    z_levels: list[int]
    """The z level values themselves; ``len`` must equal *n_z*."""

    dz_um: Optional[float] = Field(default=None, ge=0)
    """Z step in micrometres; legitimately 0.0 on a single-plane acquisition (hence ge=0)."""

    pixel_size_um: Optional[float] = Field(default=None, gt=0)
    """Object-space pixel size (µm), binning-aware; use :meth:`require_pixel_size_um` when load-bearing."""

    wellplate_format: Optional[str] = None
    """e.g. ``"24 well plate"``. Optional: an OME/NGFF store need not declare one."""

    frame_shape: tuple[int, int]
    """One FOV's ``(height, width)`` in pixels — from a real decoded frame, not declared."""

    dtype: np.dtype
    """Native pixel dtype. A real ``np.dtype``, because consumers allocate against it."""

    n_t: int = Field(ge=1)
    """Timepoints. Folder/axis-derived, cross-checked against the declared Nt."""

    @field_validator("dtype", mode="before")
    @classmethod
    def _as_dtype(cls, v):
        # Readers pass `sample.dtype` (already a dtype) or `np.dtype(...)`; normalise both.
        return np.dtype(v)

    @field_validator("frame_shape", "z_levels", "regions", mode="before")
    @classmethod
    def _as_sequence(cls, v):
        # A bare numpy array does not iterate into a tuple field, so normalise here.
        return tuple(v) if isinstance(v, np.ndarray) else v

    @model_validator(mode="after")
    def _cross_check(self):
        missing = [r for r in self.regions if r not in self.fovs_per_region]
        if missing:
            raise ValueError(
                f"fovs_per_region has no entry for region(s) {missing[:8]} — a region with no "
                "FOV list renders as a blank well rather than as an error. The reader must "
                "list every region it reports."
            )
        if len(self.z_levels) != self.n_z:
            raise ValueError(
                f"z_levels has {len(self.z_levels)} entries but n_z is {self.n_z}; they name "
                "the same axis and a disagreement means one of them is wrong."
            )
        return self

    def require_pixel_size_um(self) -> float:
        """*pixel_size_um*, or raise naming the field and the file that supplies it."""
        if self.pixel_size_um is None:
            raise ValueError(
                "pixel_size_um is required here, but this acquisition has none. Without it "
                "micrometres cannot be converted to pixels and every FOV would be placed at "
                "the same spot — a plausible-looking but wrong image. Add "
                "objective.pixel_size_um to acquisition.yaml."
            )
        return float(self.pixel_size_um)

    def require_dz_um(self) -> float:
        """*dz_um*, or raise naming the field."""
        if not self.dz_um:
            raise ValueError(
                f"dz_um is required here, but this acquisition has dz_um={self.dz_um!r}. "
                "Defaulting it to 1.0 would render an anisotropic z-stack as an isotropic "
                "volume — on the tissue set, dz 1.5um against pixel 0.752um, i.e. 2x squashed "
                "in z, with nothing said. A 0.0 step is stored honestly on a single-plane "
                "acquisition but cannot be used as a scale either. Add a real "
                "z_stack.delta_z_mm to acquisition.yaml."
            )
        return float(self.dz_um)

    @property
    def channel_names(self) -> list[str]:
        """``[c.name for c in channels]`` — written out at a dozen call sites."""
        return [c.name for c in self.channels]

    def channel_index(self, name: str) -> int:
        """Index of *name* in the acquisition's channel order, or raise naming it (no fallback)."""
        names = self.channel_names
        if name not in names:
            raise KeyError(
                f"channel {name!r} is not a channel of this acquisition: {names}"
            )
        return names.index(name)

    # Mapping shim: keeps `reader.metadata["..."]` working; delete once the last subscript is gone.

    def __getitem__(self, key: str) -> Any:
        if key not in _ACQ_FIELDSET:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in _ACQ_FIELDSET else default

    def __contains__(self, key: object) -> bool:
        return key in _ACQ_FIELDSET

    def keys(self):
        return _ACQ_KEYS

    def values(self):
        return [getattr(self, k) for k in _ACQ_KEYS]

    def items(self):
        return [(k, getattr(self, k)) for k in _ACQ_KEYS]

    def __iter__(self) -> Iterator[str]:      # type: ignore[override]
        # Mapping iteration (so `dict(meta)` works); BaseModel.__iter__ yields (k, v) pairs.
        return iter(_ACQ_KEYS)

    def __len__(self) -> int:
        return len(_ACQ_KEYS)


# Field names resolved once at import: `model_fields` is a pydantic classproperty doing real
# work per access, ~18x slower than these lookups on the viewer's paint path.
_ACQ_KEYS: tuple[str, ...] = tuple(Acquisition.model_fields)
_ACQ_FIELDSET = frozenset(_ACQ_KEYS)
_CHANNEL_KEYS: tuple[str, ...] = tuple(Channel.model_fields)
_CHANNEL_FIELDSET = frozenset(_CHANNEL_KEYS)
