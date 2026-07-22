"""Physical / scalar acquisition metadata from ``acquisition.yaml`` (the single format).

``acquisition.yaml`` is Squid's authoritative metadata: the objective pixel size ALREADY
computed for the objective + camera binning (so no fragile sensor/mag recompute), the
wellplate format, and the z-stack / time-series parameters. It is **required**.

The legacy flat ``acquisition parameters.json`` is intentionally NOT supported: every current
Squid acquisition writes ``acquisition.yaml``, so a JSON fallback has no real input — it would
be dead code carrying a permanent second-format test burden. One format, required, loud on
absence. (If a genuinely pre-yaml dataset ever resurfaces, convert it to ``acquisition.yaml``
up front rather than adding a second read path here.)

``coordinates.csv`` is not read *here* — this module owns the scalar/physical metadata only.
Per-FOV stage positions moved into ``reader.load_fov_positions`` (IMA-187), which needs the
filename-derived FOV index to key rows against and so belongs beside the reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# ══════════════════════════════════════════════════════════════════════════════════════════
# The typed acquisition model.
#
# Everything above this line answers "what does acquisition.yaml say". Everything below
# answers "what IS this acquisition" — the whole of `reader.metadata`, validated ONCE at the
# reader boundary instead of being trusted at ~96 separate call sites.
#
# Why this exists. `reader.metadata` was a raw dict. pydantic was already a dependency but
# was used only on the COMMAND LINE (`_cli.ProcessParameters`), so we validated the thing a
# human typed and not the thing a microscope wrote. A missing key did not raise; it surfaced
# as a blank render or an opaque failure several layers away. `pixel_size_um` is the
# documented case — `_placement._require_pixel_size` is a hand-written guard for exactly one
# field, which is what a schema looks like before you write it down.
#
# Two deliberate design choices:
#
# * `Acquisition` is ALSO a Mapping. The dict is read in ~96 places; a flag-day rewrite of
#   all of them could not be kept green commit-by-commit, and a big-bang migration of the
#   thing every render depends on is exactly the change you cannot review. Subscript access
#   stays, call sites move to attributes incrementally, and the validation benefit lands on
#   commit one for every consumer at once.
# * The optional fields stay Optional and get LOUD accessors. `pixel_size_um` is genuinely
#   absent on some acquisitions, so modelling it required would refuse datasets that open
#   fine today. Modelling it Optional and letting `None` flow is how the mosaic silently
#   collapsed. `require_pixel_size_um()` is the third answer: honest about absence, fatal at
#   the point of use, and it names the field and the file to fix.
# ══════════════════════════════════════════════════════════════════════════════════════════


class Channel(BaseModel):
    """One channel of the acquisition, keyed on its canonical (filename) name.

    Mirrors what ``_channels.resolve_channels`` produces. ``display_color`` is required, not
    optional: ``resolve_channels`` already REFUSES a channel it cannot colour rather than
    handing back a placeholder white, and this model must not quietly re-open that door.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Canonical filename form, e.g. ``Fluorescence_488_nm_Ex``. The key ``read()`` accepts."""

    display_name: str
    """Human label for the UI. Falls back to ``name`` when the channel YAML has none."""

    display_color: str
    """Hex colour, e.g. ``#1FFF00``. Never a placeholder — see :func:`resolve_channels`."""

    ex: Optional[float] = None
    """Excitation wavelength (nm) when the channel YAML records one."""

    # -- Mapping shim: `c["name"]` is written at many call sites; keep it working. -------
    def __getitem__(self, key: str):
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default) if key in type(self).model_fields else default

    def __contains__(self, key: object) -> bool:
        return key in type(self).model_fields

    def keys(self):
        return type(self).model_fields.keys()


class Acquisition(BaseModel):
    """The validated acquisition model — what ``reader.metadata`` returns.

    Constructed once, at the reader boundary, from what the reader actually produces. Every
    field below exists because a reader emits it; none were invented.
    """

    # arbitrary_types_allowed: `dtype` is a real numpy dtype and must STAY one — consumers
    # allocate against it. extra="forbid": a typo'd key (`pixel_size` for `pixel_size_um`)
    # must be refused here, not stored and silently never read.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    regions: list[str]
    """Well / region ids, plate-ordered."""

    fovs_per_region: dict[str, list[int]]
    """``{region: [fov, ...]}``. Must cover every region — see the validator."""

    fov_positions_um: dict[tuple[str, int], tuple[float, float]]
    """``{(region, fov): (x_um, y_um)}`` — stage MICROMETRES. ``{}`` when the acquisition
    carries no usable positions, which degrades the mosaic to a single field (a documented
    degradation, unlike a wrong-unit value: see ``_placement``'s silent-1000x note)."""

    channels: list[Channel]
    """Acquisition channels, in C-axis order."""

    n_z: int = Field(ge=1)
    """Distinct z levels. Filename/page-derived, cross-checked against the declared Nz."""

    z_levels: list[int]
    """The z level values themselves; ``len`` must equal *n_z*."""

    dz_um: Optional[float] = Field(default=None, gt=0)
    """Z step in micrometres. Optional; use :meth:`require_dz_um` where it is load-bearing."""

    pixel_size_um: Optional[float] = Field(default=None, gt=0)
    """Object-space pixel size (µm), binning-aware, straight from ``acquisition.yaml``.
    Optional; use :meth:`require_pixel_size_um` where it is load-bearing."""

    wellplate_format: Optional[str] = None
    """e.g. ``"24 well plate"``. Optional: an OME/NGFF store need not declare one."""

    frame_shape: tuple[int, int]
    """One FOV's ``(height, width)`` in pixels — from a real decoded frame, not declared."""

    dtype: np.dtype
    """Native pixel dtype. A real ``np.dtype``, because consumers allocate against it."""

    n_t: int = Field(ge=1)
    """Timepoints. Folder/axis-derived, cross-checked against the declared Nt."""

    # -- validators: refuse at the boundary what would otherwise mis-render far away -----

    @field_validator("dtype", mode="before")
    @classmethod
    def _as_dtype(cls, v):
        # Readers pass `sample.dtype` (already a dtype) or `np.dtype(...)`; normalise both.
        return np.dtype(v)

    @field_validator("frame_shape", "z_levels", "regions", mode="before")
    @classmethod
    def _as_sequence(cls, v):
        # `tuple(sample.shape)` arrives as a tuple of np.int64; pydantic coerces those fine,
        # but a bare numpy array does not iterate into a tuple field, so normalise here.
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

    # -- loud accessors for the genuinely-optional fields --------------------------------

    def require_pixel_size_um(self) -> float:
        """*pixel_size_um*, or raise naming the field and the file that supplies it.

        Call this wherever a missing pixel size would produce a WRONG PICTURE rather than an
        error: micrometre->pixel conversion, mosaic placement, physical scale on a layer.
        Never substitute a default — a mosaic drawn at the wrong scale looks like data.
        """
        if self.pixel_size_um is None:
            raise ValueError(
                "pixel_size_um is required here, but this acquisition has none. Without it "
                "micrometres cannot be converted to pixels and every FOV would be placed at "
                "the same spot — a plausible-looking but wrong image. Add "
                "objective.pixel_size_um to acquisition.yaml."
            )
        return float(self.pixel_size_um)

    def require_dz_um(self) -> float:
        """*dz_um*, or raise naming the field.

        A missing z step rendered as 1.0 makes an anisotropic stack look isotropic — on the
        tissue set, dz 1.5µm against pixel 0.752µm, i.e. 2x squashed in z, with nothing said.
        """
        if self.dz_um is None:
            raise ValueError(
                "dz_um is required here, but this acquisition has none. Defaulting it to 1.0 "
                "would render an anisotropic z-stack as an isotropic volume. Add "
                "z_stack.delta_z_mm to acquisition.yaml."
            )
        return float(self.dz_um)

    # -- convenience that removes a repeated comprehension, not an abstraction -----------

    @property
    def channel_names(self) -> list[str]:
        """``[c.name for c in channels]`` — written out at a dozen call sites."""
        return [c.name for c in self.channels]

    def channel_index(self, name: str) -> int:
        """Index of *name* in the acquisition's channel order, or raise naming it.

        Deliberately has NO fallback. ``names.index(x) if x in names else 0`` is the exact
        shape of the registration bug in ``_stitch``: it answers a question it could not
        answer, with a value that is indistinguishable from a correct one.
        """
        names = self.channel_names
        if name not in names:
            raise KeyError(
                f"channel {name!r} is not a channel of this acquisition: {names}"
            )
        return names.index(name)

    # -- Mapping shim -------------------------------------------------------------------
    # `reader.metadata["..."]` is written ~96 times. These keep every one of them working so
    # the migration to attributes is incremental and each step stays green. Delete this
    # block once the last subscript is gone.

    def __getitem__(self, key: str) -> Any:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in type(self).model_fields else default

    def __contains__(self, key: object) -> bool:
        return key in type(self).model_fields

    def keys(self):
        return type(self).model_fields.keys()

    def values(self):
        return [getattr(self, k) for k in type(self).model_fields]

    def items(self):
        return [(k, getattr(self, k)) for k in type(self).model_fields]

    def __iter__(self) -> Iterator[str]:      # type: ignore[override]
        # Mapping iteration (so `dict(meta)` works). BaseModel.__iter__ yields (k, v) pairs;
        # a dict built from THAT is right by accident and wrong for `for k in meta`.
        return iter(type(self).model_fields)

    def __len__(self) -> int:
        return len(type(self).model_fields)
