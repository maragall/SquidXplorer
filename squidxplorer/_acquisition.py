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

import json
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


def load_objective_na(root) -> Optional[float]:
    """Objective numerical aperture, from ``acquisition parameters.json``. ``None`` if unstated.

    NOT the JSON fallback this module refuses above. That refusal is about a SECOND SOURCE for
    fields ``acquisition.yaml`` already carries; the NA is not one of them. Squid's own writer
    (``control/acquisition_yaml_loader.py``, and every yaml on this machine) puts
    ``name / magnification / pixel_size_um / camera_binning`` in the ``objective:`` block and no
    aperture at all, while ``acquisition parameters.json`` records
    ``objective: {magnification, NA, tube_lens_f_mm, name}``. So this file is the ONLY place the
    NA is written, and reading it here is what stops a caller from going to a whole second
    acquisition parser to get one number.

    Returns ``None`` — never a default — when the file is absent, unparseable, or states no NA.
    An NA is the width of the PSF; guessing one would produce a confidently wrong deconvolution,
    so the refusal belongs to the caller that knows what the number is for
    (:func:`squidxplorer._decon.optics_for_channel`).
    """
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

    exposure_time_ms: Optional[float] = None
    """Camera exposure (ms) when the channel YAML records one.

    THIS FIELD WAS CALLED ``ex`` AND DOCUMENTED AS "excitation wavelength (nm)", while
    ``_channels._extract_exposure`` filled it with ``exposure_time_ms``. Nothing outside the
    tests read it, so the contradiction never showed up as a wrong picture — but it is exactly
    the kind of double-booked name that :attr:`excitation_nm` below would have collided with.
    Renamed to what it holds."""

    excitation_nm: Optional[float] = None
    """Excitation wavelength (nm), parsed from the channel name by ``_channels.excitation_nm``.

    ``None`` for a broadband channel (``BF_LED_matrix_full``, ``DF_LED_matrix``), which is an
    answer rather than a gap: a consumer that needs a wavelength must refuse, not substitute.
    Deconvolution's PSF is derived from this (``_decon.OpticsParams.from_acquisition``); no
    acquisition on this machine states an EMISSION wavelength anywhere, so the Stokes-shifted
    emission stays an explicit estimate made in the optics layer and is not stored here as if
    the instrument had reported it."""

    # -- Mapping shim: `c["name"]` is written at many call sites; keep it working. -------
    # Membership is tested against the module-level _CHANNEL_KEYS, never against
    # `model_fields` — see the note above _ACQ_KEYS.
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

    dz_um: Optional[float] = Field(default=None, ge=0)
    """Z step in micrometres. Optional, and legitimately **0.0** on a single-plane acquisition
    (``delta_z_mm: 0`` with ``nz: 1``), which is why the bound here is ``ge=0`` and not ``gt=0``
    — an early draft used ``gt=0`` and correctly refused four of this repo's own fixtures.
    A zero step is only meaningless once it is used as a z SCALE, so that is where it is
    refused: :meth:`require_dz_um`."""

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
        # Mapping iteration (so `dict(meta)` works). BaseModel.__iter__ yields (k, v) pairs;
        # a dict built from THAT is right by accident and wrong for `for k in meta`.
        return iter(_ACQ_KEYS)

    def __len__(self) -> int:
        return len(_ACQ_KEYS)


# The field names, resolved ONCE at import.
#
# This is not a micro-optimisation, it is a bug fix. `model_fields` is a pydantic
# CLASSPROPERTY that does real work on every access, so the first draft of the Mapping shim
# above — which consulted it on every `meta["..."]` — was ~18x slower than the plain dict it
# replaced (measured: 0.42us vs 0.023us per lookup). `reader.metadata` is read in the viewer's
# paint and ingest paths, and that was enough to push Qt tests past their _drain_until
# timeouts: test_viewer.py went from ~60s to ~110s and failed a DIFFERENT test on 2 of 3 runs,
# which reads exactly like flakiness and was not. Membership is a frozenset, order comes from
# the tuple, and neither touches pydantic at lookup time.
_ACQ_KEYS: tuple[str, ...] = tuple(Acquisition.model_fields)
_ACQ_FIELDSET = frozenset(_ACQ_KEYS)
_CHANNEL_KEYS: tuple[str, ...] = tuple(Channel.model_fields)
_CHANNEL_FIELDSET = frozenset(_CHANNEL_KEYS)
