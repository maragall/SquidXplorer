"""The sample holder as ONE model: a grid of cells, each holding 0..N FOVs (IMA-214).

A slide carrier IS a plate. Both are a rectangular grid of addressable cells at a fixed physical
pitch; the only real difference is how a cell is NAMED ("B3" vs "manual0") and how many FOVs it
holds. Modelling them as two unrelated things is exactly what would force the mosaic, selection
and loupe code to fork -- so they do not fork here::

    Plate (ABC)          grid + physical geometry + occupancy; everything downstream talks to this
      +- WellPlate       cells named A1..H12, geometry from Squid's sample_formats.csv
      +- SlideCarrier    cells named by the acquisition's own freeform region ids (manual0, ...)

Why this module exists at all, concretely: the viewer refuses the real tissue dataset
(``regions = ["manual0", "manual1"]``) because every layout path in it parses region ids as well
ids. A glass slide has no well grid and no well ids, and pretending it is a degenerate 1x1
wellplate would put both regions in the same cell. Here a freeform region is a first-class cell.

UNITS
-----
Micrometres everywhere, every value ending ``_um``, per ``_placement.py``'s contract. Squid's
``sample_formats.csv`` is MILLIMETRES; it is converted exactly once, in :func:`load_sample_formats`,
at the producer -- the same discipline the reader applies to coordinates.csv. A bare mm value must
never travel in a geometry attribute; that is the silent-1000x defect class.

NAMED CELLS vs POSITIONED CELLS (IMA-253)
-----------------------------------------
A well id ENCODES ITS POSITION: "B3" *is* row 1, column 2, and no stage coordinate is needed to
lay a well plate out. A freeform region id does not: ``manual0`` and ``manual1`` say nothing
about where on the glass those tissues sat. Assigning them cells by ENUMERATION ORDER -- which is
what a "1 x N carrier, left to right, in the order the acquisition reports them" rule does -- is
therefore inventing a layout, and it is why the real 10x tissue acquisition rendered as two
squares side by side when the two regions are in fact separated in Y and overlapping in X.

So the choice of layout is driven by WHETHER THE IDS CARRY POSITION, never by the format string::

    well_span(regions) is not None   ->  WellPlate: ids are the layout. Unchanged, forever.
    well_span(regions) is None       ->  freeform: the layout comes from fov_positions_um.

For the freeform case each region gets a cell that is ITS OWN MOSAIC'S BOUNDING BOX in stage
micrometres (:func:`region_stage_boxes_um`), so regions of different size and geometry -- which is
the normal case for tissue -- get different-sized cells instead of being stretched or cropped into
a uniform grid. :func:`freeform_grid` then derives (row, col) by CLUSTERING those boxes rather
than by enumerating names, and :func:`freeform_layout` emits the exact rectangles, normalised into
the grid's own units, that the viewer draws. Shuffling the region names changes nothing.

STAGE vs COMPACT PLACEMENT (Task 5, 2026-07-29)
-----------------------------------------------
``build_plate(..., placement="stage" | "compact")``. The words are defined once, in
:mod:`squidmip._placement`, so the geometry and the label on screen cannot spell them differently.

    stage    (DEFAULT) cells where the stage says. Byte-identical to what this module has always
             built. A well id encodes its own position, so an unacquired well keeps its space and
             the plate measures like the plate.
    compact  the space BETWEEN regions is closed. A 3-well scan of a 384-well plate reads as three
             large cells rather than three dots in a 16 x 24 field of nothing.

Compact is :func:`even_carrier_layout`, which already did exactly this for tissue carriers,
generalised to every format. It is a PROMOTION of a shipped layout, not a new one.
:func:`freeform_grid` / :func:`freeform_layout` remain as the stage-proportional pair.

**What compact never moves.** FOVs. Overlapping or adjacent FOVs carry the registration stitching
solves against; sparse ones (Squid schema v2's ``grid_subset`` and ``random`` patterns) carry
sampling geometry. Either way the space inside a region carries information, so only the space
between regions is free. Within-region geometry lives in :mod:`squidmip._placement` and takes no
mode argument at all, which is how that guarantee is enforced rather than merely intended.

**Why the mode is reported, not remembered.** This codebase refuses to guess positions
(``reader.py`` raises rather than placing FOVs "at positions that would look plausible but be
wrong"). A compact view is a PRESENTATION, and a compact view mistaken for a stage view is a
mis-measurement that ends up in a figure. So :attr:`Plate.placement_mode` is the mode the cells
ACTUALLY have and is meant to be on screen at all times, while :attr:`Plate.placement_requested`
is what the caller asked for. The two differ in exactly one case, and it is not hypothetical: a
FREEFORM tissue carrier has no stage-proportional layout in the product, because Julio removed it
on 2026-07-23 (it "stacked two tissues into a tall, tiny, uneven column"). Its cells are even in
both modes, so it reports ``compact`` even when ``stage`` was asked for. Labelling that "stage"
would be the lie this attribute exists to prevent.

DECLARED vs MEASURED
--------------------
``~/Downloads/synthetic_2x2_wellplate`` declares ``384 well plate`` in its yaml but its stage
coordinates measure a 9.000 mm pitch on both axes, which is physically a 96-well plate. 96 and 384
differ by EXACTLY 2x in pitch, so believing the declaration draws the carrier art at exactly half
scale (the IMA-220 hazard). The precedence rule is therefore::

    override  >  measured  >  declared  >  inferred-from-span

with MEASURED beating DECLARED, loudly (a ``UserWarning`` naming both formats and the measured
pitch). Rationale: the declaration is a string a human typed into a yaml; the pitch is physics
recorded by the stage. The measurement only wins when it is unambiguous -- both axes agree on one
standard pitch to within tolerance, AND the resulting grid is big enough to contain every observed
well. Otherwise the declaration stands and the disagreement is still warned about, never swallowed.

Prior art
---------
* **OME-NGFF 0.4/0.5 plate spec** -- a plate is ``{rows: [{name}], columns: [{name}], wells:
  [{path, rowIndex, columnIndex}]}``, and every row/column of the physical plate MUST be declared
  even when unpopulated. It is a purely LOGICAL grid: no pitch, no well diameter, no A1 offset, no
  units anywhere in the schema. Taken: 0-based rowIndex/columnIndex addressing, and the
  full-grid-vs-present-wells distinction (:attr:`Plate.cell_ids` vs :attr:`Plate.occupied_cells`).
  Not taken: the absence of geometry -- carrier art and stage placement need real micrometres.
* **OME-XML 2016-06** (which NGFF dropped this from) -- ``Plate/@WellOriginX|Y`` + explicit
  ``@WellOriginXUnit``, and ``RowNamingConvention``/``ColumnNamingConvention``. Taken: the idea
  that the origin is a first-class plate attribute carrying its unit. Even OME-XML has no PITCH.
* **ngio / Fractal** (``OmeZarrPlate`` -> ``OmeZarrWell`` -> ``OmeZarrContainer``) -- composition,
  not inheritance, and physical coordinates pushed out into ROI tables (``x_micrometer``,
  ``len_x_micrometer``). Taken: the ``_micrometer``-suffixed-everywhere discipline, which is our
  ``_um`` rule. Not taken: the split, since our cells need geometry to draw a carrier.
* **Opentrons labware schema v2** -- the only ecosystem that models holder geometry properly:
  ``dimensions``, ``cornerOffsetFromSlot``, and per-well ``{x, y, depth, diameter}``. Taken
  directly as the shape of :class:`PlateGeometry` (offset + pitch + cell size), which is what
  OME lacks.
* **CellProfiler / platetools** -- plates are metadata STRINGS (``Metadata_Plate``,
  ``Metadata_Well``) or well-id<->(row, col) integer conversion over a size enum
  (6/12/24/48/96/384/1536). No geometry. Confirms the well-id parsing here is conventional.
* **Slide carriers** -- no public library has a first-class ``SlideCarrier`` type. Where carriers
  exist (Opera Phenix 1- and 4-slide holders), the vendor pattern is to DECLARE THE CARRIER AS A
  PLATE TYPE with N wells; slide carriers are even built to the ANSI/SLAS microplate footprint so
  plate-shaped machinery holds them. That is independent confirmation of this ticket's design
  insight, so :class:`SlideCarrier` subclasses :class:`Plate` rather than forking.
* **Declared-vs-measured reconciliation** -- searched and found NO prior art. Micro-Manager's
  ``SBSPlate`` takes the size as a user-picked enum and only calibrates the A1 offset for an
  already-declared format; information flows declared -> predicted coordinates everywhere. The
  inverse (measured coordinates validating the declaration) is unoccupied ground, so the rule
  below is ours and is documented rather than borrowed.
* **Squid upstream** (``control/_def.py:read_sample_formats_csv``, ``core.py:NavigationViewer``) --
  the CSV schema, the ``"{n} well plate"`` key convention, the carrier-PNG filenames and the
  per-sample ``mm_per_pixel`` / origin-pixel art scale. Taken wholesale so our art lines up with
  Squid's; mirrored in :data:`_ART`, never invented.
* **_plate_shape.py (IMA-219)** -- format normalisation and SPAN+SNAP inference. Reused, not
  duplicated: this module calls ``normalize_plate_format`` / ``infer_plate_format`` and only adds
  the geometry-measured tier that IMA-219's docstring flags as future work (its "D5").
"""

from __future__ import annotations

import csv
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from ._placement import (
    COMPACT,
    DEFAULT_PLACEMENT_MODE,
    PLACEMENT_MODES,
    STAGE,
    normalize_placement_mode,
    placement_mode_label,
)
from ._plate_shape import (
    GLASS_SLIDE,
    PlateShapeError,
    infer_plate_format,
    normalize_plate_format,
    well_span,
)

__all__ = [
    "COMPACT",
    "DEFAULT_PLACEMENT_MODE",
    "PLACEMENT_MODES",
    "STAGE",
    "CarrierArt",
    "CompactPlate",
    "Plate",
    "PlateBuildError",
    "PlateGeometry",
    "SlideCarrier",
    "WellPlate",
    "build_plate",
    "carrier_art",
    "format_from_pitch_um",
    "freeform_grid",
    "freeform_layout",
    "load_sample_formats",
    "measure_region_pitch_um",
    "normalize_placement_mode",
    "placement_mode_label",
    "region_stage_boxes_um",
    "squid_images_dir",
]

# Carrier formats that are a grid of SLIDES, not wells. "glass slide" is Squid's own name for the
# single-slide holder; "4 slide carrier" is the 4-up carrier its GUI calls "4 glass slide".
FOUR_SLIDE_CARRIER = "4 slide carrier"
_SLIDE_FORMATS = (GLASS_SLIDE, FOUR_SLIDE_CARRIER)

# Where Squid's checkout may live, for the CSV and the carrier PNGs. Env var first so a user with
# an unusual layout is never stuck; every lookup DEGRADES to None/vendored rather than guessing.
_SQUID_ENV = "SQUIDMIP_SQUID_SOFTWARE"
_SQUID_GUESSES = (
    Path.home() / "Cephla" / "projects" / "Squid" / "software",
    Path.home() / "Squid" / "software",
    Path.home() / "projects" / "Squid" / "software",
)

# Squid's sample_formats.csv, vendored verbatim (mm, as upstream writes it) so a machine with no
# Squid checkout still lays plates out correctly. Values mirror
# Squid/software/objective_and_sample_formats/sample_formats.csv.
#   name: (a1_x_mm, a1_y_mm, a1_x_px, a1_y_px, well_size_mm, well_spacing_mm, skip, rows, cols)
_VENDORED_MM = {
    GLASS_SLIDE:        (0.0,   0.0,   0,   0,   0.0,   0.0,  0,  1,  1),
    "6 well plate":     (24.55, 23.01, 290, 272, 34.94, 39.2, 0,  2,  3),
    "12 well plate":    (24.75, 16.86, 293, 198, 22.05, 26.0, 0,  3,  4),
    "24 well plate":    (24.45, 22.07, 233, 210, 15.54, 19.3, 0,  4,  6),
    "96 well plate":    (11.31, 10.75, 171, 135, 6.21,  9.0,  0,  8, 12),
    "384 well plate":   (12.05, 9.05,  143, 106, 3.3,   4.5,  1, 16, 24),
    "1536 well plate":  (11.01, 7.87,  130, 93,  1.53,  2.25, 0, 32, 48),
}

# The 4-up slide carrier is NOT in sample_formats.csv -- upstream hardcodes it in the GUI
# ("4 glass slide" -> images/4 slide carrier_1509x1010.png, mm_per_pixel 0.084665, origin 50,0).
# Slot pitch/size are derived from a standard 75 x 25 mm slide sitting in a 4-up carrier; they are
# a LAYOUT approximation, not a measured calibration, and are only used to place cells on the art.
_VENDORED_MM[FOUR_SLIDE_CARRIER] = (14.0, 20.0, 50, 0, 25.0, 27.0, 0, 1, 4)

# Squid NavigationViewer.update_display_properties: mm/px of each background PNG, and the PNG name.
# Filenames are copied from upstream's image_paths dict -- NEVER constructed -- so a missing
# checkout yields None instead of a plausible-but-wrong path.
#   format -> (png filename, mm_per_pixel)
_ART = {
    GLASS_SLIDE:       ("slide carrier_828x662.png",   0.1453),
    FOUR_SLIDE_CARRIER: ("4 slide carrier_1509x1010.png", 0.084665),
    "6 well plate":    ("6 well plate_1509x1010.png",    0.084665),
    "12 well plate":   ("12 well plate_1509x1010.png",   0.084665),
    "24 well plate":   ("24 well plate_1509x1010.png",   0.084665),
    "96 well plate":   ("96 well plate_1509x1010.png",   0.084665),
    "384 well plate":  ("384 well plate_1509x1010.png",  0.084665),
    "1536 well plate": ("1536 well plate_1509x1010.png", 0.084665),
}
# Upstream hardcodes the art origin for the two slide holders instead of deriving it from a1.
_ART_ORIGIN_PX = {GLASS_SLIDE: (200.0, 120.0), FOUR_SLIDE_CARRIER: (50.0, 0.0)}

# Fractional slack when matching a measured pitch to a standard one. The closest pair of standard
# pitches is 96wp/384wp at 2.0x apart, so 5% is loose enough for stage noise and nowhere near
# ambiguous.
_PITCH_TOL = 0.05


class PlateBuildError(ValueError):
    """The acquisition cannot be expressed as a plate (regions outside the grid, too many slides)."""


# --------------------------------------------------------------------------- geometry

@dataclass(frozen=True)
class PlateGeometry:
    """Physical layout of one sample format. MICROMETRES; ``_px`` values are art pixels.

    ``pitch_*_um`` is centre-to-centre cell spacing, ``cell_size_um`` the well diameter / slide
    width, ``a1_*_um`` the stage position of the top-left cell's centre. ``a1_*_px`` is where that
    same point sits in the carrier PNG, which is what makes :class:`CarrierArt` able to convert.
    """

    name: str
    rows: int
    cols: int
    a1_x_um: float
    a1_y_um: float
    pitch_x_um: float
    pitch_y_um: float
    cell_size_um: float
    a1_x_px: int = 0
    a1_y_px: int = 0
    number_of_skip: int = 0

    @classmethod
    def from_mm(cls, name: str, row) -> "PlateGeometry":
        """Build from one sample_formats.csv row's MILLIMETRES. The only mm->um conversion here."""
        a1x, a1y, a1xp, a1yp, size, spacing, skip, rows, cols = row
        return cls(
            name=name,
            rows=int(rows),
            cols=int(cols),
            a1_x_um=float(a1x) * 1000.0,
            a1_y_um=float(a1y) * 1000.0,
            pitch_x_um=float(spacing) * 1000.0,
            pitch_y_um=float(spacing) * 1000.0,
            cell_size_um=float(size) * 1000.0,
            a1_x_px=int(a1xp),
            a1_y_px=int(a1yp),
            number_of_skip=int(skip),
        )

    @classmethod
    def vendored(cls, name: str) -> "PlateGeometry":
        """Geometry for *name* from the vendored table, ignoring any Squid checkout."""
        key = _canonical_format(name)
        if key not in _VENDORED_MM:
            raise PlateShapeError(f"{name!r} is not a known sample format.")
        return cls.from_mm(key, _VENDORED_MM[key])


def _canonical_format(name) -> str:
    """Canonical format name, extending ``normalize_plate_format`` with the slide CARRIERS.

    IMA-219's normaliser collapses anything containing "slide" to ``"glass slide"`` -- correct for
    its question ("is this a plate at all?"), wrong for ours, because a 4-up carrier is a real
    4-cell grid. So the carrier is disambiguated here and everything else is delegated, never
    reimplemented.
    """
    s = str(name or "").strip().lower()
    if "slide" in s and ("4" in s or "four" in s):
        return FOUR_SLIDE_CARRIER
    resolved = normalize_plate_format(name, strict=False)
    if resolved is None:
        raise PlateShapeError(f"{name!r} is not a known sample format.")
    return resolved


def load_sample_formats(csv_path=None) -> dict[str, PlateGeometry]:
    """``{format name: PlateGeometry}`` from Squid's sample_formats.csv, in MICROMETRES.

    *csv_path* defaults to the CSV inside a discovered Squid checkout. A missing or unreadable CSV
    is NOT an error: the vendored table is returned instead, so SquidMIP lays plates out correctly
    on a machine that has no Squid source at all. The 4-up slide carrier is always merged in --
    upstream keeps it out of the CSV and hardcodes it in the GUI.
    """
    formats = {name: PlateGeometry.from_mm(name, row) for name, row in _VENDORED_MM.items()}
    path = Path(csv_path) if csv_path is not None else _default_formats_csv()
    if path is None or not Path(path).is_file():
        return formats
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                raw = str(row["format"]).strip()
                name = f"{raw} well plate" if raw.isdigit() else raw
                formats[name] = PlateGeometry.from_mm(
                    name,
                    (row["a1_x_mm"], row["a1_y_mm"], row["a1_x_pixel"], row["a1_y_pixel"],
                     row["well_size_mm"], row["well_spacing_mm"], row["number_of_skip"],
                     row["rows"], row["cols"]),
                )
    except (OSError, KeyError, ValueError) as e:
        warnings.warn(f"unreadable sample_formats.csv at {path} ({e}); using vendored geometry.")
    return formats


def _squid_software_dir() -> Optional[Path]:
    env = os.environ.get(_SQUID_ENV)
    for cand in ([Path(env)] if env else []) + list(_SQUID_GUESSES):
        if cand.is_dir():
            return cand
    return None


def _default_formats_csv() -> Optional[Path]:
    root = _squid_software_dir()
    if root is None:
        return None
    p = root / "objective_and_sample_formats" / "sample_formats.csv"
    return p if p.is_file() else None


def squid_images_dir() -> Optional[Path]:
    """Squid's ``software/images`` directory, or None. Never raises: carrier art is optional."""
    root = _squid_software_dir()
    if root is None:
        return None
    p = root / "images"
    return p if p.is_dir() else None


# --------------------------------------------------------------------------- carrier art

@dataclass(frozen=True)
class CarrierArt:
    """A carrier background PNG plus the transform that puts stage micrometres onto its pixels.

    ``um_per_px`` and the origin are Squid's own (NavigationViewer), so an overlay drawn through
    this lands where Squid's navigator would draw it. IMA-220 consumes this; getting the scale
    from the WRONG plate format is what would render it at 2x.
    """

    format_name: str
    path: Path
    um_per_px: float
    origin_x_px: float
    origin_y_px: float

    def um_to_px(self, x_um: float, y_um: float) -> tuple[float, float]:
        """Stage micrometres -> pixel coordinates in this PNG."""
        return (self.origin_x_px + x_um / self.um_per_px,
                self.origin_y_px + y_um / self.um_per_px)


def carrier_art(format_name, images_dir=None, geometry: Optional[PlateGeometry] = None
                ) -> Optional[CarrierArt]:
    """The carrier PNG for *format_name*, or None when it is not on disk.

    Degrades to None rather than inventing a filename: every name in :data:`_ART` is copied from
    Squid's ``image_paths`` dict, and a name with no entry -- or an entry whose file is absent --
    yields None so callers draw their own grid instead of failing to load a fabricated path.
    """
    try:
        name = _canonical_format(format_name)
    except PlateShapeError:
        return None
    entry = _ART.get(name)
    if entry is None:
        return None
    filename, mm_per_px = entry
    root = Path(images_dir) if images_dir is not None else squid_images_dir()
    if root is None:
        return None
    path = root / filename
    if not path.is_file():
        return None
    um_per_px = mm_per_px * 1000.0
    if name in _ART_ORIGIN_PX:
        ox, oy = _ART_ORIGIN_PX[name]
    else:
        g = geometry or PlateGeometry.vendored(name)
        # Squid: origin_pixel = a1_pixel - a1_mm / mm_per_pixel  (same identity in um)
        ox = g.a1_x_px - g.a1_x_um / um_per_px
        oy = g.a1_y_px - g.a1_y_um / um_per_px
    return CarrierArt(format_name=name, path=path, um_per_px=um_per_px,
                      origin_x_px=float(ox), origin_y_px=float(oy))


# --------------------------------------------------------------------------- the Plate ABC

class Plate(ABC):
    """A grid of cells, each holding 0, 1 or many FOVs.

    Subclasses supply only NAMING -- how a cell id maps to a (row, col) and back, and what the
    axis labels are. Geometry, occupancy, extent and carrier art are shared, which is the whole
    point: mosaic/selection/loupe code written against ``Plate`` serves wells and slides alike.
    """

    def __init__(
        self,
        geometry: PlateGeometry,
        occupancy: Optional[Mapping[str, Sequence[int]]] = None,
        *,
        format_name: Optional[str] = None,
        declared_format: Optional[str] = None,
        format_source: str = "declared",
        measured_pitch_um: Optional[tuple] = None,
        placement_mode: str = DEFAULT_PLACEMENT_MODE,
        placement_requested: Optional[str] = None,
    ):
        self.geometry = geometry
        self.format_name = format_name or geometry.name
        self.declared_format = declared_format
        #: how ``format_name`` was decided: "override" | "measured" | "declared" | "inferred".
        self.format_source = format_source
        self.measured_pitch_um = measured_pitch_um
        #: the geometry the cells ACTUALLY have: "stage" | "compact". What the label must show.
        self.placement_mode = normalize_placement_mode(placement_mode)
        #: the mode the caller ASKED for. Differs from ``placement_mode`` only for a freeform
        #: tissue carrier, whose cells are even in both modes (see the module docstring).
        self.placement_requested = normalize_placement_mode(
            placement_mode if placement_requested is None else placement_requested
        )
        self._occupancy = {k: list(v) for k, v in (occupancy or {}).items()}
        unknown = [c for c in self._occupancy if not self.has_cell(c)]
        if unknown:
            raise PlateBuildError(
                f"region(s) {unknown[:8]} are not cells of a {self.format_name} "
                f"({self.rows} x {self.cols}). The acquisition and the resolved format disagree; "
                f"refusing to drop regions silently."
            )

    # -- naming: the only thing subclasses define -----------------------------------------
    @property
    @abstractmethod
    def row_labels(self) -> list[str]:
        """Axis labels for the rows, top to bottom."""

    @property
    @abstractmethod
    def col_labels(self) -> list[str]:
        """Axis labels for the columns, left to right."""

    @abstractmethod
    def cell_id(self, row: int, col: int) -> str:
        """Id of the cell at zero-based (row, col)."""

    @abstractmethod
    def cell_index(self, cell_id: str) -> tuple[int, int]:
        """Zero-based (row, col) of *cell_id*. Raises KeyError if it is not on this plate."""

    # -- shared grid ----------------------------------------------------------------------
    @property
    def rows(self) -> int:
        return self.geometry.rows

    @property
    def cols(self) -> int:
        return self.geometry.cols

    @property
    def cell_ids(self) -> list[str]:
        """Every cell, row-major (left to right, top to bottom) -- the plate's canonical order."""
        return [self.cell_id(r, c) for r in range(self.rows) for c in range(self.cols)]

    def has_cell(self, cell_id: str) -> bool:
        try:
            self.cell_index(cell_id)
        except KeyError:
            return False
        return True

    #: Prefix of the SYNTHETIC ids a subclass invents to keep its grid rectangular ("slot3",
    #: "pad5"). Empty means every cell id is a real region. Shared so :meth:`_sole_cell` can tell
    #: a real region from a filler without each subclass re-deciding.
    _FILLER_PREFIX = ""

    def _is_filler(self, cell_id: str) -> bool:
        """Is *cell_id* a synthetic placeholder rather than a region the acquisition named?"""
        return bool(self._FILLER_PREFIX) and str(cell_id).startswith(self._FILLER_PREFIX)

    def _sole_cell(self, axis: int, i: int) -> Optional[str]:
        """The id of the ONLY real region on row (*axis* 0) or column (*axis* 1) *i*, else None.

        The axis label of a holder whose cells are named rather than positioned: a row or column
        holding exactly one region is unambiguously that region's, and one holding several is left
        blank rather than labelled with a guess.
        """
        n = self.cols if axis == 0 else self.rows
        hits = [cid for cid in (self.cell_id(i, j) if axis == 0 else self.cell_id(j, i)
                                for j in range(n)) if not self._is_filler(cid)]
        return hits[0] if len(hits) == 1 else None

    # -- physical -------------------------------------------------------------------------
    @property
    def pitch_x_um(self) -> float:
        return self.geometry.pitch_x_um

    @property
    def pitch_y_um(self) -> float:
        return self.geometry.pitch_y_um

    @property
    def placement_label(self) -> str:
        """The persistent on-screen text for this plate's geometry: ``"stage"`` or ``"compact"``.

        Always available, never empty, and read off the RESOLVED mode -- so a viewer that shows it
        cannot show a word the cells do not deserve.
        """
        return placement_mode_label(self.placement_mode)

    def cell_center_um(self, cell_id: str) -> tuple[float, float]:
        """Stage micrometres of the cell's centre (A1's own centre for the top-left cell)."""
        r, c = self.cell_index(cell_id)
        return (self.geometry.a1_x_um + c * self.geometry.pitch_x_um,
                self.geometry.a1_y_um + r * self.geometry.pitch_y_um)

    @property
    def extent_um(self) -> tuple[float, float]:
        """(width, height) of the whole grid in micrometres, centre-span plus one cell."""
        return ((self.cols - 1) * self.geometry.pitch_x_um + self.geometry.cell_size_um,
                (self.rows - 1) * self.geometry.pitch_y_um + self.geometry.cell_size_um)

    def art(self, images_dir=None) -> Optional[CarrierArt]:
        """This plate's carrier PNG, or None when the artwork is not available.

        NOT part of the default render path any more (IMA-253): the viewer draws the holder from
        :meth:`cell_layout` + :class:`PlateGeometry` instead, in the one coordinate system the
        cells already live in, so it cannot mis-register. Kept because it is small, tested, and
        the obvious implementation of an optional photographic "skin".
        """
        return carrier_art(self.format_name, images_dir=images_dir, geometry=self.geometry)

    def cell_layout(self) -> Optional[dict[str, tuple[float, float, float, float]]]:
        """``{cell_id: (x, y, w, h)}`` in GRID UNITS (1.0 = one nominal cell), or None.

        ``None`` means "uniform": cell (r, c) occupies exactly ``(c, r, 1, 1)`` -- the only thing a
        well plate can mean, and the fast path the viewer keeps for it. A non-None layout is a
        FREEFORM holder whose cells are sized and positioned by real geometry; see the module
        docstring. Rectangles are guaranteed to lie inside ``(0, 0, cols, rows)``.
        """
        return None

    # -- occupancy ------------------------------------------------------------------------
    @property
    def occupied_cells(self) -> list[str]:
        """Acquired cells in plate row-major order -- what the viewer iterates and processes."""
        order = {cid: i for i, cid in enumerate(self.cell_ids)}
        return sorted(self._occupancy, key=lambda c: order[c])

    @property
    def occupied_map(self) -> dict[tuple[int, int], str]:
        """``{(row, col): cell_id}`` for acquired cells -- exactly PlateOverview's ``wells`` arg."""
        return {self.cell_index(c): c for c in self.occupied_cells}

    def is_occupied(self, cell_id: str) -> bool:
        return cell_id in self._occupancy

    def fovs(self, cell_id: str) -> list[int]:
        """FOV indices acquired in *cell_id*; ``[]`` for a real but unacquired cell."""
        if not self.has_cell(cell_id):
            raise KeyError(f"{cell_id!r} is not a cell of this {self.format_name}.")
        return list(self._occupancy.get(cell_id, ()))

    def viewer_grid(self) -> tuple[list[str], list[str], dict[tuple[int, int], str], list[str]]:
        """``(row_labels, col_labels, {(r, c): cell_id}, occupied_cells)`` -- PlateOverview's args.

        A single call so the viewer's ingest path is one block instead of four (format guard, well-id
        parse, full-vs-present grid choice, row-major sort). It also means a slide carrier reaches
        the SAME widget as a well plate: the overview only ever sees labels and a cell map, and has
        no idea whether a cell is a well or a slide.
        """
        return self.row_labels, self.col_labels, self.occupied_map, self.occupied_cells

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.format_name} {self.rows}x{self.cols} "
                f"occupied={len(self._occupancy)} source={self.format_source} "
                f"placement={self.placement_mode}>")


# --------------------------------------------------------------------------- WellPlate

def _row_letter(i: int) -> str:
    """0->A, 25->Z, 26->AA. Local copy: the viewer's lives behind a PyQt5 import."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _row_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        if not ch.isalpha():
            raise KeyError(letters)
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def display_well_id(cell_id: str) -> str:
    """Numeric well id from the alphabet-encoded well: ``<row letter's 1-based index><column>``.

    ``A1 -> "11"`` (A is row 1, column 1), ``C18 -> "318"`` (C is row 3, column 18). This is a
    DISPLAY-ONLY re-labelling that Julio asked for; it is deliberately NOT used as a data key. The
    real ``cell_id`` (``"A1"``, ``"C18"``) stays the reader's region key on disk, because the
    numeric form is lossy for >=10-column plates (``"318"`` could be row 3/col 18 or row 31/col 8),
    so it must never be parsed back. A cell id that is not the letter+column form (a tissue
    slide's freeform region like ``"manual0"``) is returned unchanged — there is no row letter to
    encode, and inventing one would be the "plain up bs" this is meant to avoid.
    """
    s = str(cell_id)
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    letters, digits = s[:i], s[i:]
    # A well row label is UPPERCASE and short (1 letter up to 26 rows, 2 up to 702 — a 1536wp is 32
    # rows = "AF"). A freeform region like "manual0"/"slot3" is lowercase or long, and passes
    # through untouched rather than being encoded into a meaningless number.
    if not letters or not digits.isdigit() or not letters.isupper() or len(letters) > 2:
        return s
    return f"{_row_index(letters) + 1}{digits}"


# --- fixed-width integer id: the flat-cache scope + the logger's numeric id -------------------
#
# SUPERSEDED, and left in place deliberately: :mod:`squidmip._address` is the successor to
# everything below. Read its docstring for the diagnosis; the short form is that this id does three
# jobs at once (cache key, logger id, navigator row), that two of its three fields are real
# acquisition coordinates while the third is the order somebody drew boxes, and that the ROI slot
# sits exactly where Squid puts a FIELD OF VIEW. Draw the same box twice and identical work is
# cached twice; delete ROI 2 and every later id shifts under whatever pointed at it.
#
# The logger no longer uses it (Task 1, 2026-07-29): a window logs an ``Address``/``Extent`` plus
# its view id. The CACHE still does, and migrating that key is Task 2/3 work. Do not add callers.
#
# Julio + Spencer (2026-07-24): enumerate with INTEGERS, not strings (a consultant warned string
# keys are silently slow, and a machine may transform them). Fixed-width slots -- Row Column ROI --
# make the id UNAMBIGUOUS and DECODABLE, unlike display_well_id's concatenated "318". Layout:
# ``row * 1_000_000 + col * 10_000 + roi`` with 0-based row/col and a 4-digit ROI slot. A 1536-well
# plate is 32 rows x 48 cols, so 2 digits each is ample; up to 10_000 ROIs per well.
_ROW_MUL = 1_000_000
_COL_MUL = 10_000
_ROI_MAX = 10_000


def well_code(cell_id: str) -> "Optional[int]":
    """The fixed-width integer id for a WELL (ROI slot 0), or None for a freeform region (no row
    letter, e.g. a tissue slide's ``manual0``). This is the flat cache SCOPE and the logger id."""
    s = str(cell_id)
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    letters, digits = s[:i], s[i:]
    if not letters or not digits.isdigit() or not letters.isupper() or len(letters) > 2:
        return None
    row = _row_index(letters)                 # 0-based row from the letter(s)
    col = int(digits) - 1                      # 0-based column
    if not (0 <= row <= 99 and 0 <= col <= 99):
        return None
    return row * _ROW_MUL + col * _COL_MUL


def roi_code(cell_id: str, roi_index: int) -> "Optional[int]":
    """The integer id for an ROI within a well: the well code plus the ROI slot (0.._ROI_MAX-1)."""
    base = well_code(cell_id)
    if base is None:
        return None
    r = int(roi_index)
    if not (0 <= r < _ROI_MAX):
        return None
    return base + r


def decode_code(code: int) -> "tuple[int, int, int]":
    """``code -> (row, col, roi)``, all 0-based. The inverse of :func:`well_code`/:func:`roi_code`,
    which is the whole point of the fixed-width layout: no string round-trip, no ambiguity."""
    code = int(code)
    return (code // _ROW_MUL, (code // _COL_MUL) % 100, code % _ROI_MAX)


def format_code(code: int) -> str:
    """Human form ``"RR CC OOOO"`` (Row Column ROI), e.g. ``"02 17 0003"`` -- the layout Julio drew."""
    row, col, roi = decode_code(code)
    return f"{row:02d} {col:02d} {roi:04d}"


def cache_scope(cell_id: str, roi_index: "Optional[int]" = None) -> str:
    """The flat-cache SCOPE string for a well or an ROI: the integer id when the region is a real
    well, else the raw region key (freeform slides have no row/col to encode). ``str`` so it drops
    straight into ``ResultCache`` keys."""
    code = roi_code(cell_id, roi_index) if roi_index is not None else well_code(cell_id)
    return str(code) if code is not None else str(cell_id)


class WellPlate(Plate):
    """A standard microtitre plate: cells are wells named A1..{row}{col}, 1-based columns."""

    @classmethod
    def from_format(cls, format_name, occupancy=None, **kw) -> "WellPlate":
        return cls(PlateGeometry.vendored(format_name), occupancy, **kw)

    @property
    def row_labels(self) -> list[str]:
        return [_row_letter(i) for i in range(self.rows)]

    @property
    def col_labels(self) -> list[str]:
        return [str(c) for c in range(1, self.cols + 1)]

    def cell_id(self, row: int, col: int) -> str:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise KeyError((row, col))
        return f"{_row_letter(row)}{col + 1}"

    def cell_index(self, cell_id: str) -> tuple[int, int]:
        s = str(cell_id)
        i = 0
        while i < len(s) and s[i].isalpha():
            i += 1
        if i == 0 or i == len(s) or not s[i:].isdigit():
            raise KeyError(cell_id)
        row, col = _row_index(s[:i]), int(s[i:]) - 1
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise KeyError(cell_id)
        return row, col


# --------------------------------------------------------------------------- SlideCarrier

class SlideCarrier(Plate):
    """A slide holder: a 1 x N grid whose cells are named by the acquisition's own region ids.

    This is the freeform / tissue case, and it is a first-class Plate rather than a 1x1 wellplate
    hack: the real dataset has TWO regions (``manual0``, ``manual1``) on one holder, and collapsing
    them into a single degenerate cell would stack both mosaics on top of each other. Slots with no
    region keep a synthetic ``slot{n}`` id so the grid stays rectangular and drawable.
    """

    _FILLER_PREFIX = "slot"

    def __init__(self, geometry, occupancy=None, cell_ids: Optional[Iterable[str]] = None,
                 placement: Optional[Mapping[str, tuple]] = None,
                 layout: Optional[Mapping[str, tuple]] = None,
                 stage_boxes_um: Optional[Mapping[str, tuple]] = None, **kw):
        """*placement* / *layout* carry the GEOMETRIC assignment (IMA-253).

        ``placement`` is ``{region: (row, col)}`` derived from stage coordinates by
        :func:`freeform_grid`; ``layout`` is the matching ``{region: (x, y, w, h)}`` in grid units
        from :func:`freeform_layout`. Both absent is the legacy POSITIONAL carrier -- regions left
        to right in report order -- which is still exactly right when the acquisition has no stage
        coordinates at all, and is the only thing that can be done then.

        ``stage_boxes_um`` is ``{region: (x, y, w, h)}`` in stage MICROMETRES -- the raw
        measurement ``layout`` was normalised from. It is retained (not just its normalised form)
        because the SLIDE renderer (:mod:`squidmip._slide_art`) needs the true micron scale to draw
        a 25 x 75 mm slide at the right size relative to the tissue on it; the normalised ``layout``
        has already divided that scale out.
        """
        names = list(cell_ids) if cell_ids is not None else []
        n_slots = geometry.rows * geometry.cols
        if len(names) > n_slots:
            raise PlateBuildError(
                f"{len(names)} regions {names[:6]} do not fit a {geometry.name} "
                f"({n_slots} slot(s)). Force a larger carrier, or check the region ids."
            )
        if placement:
            slots: list[Optional[str]] = [None] * n_slots
            for cid, (r, c) in placement.items():
                slots[int(r) * geometry.cols + int(c)] = str(cid)
            self._ids = [s if s is not None else f"slot{i + 1}" for i, s in enumerate(slots)]
        else:
            # Positional assignment, left to right, in the order the acquisition reports them.
            self._ids = names + [f"slot{i + 1}" for i in range(len(names), n_slots)]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        self._layout = {str(k): tuple(float(v) for v in val)
                        for k, val in (layout or {}).items()} or None
        self.stage_boxes_um = {str(k): tuple(float(v) for v in val)
                               for k, val in (stage_boxes_um or {}).items()}
        super().__init__(geometry, occupancy, **kw)

    @classmethod
    def from_format(cls, format_name, occupancy=None, cell_ids=None, **kw) -> "SlideCarrier":
        return cls(PlateGeometry.vendored(format_name), occupancy, cell_ids=cell_ids, **kw)

    def cell_layout(self):
        return dict(self._layout) if self._layout else None

    @property
    def row_labels(self) -> list[str]:
        # A 1 x N carrier keeps its blank row label (the columns carry the ids, as they always
        # have). A geometrically stacked carrier has one region PER ROW, so the row is where the
        # name belongs -- and an ambiguous row is left blank rather than labelled with a guess.
        if self.rows == 1:
            return [""]
        return [self._sole_cell(0, r) or "" for r in range(self.rows)]

    @property
    def col_labels(self) -> list[str]:
        # The region ids themselves ARE the useful column labels on a carrier -- when a column
        # holds exactly one of them. Otherwise fall back to the slot number.
        return [self._sole_cell(1, c) or str(c + 1) for c in range(self.cols)]

    def cell_id(self, row: int, col: int) -> str:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise KeyError((row, col))
        return self._ids[row * self.cols + col]

    def cell_index(self, cell_id: str) -> tuple[int, int]:
        i = self._pos.get(str(cell_id))
        if i is None:
            raise KeyError(cell_id)
        return divmod(i, self.cols)


# --------------------------------------------------------------------------- CompactPlate

class CompactPlate(Plate):
    """The OCCUPIED regions of any format, packed into an even grid: ``placement="compact"``.

    A browse layout, and it says so. The cells are the regions the acquisition actually visited,
    at equal size, in the reading order the STAGE gives them; the information-free space between
    them is gone. Three wells of a 384-well plate stop being three dots in a 16 x 24 field of
    nothing.

    Not a new algorithm: the packing is :func:`even_carrier_layout`, shipped on 2026-07-23 for
    tissue carriers, applied to every format. Not a :class:`SlideCarrier` either, however similar
    the mechanism, because a compacted well plate is not a slide holder and the naming law forbids
    reusing a physical word for a software concept -- ``compact`` is ours, ``slide`` is Squid's.

    What is LOST, deliberately and visibly: the row/column topology. A compact grid's rows and
    columns are packing indices, not plate rows and columns, so A1 is not guaranteed above A2. The
    real ``cell_id`` survives on every cell, :attr:`placement_mode` reads ``compact``, and
    :meth:`cell_center_um` refuses rather than fabricating a stage position.

    Cells past the last region keep a synthetic ``pad{n}`` id so the grid stays rectangular
    ("pad" because Squid will never call anything that; ``slot`` is the carrier's word for a real
    physical bay and must not be borrowed for a blank).
    """

    _FILLER_PREFIX = "pad"

    def __init__(self, geometry, occupancy=None, placement: Optional[Mapping[str, tuple]] = None,
                 layout: Optional[Mapping[str, tuple]] = None,
                 stage_boxes_um: Optional[Mapping[str, tuple]] = None, **kw):
        n_slots = geometry.rows * geometry.cols
        slots: list[Optional[str]] = [None] * n_slots
        for cid, (r, c) in (placement or {}).items():
            i = int(r) * geometry.cols + int(c)
            if not (0 <= i < n_slots) or slots[i] is not None:
                raise PlateBuildError(
                    f"compact placement puts {cid!r} at cell ({r}, {c}) of a "
                    f"{geometry.rows} x {geometry.cols} grid, which is out of range or already "
                    f"taken by {slots[i] if 0 <= i < n_slots else '?'!r}. Two regions in one cell "
                    "would draw one on top of the other."
                )
            slots[i] = str(cid)
        self._ids = [s if s is not None else f"{self._FILLER_PREFIX}{i}"
                     for i, s in enumerate(slots)]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        self._layout = {str(k): tuple(float(v) for v in val)
                        for k, val in (layout or {}).items()} or None
        #: raw stage boxes of the regions, retained so a consumer that needs true micron scale can
        #: still get it. The compact CELLS are not at these positions; that is the point of them.
        self.stage_boxes_um = {str(k): tuple(float(v) for v in val)
                               for k, val in (stage_boxes_um or {}).items()}
        kw.setdefault("placement_mode", COMPACT)
        super().__init__(geometry, occupancy, **kw)

    def cell_layout(self):
        return dict(self._layout) if self._layout else None

    @property
    def row_labels(self) -> list[str]:
        if self.rows == 1:
            return [""]
        return [self._sole_cell(0, r) or "" for r in range(self.rows)]

    @property
    def col_labels(self) -> list[str]:
        return [self._sole_cell(1, c) or "" for c in range(self.cols)]

    def cell_id(self, row: int, col: int) -> str:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise KeyError((row, col))
        return self._ids[row * self.cols + col]

    def cell_index(self, cell_id: str) -> tuple[int, int]:
        i = self._pos.get(str(cell_id))
        if i is None:
            raise KeyError(cell_id)
        return divmod(i, self.cols)

    def cell_center_um(self, cell_id: str) -> tuple[float, float]:
        """Refused: a compact cell is not at a stage position.

        The base implementation computes ``a1 + index * pitch``, which on this plate would be a
        plausible-looking micrometre pair for a place the stage never was. Everything else in this
        codebase raises rather than answer that question wrongly, so this does too, and names the
        way out.
        """
        raise PlateBuildError(
            f"{cell_id!r} is a cell of a COMPACT plate, whose cells are packed for browsing and "
            "are not at stage positions. There is no stage centre to return. Rebuild with "
            f"placement={STAGE!r} to ask this question."
        )


# --------------------------------------------------------------------------- freeform geometry

def region_stage_boxes_um(metadata: Mapping, regions: Optional[Iterable[str]] = None
                          ) -> dict[str, tuple[float, float, float, float]]:
    """``{region: (x_um, y_um, w_um, h_um)}`` -- each region's MOSAIC bounding box on the stage.

    The box is the union of every FOV's footprint, so it is the region's real extent: FOV top-left
    positions span ``max - min``, and one frame's width/height is added because a position marks a
    corner, not a point. That makes it exactly ``_placement.mosaic_extent_px`` expressed in
    micrometres instead of pixels -- the same geometry the mosaic itself is composited from, which
    is what lets the viewer draw a region's cell and its FOVs in one coordinate system.

    Regions with no recorded position are OMITTED rather than given a zero box; a caller that
    cannot place every region must fall back to a positional layout instead of dropping one.
    """
    positions = metadata.get("fov_positions_um") or {}
    fovs_per_region = metadata.get("fovs_per_region") or {}
    scoped = list(metadata.get("regions") or []) if regions is None else list(regions)
    # A stage position marks the frame's corner, so the region spans one extra frame past the
    # last one. No pixel size / frame shape -> add nothing: the RELATIVE placement is still right,
    # only the pad is missing, and that beats refusing to lay the acquisition out at all.
    frame = metadata.get("frame_shape") or (0, 0)
    p = metadata.get("pixel_size_um") or 0.0
    fh_um, fw_um = float(frame[0]) * float(p), float(frame[1]) * float(p)

    out: dict[str, tuple[float, float, float, float]] = {}
    for region in scoped:
        pts = [positions[(region, f)] for f in (fovs_per_region.get(region) or ())
               if (region, f) in positions]
        if not pts:
            continue
        xs = [float(q[0]) for q in pts]
        ys = [float(q[1]) for q in pts]
        out[region] = (min(xs), min(ys),
                       max(xs) - min(xs) + fw_um, max(ys) - min(ys) + fh_um)
    return out


def freeform_grid(boxes_um: Mapping[str, tuple]) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """``(rows, cols, {region: (row, col)})`` clustered from stage boxes -- never from names.

    Regions whose y-intervals overlap share a ROW; within a row they are ordered by x. That is the
    minimum structure the grid-shaped parts of the viewer (labels, ``(row, col)`` keys, marquee)
    still need, and it is derived entirely from geometry, so re-ordering or renaming the regions
    produces the identical grid. Exact rectangles -- the part that actually matters visually --
    come from :func:`freeform_layout`; this only decides which cell KEY each region gets.

    Two regions overlap in y when their intervals share more than half of the shorter one, so a
    sliver of overlap between two clearly-stacked tissues does not fuse them into one row.
    """
    order = sorted(boxes_um, key=lambda r: (boxes_um[r][1], boxes_um[r][0], r))
    rows: list[list[str]] = []
    spans: list[tuple[float, float]] = []
    for region in order:
        _x, y, _w, h = boxes_um[region]
        y0, y1 = y, y + h
        for i, (s0, s1) in enumerate(spans):
            overlap = min(y1, s1) - max(y0, s0)
            shorter = min(y1 - y0, s1 - s0)
            if overlap > 0 and (shorter <= 0 or overlap > 0.5 * shorter):
                rows[i].append(region)
                spans[i] = (min(s0, y0), max(s1, y1))
                break
        else:
            rows.append([region])
            spans.append((y0, y1))
    placement: dict[str, tuple[int, int]] = {}
    for ri, members in enumerate(rows):
        for ci, region in enumerate(sorted(members, key=lambda r: (boxes_um[r][0], r))):
            placement[region] = (ri, ci)
    n_rows = max(1, len(rows))
    n_cols = max(1, max((len(m) for m in rows), default=1))
    return n_rows, n_cols, placement


def even_carrier_layout(
    regions: Sequence[str],
    order_key: Optional[Mapping[str, tuple]] = None,
    target_aspect: float = 2.4,
    gap: float = 0.14,
) -> tuple[int, int, dict[str, tuple[int, int]], dict[str, tuple[float, float, float, float]]]:
    """A tissue carrier laid out EVENLY, not by raw stage geometry (Julio, 2026-07-23).

    Returns ``(rows, cols, placement, layout)``. Regions are placed left-to-right, top-to-bottom in
    a LANDSCAPE-biased grid (like the physical 4-slide carrier, which is a horizontal row of
    slides), and every region gets an EQUAL cell with the same inset gap — so mosaics are evenly
    spaced and never overlap even when their native geometries differ wildly. This deliberately
    REPLACES ``freeform_grid``/``freeform_layout`` for the carrier: those preserve true relative
    size and position, which stacked two tissues into a tall, tiny, uneven column and wasted the
    viewer's horizontal space. Even, readable cells beat geometric fidelity for a browse view.

    ``order_key`` (region -> a sortable tuple, e.g. its stage box) orders the cells so spatially
    left tissue lands on the left; absent, region report order is used.
    """
    import math

    regs = list(regions)
    if order_key is not None:
        regs = sorted(regs, key=lambda r: (order_key.get(r, (0.0, 0.0))[0],
                                           order_key.get(r, (0.0, 0.0))[1], r))
    n = max(1, len(regs))
    cols = max(1, min(n, math.ceil(math.sqrt(n * target_aspect))))   # bias WIDE (landscape)
    rows = max(1, math.ceil(n / cols))
    placement: dict[str, tuple[int, int]] = {}
    layout: dict[str, tuple[float, float, float, float]] = {}
    for i, r in enumerate(regs):
        row, col = divmod(i, cols)
        placement[r] = (row, col)
        layout[r] = (col + gap, row + gap, 1.0 - 2 * gap, 1.0 - 2 * gap)   # equal, inset cell
    return rows, cols, placement, layout


def freeform_layout(boxes_um: Mapping[str, tuple], rows: int, cols: int
                    ) -> dict[str, tuple[float, float, float, float]]:
    """Stage boxes -> ``{region: (x, y, w, h)}`` in GRID UNITS, aspect preserved and centred.

    ONE similarity transform is applied to every box, so relative offset AND relative scale
    survive: a region twice the size of its neighbour gets a cell twice the size, and two regions
    5 mm apart stay 5 mm apart in proportion. Fitting the union into the ``cols x rows`` box the
    grid already declares means the result drops straight into the viewer's existing
    cell-units-times-zoom arithmetic, with no second coordinate system to keep in sync.
    """
    if not boxes_um:
        return {}
    x0 = min(b[0] for b in boxes_um.values())
    y0 = min(b[1] for b in boxes_um.values())
    x1 = max(b[0] + b[2] for b in boxes_um.values())
    y1 = max(b[1] + b[3] for b in boxes_um.values())
    uw, uh = x1 - x0, y1 - y0
    if not (uw > 0 and uh > 0):
        return {}          # degenerate (points, or a single axis): no scale to preserve, so the
        #                    caller keeps the nominal grid rather than dividing by ~zero
    s = min(cols / uw, rows / uh)
    ox, oy = (cols - uw * s) / 2.0, (rows - uh * s) / 2.0
    return {r: (ox + (b[0] - x0) * s, oy + (b[1] - y0) * s, b[2] * s, b[3] * s)
            for r, b in boxes_um.items()}


# --------------------------------------------------------------------------- measurement

def measure_region_pitch_um(positions_um: Mapping[tuple, tuple], regions: Iterable[str]
                            ) -> tuple[Optional[float], Optional[float]]:
    """Centre-to-centre well pitch (x_um, y_um) MEASURED from stage coordinates, or (None, None).

    Each region's anchor is its top-left FOV, which is stable regardless of how many FOVs the well
    holds (a centroid is not: it shifts when wells have different FOV counts). The pitch is the
    median of ``|dx| / |dcol|`` over every pair sharing a row, and likewise for rows -- so a plate
    scanned at A1 and A5 only still measures the true pitch instead of a 4x-too-large one.

    Returns None per axis whenever that axis cannot be measured: fewer than two distinct
    rows/columns, no coordinates at all, or region ids that are not well ids (a slide carrier).

    GROUPED, NOT SCANNED, in both halves — the only cost in this module that grew quadratically
    with plate size. Measured on a full 1536-well plate: 180 ms before, 9.6 ms after. Two loops
    were doing it. The first re-scanned every stage position once per region (1536 x 1536); it now
    buckets the positions by region in one pass. The second compared every well against every
    other well and discarded the ~97% that did not share the axis's index; it now buckets the
    wells by that index first, so only pairs that CAN contribute are formed. Neither changes what
    is measured: the same set of pairs reaches ``deltas``, and the median of a multiset does not
    depend on the order it was built in.
    """
    by_region: dict[str, list] = {}
    for key, value in positions_um.items():
        by_region.setdefault(key[0], []).append(value)

    anchors: dict[str, tuple[float, float]] = {}
    index: dict[str, tuple[int, int]] = {}
    for region in regions:
        pts = by_region.get(region)
        if not pts:
            continue
        span = well_span([region])
        if span is None:                       # freeform id -> not a well grid, nothing to measure
            return None, None
        r, c = span[0] - 1, span[1] - 1
        anchors[region] = (min(p[0] for p in pts), min(p[1] for p in pts))
        index[region] = (r, c)
    if len(anchors) < 2:
        return None, None

    def _axis(shared: int, varying: int, coord: int) -> Optional[float]:
        lanes: dict[int, list] = {}
        for name in anchors:
            lanes.setdefault(index[name][shared], []).append(name)
        deltas = []
        for names in lanes.values():
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    d_idx = index[b][varying] - index[a][varying]
                    if d_idx == 0:
                        continue
                    deltas.append(abs(anchors[b][coord] - anchors[a][coord]) / abs(d_idx))
        if not deltas:
            return None
        deltas.sort()
        return deltas[len(deltas) // 2]

    return _axis(0, 1, 0), _axis(1, 0, 1)      # x: same row, varying column; y: same column


def format_from_pitch_um(pitch_x_um: Optional[float], pitch_y_um: Optional[float],
                         tolerance: float = _PITCH_TOL) -> Optional[str]:
    """The standard wellplate format whose pitch matches, or None when the match is not unambiguous.

    Both axes must land on the SAME format (a real plate is square-pitched), so a measurement where
    x reads 96wp and y reads 384wp is refused rather than resolved by picking one -- that would be
    inventing a plate. A single measurable axis is accepted, since a one-row scan still pins the
    pitch and the pitches of all Squid formats are distinct by more than the tolerance.
    """
    def _match(p):
        if p is None or p <= 0:
            return None
        hits = [name for name, g in ((n, PlateGeometry.vendored(n)) for n in _VENDORED_MM)
                if g.pitch_x_um > 0 and abs(p - g.pitch_x_um) <= tolerance * g.pitch_x_um]
        return hits[0] if len(hits) == 1 else None

    mx, my = _match(pitch_x_um), _match(pitch_y_um)
    if mx and my:
        return mx if mx == my else None
    return mx or my


# --------------------------------------------------------------------------- the builder

def build_plate(metadata: Mapping, override=None, images_dir=None,
                placement: str = DEFAULT_PLACEMENT_MODE) -> Plate:
    """Build the :class:`Plate` an acquisition describes. The one entry point callers need.

    Precedence (see the module docstring): ``override > measured > declared > inferred-from-span``.
    A measured format that CONTRADICTS the declared one wins and warns loudly; it is ignored (with
    a warning) when it cannot contain every observed well, because then the measurement, not the
    declaration, is the thing that must be wrong.

    Freeform region ids (``manual0``) produce a :class:`SlideCarrier`, never a degenerate
    wellplate -- that is what lets a glass-slide/tissue acquisition open at all.

    *placement* is ``"stage"`` (the default, and byte-identical to every plate this function has
    ever built) or ``"compact"``, which closes the space between regions and returns a
    :class:`CompactPlate`. Format resolution is entirely unaffected: the mode decides where cells
    are DRAWN, never which holder the acquisition is on, so a compact plate still measures its
    pitch and still warns about a mis-declared format. Anything other than those two strings
    raises; see :func:`squidmip._placement.normalize_placement_mode` for why it must not default.
    """
    placement = normalize_placement_mode(placement)
    regions = list(metadata.get("regions") or [])
    fovs_per_region = dict(metadata.get("fovs_per_region") or {})
    positions_um = metadata.get("fov_positions_um") or {}
    declared_raw = metadata.get("wellplate_format")

    # Stage boxes for the freeform path (IMA-253). Computed once, here, because it is the only
    # place that still holds the metadata; every _make below is handed the same measurement.
    stage_boxes = region_stage_boxes_um(metadata, regions) if regions else {}
    if len(stage_boxes) != len(regions):
        stage_boxes = {}                   # cannot place EVERY region -> place none by geometry

    forced = normalize_plate_format(
        override if override is not None else os.environ.get("SQUIDMIP_WELLPLATE_FORMAT"),
        strict=False,
    )
    if override is not None or forced:
        name = _canonical_format(override if override is not None else forced)
        return _make(name, regions, fovs_per_region, stage_boxes, placement_mode=placement,
                     format_source="override", declared_format=_safe_canonical(declared_raw))

    declared = _safe_canonical(declared_raw)
    measured_pitch = measure_region_pitch_um(positions_um, regions)
    measured = format_from_pitch_um(*measured_pitch)

    if measured and measured != declared:
        span = well_span(regions)
        fits = span is None or (
            span[0] <= PlateGeometry.vendored(measured).rows
            and span[1] <= PlateGeometry.vendored(measured).cols
        )
        px, py = measured_pitch
        pitch_txt = f"x={px:.1f} um, y={py:.1f} um" if px and py else f"{px or py:.1f} um"
        if not fits:
            warnings.warn(
                f"stage coordinates measure a {pitch_txt} pitch, which reads as {measured!r}, but "
                f"the wells span {span[0]}x{span[1]} and do not fit it. Keeping the declared "
                f"{declared!r}; the measurement is being ignored. Check coordinates.csv."
            )
        elif declared:
            warnings.warn(
                f"declared wellplate_format {declared!r} contradicts the stage coordinates, which "
                f"measure a {pitch_txt} pitch -- physically {measured!r}. Using the MEASURED "
                f"{measured!r}: trusting the declaration would lay the plate out at "
                f"{PlateGeometry.vendored(declared).pitch_x_um / PlateGeometry.vendored(measured).pitch_x_um:.3g}x "
                f"the true scale. Override with SQUIDMIP_WELLPLATE_FORMAT if the yaml is right."
            )
            return _make(measured, regions, fovs_per_region, stage_boxes, placement_mode=placement,
                         format_source="measured",
                         declared_format=declared, measured_pitch_um=measured_pitch)
        else:
            return _make(measured, regions, fovs_per_region, stage_boxes, placement_mode=placement,
                         format_source="measured",
                         declared_format=None, measured_pitch_um=measured_pitch)

    if declared:
        return _make(declared, regions, fovs_per_region, stage_boxes, placement_mode=placement,
                     format_source="declared",
                     declared_format=declared, measured_pitch_um=measured_pitch)
    if measured:
        return _make(measured, regions, fovs_per_region, stage_boxes, placement_mode=placement,
                     format_source="measured",
                     declared_format=None, measured_pitch_um=measured_pitch)
    return _make(infer_plate_format(regions), regions, fovs_per_region, stage_boxes,
                 placement_mode=placement, format_source="inferred", declared_format=None)


def _safe_canonical(name) -> Optional[str]:
    try:
        return _canonical_format(name) if name else None
    except PlateShapeError:
        return None


def _compact_order_key(regions, stage_boxes) -> dict[str, tuple[float, float]]:
    """``{region: (y, x)}`` -- the sort key that gives a compact grid its READING order.

    ``(y, x)``, not ``(x, y)``: a compact plate is filled row-major, so the outer key has to be the
    axis that chooses the row. That reproduces ``occupied_cells`` order (top-left first, then left
    to right along a row), and inside a row lower stage x still renders left, which is the property
    that makes a compact plate comparable with the stage one at a glance.

    Stage boxes are the truth when they exist. When they do not -- a plate WE wrote carries no
    coordinates.csv -- a well id encodes its own position, so ``(row, col)`` from the id IS stage
    geometry rather than a fallback to enumeration order. A freeform region with neither is placed
    at the origin, which is the only honest answer and leaves the id as the tie-break.
    """
    key: dict[str, tuple[float, float]] = {}
    for region in regions:
        box = (stage_boxes or {}).get(region)
        if box is not None:
            key[region] = (float(box[1]), float(box[0]))       # (y_um, x_um)
            continue
        span = well_span([region])                              # "B3" -> (2, 3), 1-based
        key[region] = (float(span[0]), float(span[1])) if span else (0.0, 0.0)
    return key


def _make(name, regions, fovs_per_region, stage_boxes=None,
          placement_mode=DEFAULT_PLACEMENT_MODE, **kw) -> Plate:
    """Instantiate the right subclass for *name*, sizing a slide carrier to the regions present.

    *placement_mode* is the REQUESTED mode. Which class results, and what mode that class ends up
    reporting, is decided here and nowhere else:

    ``compact`` on any well plate
        a :class:`CompactPlate` of the occupied wells, packed by :func:`even_carrier_layout`.
    ``stage`` on any well plate
        the :class:`WellPlate` this function has always built, untouched.
    either mode on a FREEFORM carrier
        the same even :class:`SlideCarrier`, reporting ``compact``, because the stage-proportional
        carrier layout was removed from the product on 2026-07-23 and no longer exists to return.
        See the module docstring; this is the one place requested and resolved diverge.
    """
    occupancy = {r: list(fovs_per_region.get(r, ())) for r in regions}
    if name in _SLIDE_FORMATS or well_span(regions) is None:
        # SPAN+SNAP for carriers, mirroring _plate_shape's rule for plates: pick the smallest
        # standard holder with room for every region. A 2-region tissue slide is a 4-up carrier,
        # because that is the holder it physically sat in; 1 region on a declared glass slide is
        # the single-slide holder.
        n = max(1, len(regions))
        if name not in _SLIDE_FORMATS:
            name = GLASS_SLIDE                     # freeform ids under a plate format name
        if n > 1 and name == GLASS_SLIDE:
            name = FOUR_SLIDE_CARRIER
        geom = PlateGeometry.vendored(name)
        # EVEN layout (2026-07-23): a tissue carrier is drawn as evenly-spaced, equal, landscape-
        # biased cells — NOT by raw stage geometry. Julio: the stage-proportional freeform layout
        # stacked two tissues into a tall, tiny, uneven column and wasted the viewer's horizontal
        # space; even cells that never overlap read far better for browsing, whatever each mosaic's
        # native geometry. Stage boxes, when present, only ORDER the cells left-to-right so spatial
        # left tissue stays on the left. (freeform_grid/freeform_layout are kept for reference.)
        placement = layout = None
        if n >= 1:
            rows, cols, placement, layout = even_carrier_layout(list(regions), order_key=stage_boxes)
            geom = PlateGeometry(**{**vars(geom), "rows": rows, "cols": cols})
        if placement is None and n > geom.rows * geom.cols:
            # More slides than any standard carrier: widen rather than refuse. There is no art for
            # this, and carrier_art() will correctly return None instead of a wrong-scale PNG.
            geom = PlateGeometry(**{**vars(geom), "cols": n})
        # The carrier's cells are EVEN in both modes, so it reports the mode it actually has and
        # remembers what was asked. A carrier labelled "stage" would be claiming a proportionality
        # that 2b8fbc5 deliberately removed.
        carrier_mode = COMPACT if placement else placement_mode
        return SlideCarrier(geom, occupancy, cell_ids=list(regions), placement=placement,
                            layout=layout, stage_boxes_um=stage_boxes,
                            format_name=geom.name, placement_mode=carrier_mode,
                            placement_requested=placement_mode, **kw)

    if placement_mode == COMPACT and regions:
        # COMPACT well plate: only the wells that were VISITED, packed evenly, ordered by the
        # stage. `even_carrier_layout` is the shipped tissue-carrier packing, unchanged and
        # unconditioned on format -- promoting it is the whole of this mode.
        base = PlateGeometry.vendored(name)
        rows, cols, cells, layout = even_carrier_layout(
            list(regions), order_key=_compact_order_key(regions, stage_boxes))
        geom = PlateGeometry(**{**vars(base), "rows": rows, "cols": cols})
        return CompactPlate(geom, occupancy, placement=cells, layout=layout,
                            stage_boxes_um=stage_boxes, format_name=name,
                            placement_requested=placement_mode, **kw)

    return WellPlate(PlateGeometry.vendored(name), occupancy, format_name=name,
                     placement_mode=placement_mode, **kw)
