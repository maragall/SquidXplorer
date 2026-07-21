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

from ._plate_shape import (
    GLASS_SLIDE,
    PlateShapeError,
    infer_plate_format,
    normalize_plate_format,
    well_span,
)

__all__ = [
    "CarrierArt",
    "Plate",
    "PlateBuildError",
    "PlateGeometry",
    "SlideCarrier",
    "WellPlate",
    "build_plate",
    "carrier_art",
    "format_from_pitch_um",
    "load_sample_formats",
    "measure_region_pitch_um",
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
    ):
        self.geometry = geometry
        self.format_name = format_name or geometry.name
        self.declared_format = declared_format
        #: how ``format_name`` was decided: "override" | "measured" | "declared" | "inferred".
        self.format_source = format_source
        self.measured_pitch_um = measured_pitch_um
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

    # -- physical -------------------------------------------------------------------------
    @property
    def pitch_x_um(self) -> float:
        return self.geometry.pitch_x_um

    @property
    def pitch_y_um(self) -> float:
        return self.geometry.pitch_y_um

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
        """This plate's carrier PNG, or None when the artwork is not available."""
        return carrier_art(self.format_name, images_dir=images_dir, geometry=self.geometry)

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
                f"occupied={len(self._occupancy)} source={self.format_source}>")


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

    def __init__(self, geometry, occupancy=None, cell_ids: Optional[Iterable[str]] = None, **kw):
        names = list(cell_ids) if cell_ids is not None else []
        n_slots = geometry.rows * geometry.cols
        if len(names) > n_slots:
            raise PlateBuildError(
                f"{len(names)} regions {names[:6]} do not fit a {geometry.name} "
                f"({n_slots} slot(s)). Force a larger carrier, or check the region ids."
            )
        # Positional assignment, left to right, in the order the acquisition reports them.
        self._ids = names + [f"slot{i + 1}" for i in range(len(names), n_slots)]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        super().__init__(geometry, occupancy, **kw)

    @classmethod
    def from_format(cls, format_name, occupancy=None, cell_ids=None, **kw) -> "SlideCarrier":
        return cls(PlateGeometry.vendored(format_name), occupancy, cell_ids=cell_ids, **kw)

    @property
    def row_labels(self) -> list[str]:
        # One row, and it needs no name: the columns carry the region ids.
        return [""] * self.rows

    @property
    def col_labels(self) -> list[str]:
        # The region ids themselves ARE the useful column labels on a carrier.
        return [self._ids[c] for c in range(self.cols)]

    def cell_id(self, row: int, col: int) -> str:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise KeyError((row, col))
        return self._ids[row * self.cols + col]

    def cell_index(self, cell_id: str) -> tuple[int, int]:
        i = self._pos.get(str(cell_id))
        if i is None:
            raise KeyError(cell_id)
        return divmod(i, self.cols)


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
    """
    anchors: dict[str, tuple[float, float]] = {}
    index: dict[str, tuple[int, int]] = {}
    for region in regions:
        pts = [v for k, v in positions_um.items() if k[0] == region]
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
        deltas = []
        names = list(anchors)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if index[a][shared] != index[b][shared]:
                    continue
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

def build_plate(metadata: Mapping, override=None, images_dir=None) -> Plate:
    """Build the :class:`Plate` an acquisition describes. The one entry point callers need.

    Precedence (see the module docstring): ``override > measured > declared > inferred-from-span``.
    A measured format that CONTRADICTS the declared one wins and warns loudly; it is ignored (with
    a warning) when it cannot contain every observed well, because then the measurement, not the
    declaration, is the thing that must be wrong.

    Freeform region ids (``manual0``) produce a :class:`SlideCarrier`, never a degenerate
    wellplate -- that is what lets a glass-slide/tissue acquisition open at all.
    """
    regions = list(metadata.get("regions") or [])
    fovs_per_region = dict(metadata.get("fovs_per_region") or {})
    positions_um = metadata.get("fov_positions_um") or {}
    declared_raw = metadata.get("wellplate_format")

    forced = normalize_plate_format(
        override if override is not None else os.environ.get("SQUIDMIP_WELLPLATE_FORMAT"),
        strict=False,
    )
    if override is not None or forced:
        name = _canonical_format(override if override is not None else forced)
        return _make(name, regions, fovs_per_region,
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
            return _make(measured, regions, fovs_per_region, format_source="measured",
                         declared_format=declared, measured_pitch_um=measured_pitch)
        else:
            return _make(measured, regions, fovs_per_region, format_source="measured",
                         declared_format=None, measured_pitch_um=measured_pitch)

    if declared:
        return _make(declared, regions, fovs_per_region, format_source="declared",
                     declared_format=declared, measured_pitch_um=measured_pitch)
    if measured:
        return _make(measured, regions, fovs_per_region, format_source="measured",
                     declared_format=None, measured_pitch_um=measured_pitch)
    return _make(infer_plate_format(regions), regions, fovs_per_region,
                 format_source="inferred", declared_format=None)


def _safe_canonical(name) -> Optional[str]:
    try:
        return _canonical_format(name) if name else None
    except PlateShapeError:
        return None


def _make(name, regions, fovs_per_region, **kw) -> Plate:
    """Instantiate the right subclass for *name*, sizing a slide carrier to the regions present."""
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
        if n > geom.rows * geom.cols:
            # More slides than any standard carrier: widen rather than refuse. There is no art for
            # this, and carrier_art() will correctly return None instead of a wrong-scale PNG.
            geom = PlateGeometry(**{**vars(geom), "cols": n})
        return SlideCarrier(geom, occupancy, cell_ids=list(regions),
                            format_name=geom.name, **kw)
    return WellPlate(PlateGeometry.vendored(name), occupancy, format_name=name, **kw)
