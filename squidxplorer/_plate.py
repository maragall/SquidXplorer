"""The sample holder as one model: a grid of cells, each holding 0..N FOVs.

Plate (ABC) holds grid + geometry + occupancy; WellPlate names cells A1..H12, SlideCarrier
names them by the acquisition's own freeform region ids. All geometry is micrometres.
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
    _row_index,
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

# Carrier formats that are a grid of slides, not wells.
FOUR_SLIDE_CARRIER = "4 slide carrier"
_SLIDE_FORMATS = (GLASS_SLIDE, FOUR_SLIDE_CARRIER)

_SQUID_ENV = "SQUIDXPLORER_SQUID_SOFTWARE"
_SQUID_GUESSES = (
    Path.home() / "Cephla" / "projects" / "Squid" / "software",
    Path.home() / "Squid" / "software",
    Path.home() / "projects" / "Squid" / "software",
)

# Squid's sample_formats.csv, vendored verbatim (mm, as upstream writes it).
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

# Not in sample_formats.csv; upstream hardcodes it in the GUI. Slot pitch/size are a layout
# approximation for a standard 75 x 25 mm slide, only used to place cells on the art.
_VENDORED_MM[FOUR_SLIDE_CARRIER] = (14.0, 20.0, 50, 0, 25.0, 27.0, 0, 1, 4)

# Pitch matching candidates: wellplates only. The 4-up carrier's slot pitch is within tolerance
# of a 12-well plate's, so including slide holders makes every 12wp match ambiguous.
_WELLPLATE_FORMATS = tuple(n for n in _VENDORED_MM if n not in _SLIDE_FORMATS)

# Carrier art: format -> (png filename, mm_per_pixel), copied from Squid's image_paths dict.
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

# Fractional slack when matching a measured pitch to a standard one.
_PITCH_TOL = 0.05


class PlateBuildError(ValueError):
    """The acquisition cannot be expressed as a plate (regions outside the grid, too many slides)."""


# --------------------------------------------------------------------------- geometry

@dataclass(frozen=True)
class PlateGeometry:
    """Physical layout of one sample format, in micrometres; ``_px`` values are art pixels."""

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
    """Canonical format name, extending ``normalize_plate_format`` with the slide carriers."""
    s = str(name or "").strip().lower()
    if "slide" in s and ("4" in s or "four" in s):
        return FOUR_SLIDE_CARRIER
    resolved = normalize_plate_format(name, strict=False)
    if resolved is None:
        raise PlateShapeError(f"{name!r} is not a known sample format.")
    return resolved


def load_sample_formats(csv_path=None) -> dict[str, PlateGeometry]:
    """``{format name: PlateGeometry}`` from Squid's sample_formats.csv, falling back to the vendored table."""
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
    """A carrier background PNG plus the transform that puts stage micrometres onto its pixels."""

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
    """The carrier PNG for *format_name*, or None when it is not on disk."""
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
    """A grid of cells, each holding 0, 1 or many FOVs. Subclasses supply only naming."""

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
        #: the geometry the cells actually have: "stage" | "compact".
        self.placement_mode = normalize_placement_mode(placement_mode)
        #: the mode the caller asked for; differs only for a freeform tissue carrier.
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

    #: Prefix of the synthetic ids a subclass invents to keep its grid rectangular.
    _FILLER_PREFIX = ""

    def _is_filler(self, cell_id: str) -> bool:
        """Is *cell_id* a synthetic placeholder rather than a region the acquisition named?"""
        return bool(self._FILLER_PREFIX) and str(cell_id).startswith(self._FILLER_PREFIX)

    def _sole_cell(self, axis: int, i: int) -> Optional[str]:
        """The id of the only real region on row (*axis* 0) or column (*axis* 1) *i*, else None."""
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
        """The on-screen text for this plate's geometry: ``"stage"`` or ``"compact"``."""
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
        """This plate's carrier PNG, or None when the artwork is not available."""
        return carrier_art(self.format_name, images_dir=images_dir, geometry=self.geometry)

    def cell_layout(self) -> Optional[dict[str, tuple[float, float, float, float]]]:
        """``{cell_id: (x, y, w, h)}`` in grid units, or None for a uniform grid."""
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
        """``(row_labels, col_labels, {(r, c): cell_id}, occupied_cells)`` -- PlateOverview's args."""
        return self.row_labels, self.col_labels, self.occupied_map, self.occupied_cells

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.format_name} {self.rows}x{self.cols} "
                f"occupied={len(self._occupancy)} source={self.format_source} "
                f"placement={self.placement_mode}>")


# --------------------------------------------------------------------------- WellPlate

def _row_letter(i: int) -> str:
    """0->A, 25->Z, 26->AA; `_plate_shape._row_index` is its inverse."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def display_well_id(cell_id: str) -> str:
    """Display-only numeric well id (``A1 -> "11"``); never parsed back, freeform ids pass through."""
    s = str(cell_id)
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    letters, digits = s[:i], s[i:]
    # A well row label is uppercase and short; freeform ids pass through untouched.
    if not letters or not digits.isdigit() or not letters.isupper() or len(letters) > 2:
        return s
    return f"{_row_index(letters) + 1}{digits}"


# Fixed-width integer id (superseded by squidxplorer._address; the cache still uses it):
# row * 1_000_000 + col * 10_000 + roi, 0-based row/col, 4-digit ROI slot.
_ROW_MUL = 1_000_000
_COL_MUL = 10_000
_ROI_MAX = 10_000


def well_code(cell_id: str) -> "Optional[int]":
    """The fixed-width integer id for a well (ROI slot 0), or None for a freeform region."""
    s = str(cell_id)
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    letters, digits = s[:i], s[i:]
    if not letters or not digits.isdigit() or not letters.isupper() or len(letters) > 2:
        return None
    row = _row_index(letters)
    col = int(digits) - 1
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
    """``code -> (row, col, roi)``, all 0-based; the inverse of :func:`well_code`/:func:`roi_code`."""
    code = int(code)
    return (code // _ROW_MUL, (code // _COL_MUL) % 100, code % _ROI_MAX)


def format_code(code: int) -> str:
    """Human form ``"RR CC OOOO"`` (Row Column ROI)."""
    row, col, roi = decode_code(code)
    return f"{row:02d} {col:02d} {roi:04d}"


def cache_scope(cell_id: str, roi_index: "Optional[int]" = None) -> str:
    """The flat-cache scope string: the integer id for a real well, else the raw region key."""
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
    """A slide holder: a grid whose cells are named by the acquisition's own region ids."""

    _FILLER_PREFIX = "slot"

    def __init__(self, geometry, occupancy=None, cell_ids: Optional[Iterable[str]] = None,
                 placement: Optional[Mapping[str, tuple]] = None,
                 layout: Optional[Mapping[str, tuple]] = None,
                 stage_boxes_um: Optional[Mapping[str, tuple]] = None, **kw):
        """*placement*/*layout* carry the geometric assignment; both absent means positional order."""
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
        if self.rows == 1:
            return [""]
        return [self._sole_cell(0, r) or "" for r in range(self.rows)]

    @property
    def col_labels(self) -> list[str]:
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
    """The occupied regions of any format, packed into an even grid: ``placement="compact"``."""

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
        #: raw stage boxes of the regions; the compact cells are not at these positions.
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
        """Refused: a compact cell is not at a stage position."""
        raise PlateBuildError(
            f"{cell_id!r} is a cell of a COMPACT plate, whose cells are packed for browsing and "
            "are not at stage positions. There is no stage centre to return. Rebuild with "
            f"placement={STAGE!r} to ask this question."
        )


# --------------------------------------------------------------------------- freeform geometry

def region_stage_boxes_um(metadata: Mapping, regions: Optional[Iterable[str]] = None
                          ) -> dict[str, tuple[float, float, float, float]]:
    """``{region: (x_um, y_um, w_um, h_um)}`` -- each region's mosaic bounding box on the stage.

    Regions with no recorded position are omitted rather than given a zero box.
    """
    positions = metadata.get("fov_positions_um") or {}
    fovs_per_region = metadata.get("fovs_per_region") or {}
    scoped = list(metadata.get("regions") or []) if regions is None else list(regions)
    # A stage position marks the frame's corner, so one extra frame is added past the last one.
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
    """``(rows, cols, {region: (row, col)})`` clustered from stage boxes, never from names."""
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
    """``(rows, cols, placement, layout)``: regions packed into equal, landscape-biased cells."""
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
    """Stage boxes -> ``{region: (x, y, w, h)}`` in grid units, aspect preserved and centred."""
    if not boxes_um:
        return {}
    x0 = min(b[0] for b in boxes_um.values())
    y0 = min(b[1] for b in boxes_um.values())
    x1 = max(b[0] + b[2] for b in boxes_um.values())
    y1 = max(b[1] + b[3] for b in boxes_um.values())
    uw, uh = x1 - x0, y1 - y0
    if not (uw > 0 and uh > 0):
        return {}  # degenerate: no scale to preserve, caller keeps the nominal grid
    s = min(cols / uw, rows / uh)
    ox, oy = (cols - uw * s) / 2.0, (rows - uh * s) / 2.0
    return {r: (ox + (b[0] - x0) * s, oy + (b[1] - y0) * s, b[2] * s, b[3] * s)
            for r, b in boxes_um.items()}


# --------------------------------------------------------------------------- measurement

def measure_region_pitch_um(positions_um: Mapping[tuple, tuple], regions: Iterable[str]
                            ) -> tuple[Optional[float], Optional[float]]:
    """Centre-to-centre well pitch (x_um, y_um) measured from stage coordinates, or (None, None)."""
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
        if span is None:  # freeform id -> not a well grid, nothing to measure
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

    return _axis(0, 1, 0), _axis(1, 0, 1)  # x: same row, varying column; y: same column


def format_from_pitch_um(pitch_x_um: Optional[float], pitch_y_um: Optional[float],
                         tolerance: float = _PITCH_TOL) -> Optional[str]:
    """The standard wellplate format whose pitch matches, or None when the match is not unambiguous."""
    def _match(p):
        if p is None or p <= 0:
            return None
        hits = [name for name, g in ((n, PlateGeometry.vendored(n)) for n in _WELLPLATE_FORMATS)
                if g.pitch_x_um > 0 and abs(p - g.pitch_x_um) <= tolerance * g.pitch_x_um]
        return hits[0] if len(hits) == 1 else None

    mx, my = _match(pitch_x_um), _match(pitch_y_um)
    if mx and my:
        return mx if mx == my else None
    return mx or my


# --------------------------------------------------------------------------- the builder

def build_plate(metadata: Mapping, override=None, images_dir=None,
                placement: str = DEFAULT_PLACEMENT_MODE) -> Plate:
    """Build the :class:`Plate` an acquisition describes.

    Format precedence: ``override > measured > declared > inferred-from-span``.
    """
    placement = normalize_placement_mode(placement)
    regions = list(metadata.get("regions") or [])
    fovs_per_region = dict(metadata.get("fovs_per_region") or {})
    positions_um = metadata.get("fov_positions_um") or {}
    declared_raw = metadata.get("wellplate_format")

    stage_boxes = region_stage_boxes_um(metadata, regions) if regions else {}
    if len(stage_boxes) != len(regions):
        stage_boxes = {}  # cannot place every region -> place none by geometry

    forced = normalize_plate_format(
        override if override is not None else os.environ.get("SQUIDXPLORER_WELLPLATE_FORMAT"),
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
                f"the true scale. Override with SQUIDXPLORER_WELLPLATE_FORMAT if the yaml is right."
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
    """``{region: (y, x)}`` -- the row-major sort key that gives a compact grid its reading order."""
    key: dict[str, tuple[float, float]] = {}
    for region in regions:
        box = (stage_boxes or {}).get(region)
        if box is not None:
            key[region] = (float(box[1]), float(box[0]))  # (y_um, x_um)
            continue
        span = well_span([region])  # "B3" -> (2, 3), 1-based
        key[region] = (float(span[0]), float(span[1])) if span else (0.0, 0.0)
    return key


def _make(name, regions, fovs_per_region, stage_boxes=None,
          placement_mode=DEFAULT_PLACEMENT_MODE, **kw) -> Plate:
    """Instantiate the right subclass for *name*, sizing a slide carrier to the regions present."""
    occupancy = {r: list(fovs_per_region.get(r, ())) for r in regions}
    if name in _SLIDE_FORMATS or well_span(regions) is None:
        # Pick the smallest standard holder with room for every region.
        n = max(1, len(regions))
        if name not in _SLIDE_FORMATS:
            name = GLASS_SLIDE  # freeform ids under a plate format name
        if n > 1 and name == GLASS_SLIDE:
            name = FOUR_SLIDE_CARRIER
        geom = PlateGeometry.vendored(name)
        # Even layout: stage boxes, when present, only order the cells left-to-right.
        placement = layout = None
        if n >= 1:
            rows, cols, placement, layout = even_carrier_layout(list(regions), order_key=stage_boxes)
            geom = PlateGeometry(**{**vars(geom), "rows": rows, "cols": cols})
        if placement is None and n > geom.rows * geom.cols:
            # More slides than any standard carrier: widen rather than refuse.
            geom = PlateGeometry(**{**vars(geom), "cols": n})
        # The carrier's cells are even in both modes, so it reports the mode it actually has.
        carrier_mode = COMPACT if placement else placement_mode
        return SlideCarrier(geom, occupancy, cell_ids=list(regions), placement=placement,
                            layout=layout, stage_boxes_um=stage_boxes,
                            format_name=geom.name, placement_mode=carrier_mode,
                            placement_requested=placement_mode, **kw)

    if placement_mode == COMPACT and regions:
        base = PlateGeometry.vendored(name)
        rows, cols, cells, layout = even_carrier_layout(
            list(regions), order_key=_compact_order_key(regions, stage_boxes))
        geom = PlateGeometry(**{**vars(base), "rows": rows, "cols": cols})
        return CompactPlate(geom, occupancy, placement=cells, layout=layout,
                            stage_boxes_um=stage_boxes, format_name=name,
                            placement_requested=placement_mode, **kw)

    return WellPlate(PlateGeometry.vendored(name), occupancy, format_name=name,
                     placement_mode=placement_mode, **kw)
