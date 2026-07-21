"""Squid acquisition -> tiles + stage positions, for the benchmark harness.

Identity comes from FILENAMES, geometry comes from coordinates.csv. That split is
deliberate and load-bearing: Squid's ``coordinates.csv`` has no ``fov`` column in the
20x layout, so row order is the only thing that carries the FOV index, and row order
has been observed to disagree with the filename token. Filenames are ground truth for
*which* tile this is; the CSV is consulted only for *where* it sat on the stage.

Two on-disk coordinate layouts exist and both are handled:

    Monkey-style  region,fov,z_level,x (mm),y (mm),z (um),time   <- explicit fov column
    20x-style     region,x (mm),y (mm),z (mm)                    <- row order = fov index

Filename grammar (Squid ``individual_tiffs``)::

    {region}_{fov}_{z}_{channel}.tiff
    C5_15_0_Fluorescence_561_nm_Ex.tiff
     |   |  |  \\_ channel (may itself contain underscores)
     |   |  \\_ z index
     |   \\_ fov index
     \\_ region / well id
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

# {region}_{fov}_{z}_{channel}.tif[f] -- channel is greedy-to-extension because real
# channel names carry underscores ("Fluorescence_561_nm_Ex").
_NAME_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)\.tiff?$")

# A pair of tiles must share at least this fraction of a frame to be worth measuring.
# Below it the overlap strip is too thin for a meaningful block correlation.
MIN_OVERLAP_FRACTION = 0.02


class AcquisitionError(RuntimeError):
    """The acquisition on disk is not usable as a benchmark input."""


@dataclass(frozen=True)
class Tile:
    region: str
    fov: int
    z: int
    channel: str
    path: Path

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.region, self.fov, self.z, self.channel)


@dataclass
class Acquisition:
    """A parsed Squid acquisition, ready to hand to a stitcher adapter."""

    root: Path
    tiles: dict[tuple[str, int, int, str], Tile]
    positions_mm: dict[tuple[str, int], tuple[float, float]]
    pixel_size_um: float
    frame_shape: tuple[int, int]
    dtype: str
    regions: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    z_levels: list[int] = field(default_factory=list)

    def fovs(self, region: str) -> list[int]:
        return sorted({f for (r, f) in self.positions_mm if r == region})

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    def tile_path(self, region: str, fov: int, z: int, channel: str) -> Path:
        try:
            return self.tiles[(region, fov, z, channel)].path
        except KeyError as exc:  # pragma: no cover - defensive
            raise AcquisitionError(
                f"no tile for region={region} fov={fov} z={z} channel={channel}"
            ) from exc

    def read(self, region: str, fov: int, z: int, channel: str) -> np.ndarray:
        return tifffile.imread(self.tile_path(region, fov, z, channel))

    def positions_px(self, region: str) -> dict[int, tuple[float, float]]:
        """Stage positions for one region, converted to pixels, origin at the min corner.

        Returns ``{fov: (row, col)}``. Stage x maps to column, stage y maps to row.
        """
        pos = {f: self.positions_mm[(region, f)] for f in self.fovs(region)}
        if not pos:
            return {}
        x0 = min(x for x, _ in pos.values())
        y0 = min(y for _, y in pos.values())
        scale = 1000.0 / self.pixel_size_um  # mm -> um -> px
        return {f: ((y - y0) * scale, (x - x0) * scale) for f, (x, y) in pos.items()}


def _parse_pixel_size_um(root: Path) -> float:
    """sensor pixel size / objective magnification, from ``acquisition parameters.json``."""
    for name in ("acquisition parameters.json", "acquisition_parameters.json"):
        p = root / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        sensor = data.get("sensor_pixel_size_um")
        mag = (data.get("objective") or {}).get("magnification")
        if sensor and mag:
            return float(sensor) / float(mag)
        raise AcquisitionError(
            f"{p} lacks sensor_pixel_size_um / objective.magnification; "
            "cannot convert stage mm to pixels"
        )
    raise AcquisitionError(f"no acquisition parameters.json under {root}")


def _parse_coordinates(csv_path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """coordinates.csv -> {(region, fov): (x_mm, y_mm)}, handling both known layouts."""
    import csv as _csv

    with csv_path.open(newline="") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        raise AcquisitionError(f"{csv_path} is empty")

    cols = {c.strip().lower(): c for c in rows[0]}

    def col(*cands: str) -> str | None:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    region_c = col("region")
    x_c = col("x (mm)", "x(mm)", "x")
    y_c = col("y (mm)", "y(mm)", "y")
    fov_c = col("fov")
    if not (region_c and x_c and y_c):
        raise AcquisitionError(f"{csv_path} lacks region/x/y columns; got {list(cols)}")

    out: dict[tuple[str, int], tuple[float, float]] = {}
    seq: dict[str, int] = {}
    for row in rows:
        region = (row[region_c] or "").strip()
        if not region:
            continue
        if fov_c is not None and (row.get(fov_c) or "").strip():
            fov = int(float(row[fov_c]))
        else:
            # 20x layout: no fov column, row order within a region IS the fov index.
            fov = seq.get(region, 0)
            seq[region] = fov + 1
        key = (region, fov)
        if key in out:
            continue  # z-stack repeats the same XY per fov; first wins
        out[key] = (float(row[x_c]), float(row[y_c]))
    if not out:
        raise AcquisitionError(f"{csv_path} produced no positions")
    return out


def _scan_tiles(root: Path) -> dict[tuple[str, int, int, str], Tile]:
    """Walk the acquisition for TIFFs and parse identity from each filename."""
    tiles: dict[tuple[str, int, int, str], Tile] = {}
    for path in sorted(root.rglob("*.tif*")):
        if not path.is_file():
            continue  # a dead symlink resolves False here -- see load_acquisition()
        m = _NAME_RE.match(path.name)
        if not m:
            continue
        t = Tile(
            region=m["region"],
            fov=int(m["fov"]),
            z=int(m["z"]),
            channel=m["channel"],
            path=path,
        )
        tiles[t.key] = t
    return tiles


def load_acquisition(root: str | Path) -> Acquisition:
    """Parse a Squid acquisition directory into an :class:`Acquisition`.

    Raises :class:`AcquisitionError` with an actionable message rather than letting a
    decode failure surface from deep inside tifffile. This matters: the ``sim_*``
    fixtures on this machine are symlink farms whose source was deleted, and a guard
    that only checks ``is_dir()`` lets a dead tree through to fail confusingly later.
    """
    root = Path(root)
    if not root.is_dir():
        raise AcquisitionError(f"not a directory: {root}")

    tiles = _scan_tiles(root)
    if not tiles:
        # Distinguish "no images" from "images present but every link is broken",
        # because the second one is the sim_* failure mode and reads very differently.
        dangling = [p for p in root.rglob("*.tif*") if p.is_symlink() and not p.exists()]
        if dangling:
            raise AcquisitionError(
                f"{root}: {len(dangling)} TIFF symlink(s) are dangling (e.g. "
                f"{dangling[0].name} -> {dangling[0].readlink()}). The source "
                "acquisition was deleted; regenerate or remove this fixture."
            )
        raise AcquisitionError(f"{root}: no Squid-named TIFFs found")

    csv_path = root / "coordinates.csv"
    if not csv_path.is_file():
        raise AcquisitionError(
            f"{root}: no coordinates.csv -- there is no stage geometry, so tiles "
            "cannot be placed and no stitching metric is possible"
        )
    positions = _parse_coordinates(csv_path)

    probe = next(iter(tiles.values()))
    with tifffile.TiffFile(probe.path) as tf:
        page = tf.pages[0]
        frame_shape = (int(page.shape[0]), int(page.shape[1]))
        dtype = str(page.dtype)

    regions = sorted({t.region for t in tiles.values()})
    channels = sorted({t.channel for t in tiles.values()})
    z_levels = sorted({t.z for t in tiles.values()})

    # Positions are keyed by row order; if that disagrees with the filename fovs the
    # geometry is untrustworthy and every downstream residual would be silently wrong.
    for region in regions:
        named = {t.fov for t in tiles.values() if t.region == region}
        placed = {f for (r, f) in positions if r == region}
        missing = named - placed
        if missing:
            raise AcquisitionError(
                f"region {region}: {len(missing)} fov(s) named in filenames have no "
                f"row in coordinates.csv (e.g. fov={sorted(missing)[0]}). Filename "
                "fovs and CSV row order disagree; geometry cannot be trusted."
            )

    return Acquisition(
        root=root,
        tiles=tiles,
        positions_mm=positions,
        pixel_size_um=_parse_pixel_size_um(root),
        frame_shape=frame_shape,
        dtype=dtype,
        regions=regions,
        channels=channels,
        z_levels=z_levels,
    )
