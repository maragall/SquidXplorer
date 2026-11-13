"""The numbers behind a movie: per-frame intensity statistics and line profiles, as CSV.

A movie is a LOOK (contrast latched, colours baked in), so nothing may be measured off it.
These rows are computed from the SAME raw-count planes each frame was composited from, before
any windowing, so a row and the frame it names describe the same pixels. Qt-free on purpose.

Two files land beside ``<name>.mp4``:

* ``<name>.csv``: one row per frame per channel (:data:`STATS_COLUMNS`);
* ``<name>_profile.csv``: one row per position per frame per channel, along x and along y
  (:data:`PROFILE_COLUMNS`). The x profile is the mean of each column, the y profile the mean of
  each row: what Fiji's Plot Profile returns for a rectangle.

Exact zeros are the mosaic's uncovered area and padded slots (the same rule the contrast seed
states in ``_contrast.auto_contrast``), so every statistic is over the LIT pixels only.

No model is fitted here on purpose: which diffusion model applies is the scientist's knowledge,
and the table opens in Fiji, Excel or Prism for that step.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "STATS_COLUMNS",
    "PROFILE_COLUMNS",
    "frame_stats_rows",
    "frame_profile_rows",
    "measurement_paths",
    "write_rows",
]

#: ``time_s_declared`` is the timepoint times the acquisition's DECLARED interval, never a
#: per-frame clock (Squid records none per timepoint); blank when no interval is declared.
STATS_COLUMNS = ("frame", "time_point", "z_level", "time_s_declared", "channel",
                 "n_pixels", "mean", "median", "std", "min", "max", "pixel_um")
PROFILE_COLUMNS = ("frame", "time_point", "z_level", "time_s_declared", "channel",
                   "axis", "distance_um", "intensity")


def measurement_paths(movie_path) -> tuple[Path, Path]:
    """``(stats.csv, profile.csv)`` beside *movie_path*, named after it."""
    movie = Path(movie_path)
    return movie.with_suffix(".csv"), movie.with_name(f"{movie.stem}_profile.csv")


def _identity(frame: int, time_point: int, z_level: int, time_s: Optional[float]) -> dict:
    return {"frame": int(frame), "time_point": int(time_point), "z_level": int(z_level),
            "time_s_declared": "" if time_s is None else float(time_s)}


def frame_stats_rows(store: np.ndarray, channels: Sequence[str], *, frame: int, time_point: int,
                     z_level: int, time_s: Optional[float], pixel_um: float) -> list[dict]:
    """One :data:`STATS_COLUMNS` row per channel of a ``(C, H, W)`` raw-count *store*."""
    base = _identity(frame, time_point, z_level, time_s)
    rows = []
    for name, plane in zip(channels, store):
        lit = plane[plane > 0]
        if lit.size:
            stats = {"mean": float(lit.mean()), "median": float(np.median(lit)),
                     "std": float(lit.std()), "min": lit.min().item(), "max": lit.max().item()}
        else:
            stats = {"mean": "", "median": "", "std": "", "min": "", "max": ""}
        rows.append({**base, "channel": str(name), "n_pixels": int(lit.size), **stats,
                     "pixel_um": float(pixel_um)})
    return rows


def _lit_mean(plane: np.ndarray, axis: int) -> np.ndarray:
    """Mean over the lit pixels along *axis*; NaN where a whole line is uncovered."""
    lit = plane > 0
    count = lit.sum(axis=axis)
    total = np.where(lit, plane, 0).sum(axis=axis, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def frame_profile_rows(store: np.ndarray, channels: Sequence[str], *, frame: int, time_point: int,
                       z_level: int, time_s: Optional[float], pixel_um: float) -> list[dict]:
    """:data:`PROFILE_COLUMNS` rows for one frame: each channel's x and y intensity profile."""
    base = _identity(frame, time_point, z_level, time_s)
    rows = []
    for name, plane in zip(channels, store):
        for axis_name, reduce_axis in (("x", 0), ("y", 1)):
            for i, value in enumerate(_lit_mean(plane, reduce_axis)):
                rows.append({**base, "channel": str(name), "axis": axis_name,
                             "distance_um": float(i * pixel_um),
                             "intensity": "" if np.isnan(value) else float(value)})
    return rows


def write_rows(path, columns: Sequence[str], rows: Iterable[dict]) -> str:
    """Write *rows* under *columns* to *path*. Returns the path written."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
