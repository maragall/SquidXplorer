"""Persistent two-tier cache (RAM, then disk) for plate preview cells.

Cells are keyed by ``(token, t, region)`` and stored under ``platformdirs.user_cache_dir``,
never inside the acquisition folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from squidxplorer._budget import cache_budget
from squidxplorer._mosaic_source import MemoryBoundedLRUCache

from squidxplorer._logpane import get_logger

log = get_logger("platecache")

#: Part of the token; bumping it makes entries from an older format unreachable.
FORMAT_VERSION = 2

#: Set to "0" to turn the cache off entirely.
ENV_ENABLED = "SQUIDXPLORER_PLATE_CACHE"

#: Overrides ``user_cache_dir``.
ENV_DIR = "SQUIDXPLORER_CACHE_DIR"

#: Acquisition metadata files whose mtimes feed the token, one stat each.
_TOKEN_FILES = ("acquisition.yaml", "coordinates.csv", "acquisition parameters.json",
                "configurations.xml")

#: Ceiling on stats spent building one token; extra timepoint dirs hash names, not mtimes.
MAX_TOKEN_STATS = 64

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


# --- where the cells live ----------------------------------------------------------------------

def cache_root() -> Path:
    """The cache directory for this user; never inside an experiment."""
    override = os.environ.get(ENV_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("squidxplorer", "cephla"))


def experiment_slug(path) -> str:
    """A short, stable, filesystem-safe namespace for one acquisition path."""
    p = str(Path(path).expanduser())
    digest = hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
    stem = _UNSAFE.sub("_", Path(p).name)[:40] or "acq"
    return f"{stem}-{digest}"


def plate_token(path, extra: Sequence = ()) -> str:
    """A staleness token for the whole acquisition, from a bounded number of stats."""
    h = hashlib.sha256()
    h.update(f"squidxplorer-plate-cells v{FORMAT_VERSION}\0".encode("utf-8"))
    h.update(f"{Path(path).expanduser()}\0".encode("utf-8"))
    for part in extra:
        h.update(f"{part}\0".encode("utf-8"))

    root = Path(path)
    spent = 0
    for target in (root, *(root / name for name in _TOKEN_FILES)):
        h.update(_stat_mark(target))
        spent += 1

    # A new plane moves the timepoint directory's mtime, not the root's.
    try:
        subdirs = sorted(e.name for e in os.scandir(root) if e.is_dir())
    except OSError:
        subdirs = []
    h.update(("|".join(subdirs) + "\0").encode("utf-8"))
    for name in subdirs:
        if spent >= MAX_TOKEN_STATS:
            break
        h.update(_stat_mark(root / name))
        spent += 1
    return h.hexdigest()[:16]


def _stat_mark(p: Path) -> bytes:
    """``(mtime_ns, size)`` of *p*, or a marker meaning absent."""
    try:
        st = os.stat(p)
    except OSError:
        return b"\0absent\0"
    return f"\0{st.st_mtime_ns}:{st.st_size}\0".encode("utf-8")


def _assert_outside_experiment(cache_dir, experiment) -> None:
    """Refuse, loudly, to write anything under the acquisition folder."""
    try:
        c = Path(cache_dir).expanduser().resolve()
        e = Path(experiment).expanduser().resolve()
    except OSError:                                  # a vanished path cannot contain anything
        return
    if c == e or e in c.parents:
        raise RuntimeError(
            f"refusing to write the plate cache to {c}: it is inside the acquisition folder {e}. "
            f"SquidXplorer never writes into your data -- experiments live on Dropbox, NAS and "
            f"read-only mounts. Unset {ENV_DIR}, or point it at somewhere you own.")


# --- the cached value --------------------------------------------------------------------------

class CellTile(np.ndarray):
    """The cell's pixels plus ``box`` -- the ``(top, left, h, w)`` sub-rectangle they cover."""

    def __new__(cls, array, box):
        obj = np.asarray(array).view(cls)
        obj.box = tuple(int(v) for v in box)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.box = getattr(obj, "box", None)


# --- the cache -----------------------------------------------------------------------------

#: The process-wide RAM tier, bounded by bytes.
_CELLS = MemoryBoundedLRUCache(cache_budget())


def enabled(env: Optional[dict] = None) -> bool:
    """Whether caching is on. Off is a supported state, not a broken one."""
    src = os.environ if env is None else env
    return str(src.get(ENV_ENABLED, "1")).strip().lower() not in ("0", "false", "no", "off")


class PlateCellCache:
    """Two tiers over one acquisition's plate cells: bytes-bounded RAM, then atomic files on disk.

    Layout: ``<user_cache_dir>/cells/<name>-<sha(path)>/<token>/t<t>-<region>-<sha>.npz``.
    The timepoint is in the key, not the token, so timepoints coexist under one generation.
    """

    def __init__(self, experiment_path, *, cell_px: int, channels: Sequence[str], dtype,
                 time_point: int = 0, root: Optional[Path] = None) -> None:
        self.experiment = Path(experiment_path).expanduser()
        self.cell_px = int(cell_px)
        self.channels = [str(c) for c in channels]
        self.dtype = np.dtype(dtype)
        #: Which timepoint these cells are of; fixed for the life of the instance.
        self.time_point = max(0, int(time_point))
        self.token = plate_token(self.experiment,
                                 (self.cell_px, self.dtype.str, ",".join(self.channels)))
        base = Path(root) if root is not None else cache_root()
        self.dir = base / "cells" / experiment_slug(self.experiment) / self.token
        _assert_outside_experiment(self.dir, self.experiment)
        self._pruned = False
        self._index: Optional[dict] = None    # the packed page's sidecar, read once
        self._page = None                     # the memory-mapped page itself
        self.packed = False
        self.hits = 0
        self.misses = 0
        self.writes = 0

    # ---- construction from what the viewer has in hand ----------------------------------
    @classmethod
    def for_reader(cls, reader, meta, *, cell_px: int, time_point: int = 0,
                   root: Optional[Path] = None) -> "Optional[PlateCellCache]":
        """The cache for this reader's acquisition, or ``None`` with the reason logged."""
        if not enabled():
            return None
        try:
            from squidxplorer._mosaic_source import _source_token

            path = _source_token(reader)          # the acquisition PATH, never id(reader)
            channels = [c["name"] for c in meta["channels"]]
            return cls(path, cell_px=cell_px, channels=channels,
                       dtype=np.dtype(meta["dtype"]), time_point=time_point, root=root)
        except Exception as exc:                  # noqa: BLE001 - degrade to uncached, but SAY so
            log.info("plate cell cache disabled for this acquisition: %s: %s",
                     type(exc).__name__, exc)
            return None

    # ---- read ---------------------------------------------------------------------------
    def get(self, region: str) -> Optional[CellTile]:
        """This region's cell AT THIS CACHE'S TIMEPOINT, or ``None``. RAM, page, then one file."""
        ram_key = self._ram_key(region)
        hit = _CELLS.get(ram_key)
        if hit is not None:
            self.hits += 1
            return hit
        packed = self._from_pack(region)
        if packed is not None:
            self.hits += 1
            return packed
        path = self.path_for(region)
        try:
            with np.load(path) as z:
                tile = CellTile(z["tile"], tuple(int(v) for v in z["box"]))
        except FileNotFoundError:
            self.misses += 1
            return None
        except Exception as exc:                  # noqa: BLE001 - a damaged entry is just a miss
            log.warning("dropping an unreadable cache entry %s: %s: %s",
                        path, type(exc).__name__, exc)
            _unlink(path)
            self.misses += 1
            return None
        _CELLS.put(ram_key, tile)
        self.hits += 1
        return tile

    def load_all(self, regions: Iterable[str]) -> dict:
        """``{region: CellTile}`` for every region that hits. A miss is simply absent."""
        out: dict = {}
        for region in regions:
            hit = self.get(region)
            if hit is not None:
                out[str(region)] = hit
        return out

    # ---- write --------------------------------------------------------------------------
    def put(self, region: str, tile: np.ndarray, box) -> bool:
        """Publish this region's cell. Atomic; returns whether it reached disk."""
        arr = CellTile(np.asarray(tile), box)
        _CELLS.put(self._ram_key(region), arr)
        path = self.path_for(region)
        _assert_outside_experiment(path.parent, self.experiment)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                np.savez(f, tile=np.asarray(arr), box=np.asarray(arr.box, dtype=np.int32))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)          # atomic publish: a reader sees old or new, never half
        except Exception as exc:           # noqa: BLE001 - a full disk must not fail an open
            _unlink(tmp)
            log.warning("plate cell cache write failed (%s); continuing uncached: %s",
                        type(exc).__name__, exc)
            return False
        self.writes += 1
        if not self._pruned:
            self._pruned = True
            self.prune_stale()
        return True

    # ---- the packed page ----------------------------------------------------------------

    #: The compacted page and its sidecar, per timepoint; plain .npy so it can be memory-mapped.
    PACK_ARRAY = "plate-cells-t{t}.npy"
    PACK_INDEX = "plate-index-t{t}.json"

    @property
    def pack_array_path(self) -> Path:
        """This timepoint's compacted page."""
        return self.dir / self.PACK_ARRAY.format(t=self.time_point)

    @property
    def pack_index_path(self) -> Path:
        """This timepoint's sidecar, and the commit marker for its page."""
        return self.dir / self.PACK_INDEX.format(t=self.time_point)

    def pack(self, regions: Iterable[str]) -> bool:
        """Compact this generation into one memory-mapped page. False if it is not complete."""
        regions = [str(r) for r in regions]
        if not regions:
            return False
        # Already compacted: rewriting would os.replace a page this process may hold mmapped,
        # which Windows refuses (WinError 5).
        index = self._pack_index()
        if index is not None and all(str(r) in index["at"] for r in regions):
            self.packed = True
            return True
        cells = []
        for region in regions:
            hit = self.get(region)
            if hit is None:
                return False                  # incomplete: there is no page to write
            cells.append(hit)

        n, c, cell = len(regions), len(self.channels), self.cell_px
        page = np.zeros((n, c, cell, cell), dtype=self.dtype)
        boxes = []
        for i, hit in enumerate(cells):
            top, left, h, w = hit.box
            arr = np.asarray(hit)
            if arr.shape != (c, h, w) or top + h > cell or left + w > cell:
                return False                  # a cell that does not fit the page is not packable
            page[i, :, top:top + h, left:left + w] = arr
            boxes.append([int(v) for v in hit.box])

        index = {"format": FORMAT_VERSION, "token": self.token, "t": self.time_point,
                 "cell_px": cell, "channels": list(self.channels), "dtype": self.dtype.str,
                 "regions": regions, "boxes": boxes}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            _assert_outside_experiment(self.dir, self.experiment)
            arr_path = self.pack_array_path
            tmp = arr_path.with_name(f".{arr_path.name}.{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                np.lib.format.write_array(f, page)     # NOT np.save: it would append .npy to tmp
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, arr_path)
            idx_path = self.pack_index_path
            tmp = idx_path.with_name(f".{idx_path.name}.{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(index, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, idx_path)          # the commit: a reader looks for THIS first
        except Exception as exc:               # noqa: BLE001 - the per-well files still stand
            log.warning("plate cell pack failed (%s); the per-well cells still serve: %s",
                        type(exc).__name__, exc)
            return False
        for region in regions:
            _unlink(self.path_for(region))     # compacted; the page is the copy that survives
        self._index = None                     # re-read, so this instance sees its own page
        self._page = None
        self.packed = True
        return True

    def _from_pack(self, region: str) -> Optional[CellTile]:
        """One region's cell out of the memory-mapped page, or None if there is no page."""
        index = self._pack_index()
        if index is None:
            return None
        i = index["at"].get(str(region))
        if i is None:
            return None
        page = self._page
        if page is None:
            try:
                page = self._page = np.load(self.pack_array_path, mmap_mode="r")
            except Exception as exc:           # noqa: BLE001 - a damaged page is a miss
                log.warning("dropping an unreadable plate pack in %s: %s", self.dir, exc)
                self._index = self._page = None
                return None
        top, left, h, w = index["boxes"][i]
        return CellTile(page[i, :, top:top + h, left:left + w], (top, left, h, w))

    def _pack_index(self) -> Optional[dict]:
        if self._index is None:
            try:
                raw = json.loads(self.pack_index_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if raw.get("token") != self.token or raw.get("format") != FORMAT_VERSION:
                return None                    # a page from another generation is not ours
            if int(raw.get("t", 0)) != self.time_point:
                return None
            raw["at"] = {r: i for i, r in enumerate(raw["regions"])}
            self._index = raw
        return self._index

    # ---- housekeeping -------------------------------------------------------------------
    def prune_stale(self) -> int:
        """Delete this acquisition's other token directories. Returns how many went."""
        gone = 0
        parent = self.dir.parent
        try:
            siblings = [e for e in os.scandir(parent) if e.is_dir()]
        except OSError:
            return 0
        for entry in siblings:
            if entry.name == self.token:
                continue
            try:
                shutil.rmtree(entry.path)
                gone += 1
            except OSError as exc:                # noqa: PERF203 - one failure must not stop the rest
                log.info("could not prune the stale cache generation %s: %s", entry.path, exc)
        if gone:
            log.info("pruned %d stale plate cache generation(s) under %s", gone, parent)
        return gone

    def path_for(self, region: str) -> Path:
        """One file per region per timepoint; the digest suffix makes any region id safe and unique."""
        r = str(region)
        clean = _UNSAFE.sub("_", r)[:40] or "region"
        digest = hashlib.sha256(r.encode("utf-8")).hexdigest()[:8]
        return self.dir / f"t{self.time_point}-{clean}-{digest}.npz"

    def _ram_key(self, region: str) -> tuple:
        """The process-wide RAM tier's key: ``(token, t, region)``."""
        return (self.token, self.time_point, str(region))

    def __repr__(self) -> str:
        return (f"PlateCellCache({self.experiment.name!r}, token={self.token}, "
                f"t={self.time_point}, hits={self.hits}, misses={self.misses}, "
                f"writes={self.writes})")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def clear_memory_tier() -> None:
    """Drop the process-wide RAM tier. For tests, and for a window closing on a huge plate."""
    _CELLS.clear()
