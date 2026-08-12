"""The plate preview CELLS, persisted, so reopening an acquisition is a cache read.

Gap 1 of the three-viewers review (Hongquan, 2026-07-28), which is ALSO Spencer's deep-zoom
blocker. Verified at HEAD: ``grep -rn 'platformdirs|user_cache_dir' squidxplorer/`` returned nothing,
``platformdirs`` was not a declared dependency, and both plate producers re-derived every well on
every open. ``_PreviewWorker`` re-streams and re-downsamples one plane per channel per FOV;
``_ComputedPlateWorker`` re-streams two pyramid levels per well. Neither kept a single byte.

The two ends of the same piece of work
--------------------------------------
The review's gap 1 says "persist the preview pass". ``NEXT_STEPS.md`` says, from the other end:

    Coarse rungs cannot be served by ``ReaderTileSource`` as it stands: a fit-to-plate tile
    overlaps all 72 FOVs and measured 25 s to build. The fix that was scoped but not built is a
    composite source, ``InMemoryMultiscale`` fed from the existing ``_PreviewWorker`` pass for
    plate rungs, ``ReaderTileSource`` for FOV rungs.

Those are one thing. The plate rungs Spencer needs ARE the per-well cells this module persists, so
this cache is the producer and :class:`squidxplorer._tilesource.CompositePlateSource` is the consumer.
Building them separately would have built the same thing twice.

PORTED, verbatim, from ``record_zstack_viewer/cache/thumbnails.py``
-------------------------------------------------------------------
* **the mtime token as part of the KEY, not as an invalidation pass.** A changed store yields a
  new token, so a stale entry is simply never looked up. There is no sweep, no generation counter
  and no lock: the entry that would have been wrong is unreachable instead of deleted.
* **temp file plus ``os.replace``.** ``os.replace`` is atomic within a filesystem on POSIX and
  Windows alike, so a reader in another process sees either the previous file or the whole new
  one, never a half-written one. Two windows on one plate race harmlessly.
* **the stale-sibling prune.** Without it a store that changes N times accretes N generations of
  every cell, forever, because the token-in-the-key design never deletes anything by itself.

ADOPTED from ``ndviewer_hcs`` (maragall/ndviewer), the tool in the reference video
-----------------------------------------------------------------------------------
The "MIP Navigator" whose feel the deep-zoom work is chasing is that package with everything
preloaded, so its ``plate_stack.py`` is a reference implementation of this problem rather than an
analogy. What it does, in its own words: "Pre-computed Z x T plate assemblies stored as multi-page
TIFF ... memory-mapped for efficient random access", indexed by ``get_page(t_idx, z_idx)``.

* **TAKEN: the whole-plate page, memory-mapped.** Its granularity is the assembled plate, not the
  well, and for the rungs that hurt it is right: a reopen and a coarse tile both want every well
  at once. Measured here, the per-well form costs 0.261 s to replay a 1536-well plate and 1.59 s
  to seed the coarse rungs, and both numbers ARE 1536 file opens. So this cache publishes per
  well while the preview streams, then COMPACTS the generation into one ``.npy`` page and
  memory-maps it: log first, compact at the end. See :meth:`PlateCellCache.pack`.
* **TAKEN: memory-mapping in preference to holding the page.** A mapping can exceed RAM and stay
  fast, and the operating system's page cache bounds it with more information than we have. The
  byte-bounded LRU stays, but it bounds the STREAMING tier, where cells arrive one at a time and
  there is no page to map yet.
* **REJECTED: one cache file per downsample factor.** Its key is the factor
  (``plate_stack_ds{factor:.4f}.tiff``) because it has no tile ladder, so every zoom level is its
  own precomputed plate. We have a ladder: the FOV rungs are read pixel-exact from the frames and
  the intermediate plate rungs are derived from these cells in RAM under a byte budget
  (``_tilesource.InMemoryMultiscale``). Storing a file per rung would cache what is already cheap.
* **REJECTED: its cache LOCATION.** ``PlateAssembler`` writes to
  ``base_path/"downsampled_image"/assembled_tiles_cache`` -- inside the acquisition folder. That
  is exactly the thing this module refuses; see the non-negotiable section below.
* **REJECTED: pickle for the sidecar.** It uses ``pickle.dump`` for its metadata. A cache under
  $HOME that is unpickled on open is an arbitrary-code-execution surface for anything that can
  write there, for no benefit over JSON on a dict of numbers and strings.
* **TAKEN, 2026-08-05: the t index.** This said "NOT BUILT ... an axis with no producer", and that
  stopped being true the moment ``_PreviewWorker`` learned to read a timepoint. A cell is now
  identified by ``(token, t, region)`` end to end -- RAM key, file name and packed page -- which is
  the same re-keying the loupe's coarse cache took when it went from ``well`` to
  ``(well, timepoint)``. z is still not a plate axis for us by construction: the preview is one
  representative plane, and a z-resolved plate page is a different product (theirs).
  Where ours differs from theirs: their page is ONE array indexed ``get_page(t, z)``, ours is one
  page PER TIMEPOINT (``plate-cells-t{t}.npy``). A pass produces exactly one timepoint, so a
  single (t, region, C, cell, cell) page could only ever be published complete by a producer that
  walked every timepoint -- which nothing does, and which would make the plate's first paint wait
  on timepoints nobody asked to see.

ADAPTED, three things, each decided rather than open
----------------------------------------------------
1. **The RAM tier is bounded BY BYTES**, via ``_mosaic_source.MemoryBoundedLRUCache`` sized from
   ``_budget.cache_budget()``. record-zstack-viewer bounds by item count and hardcodes 192 MB for
   its byte pool; ``_budget`` argues at length that a constant "encodes an assumption about a
   machine it has never seen", and ``_tsctx`` already refused the same literal an hour earlier.
   An item count is worse still here: a cell is 62 KB on a 1536wp and 62 KB on a 4-well plate,
   but the COUNT differs by 384x, so any count is either a leak on the big plate or a no-op on
   the small one.
2. **The cached unit is the plate CELL, not a per-FOV PNG.** Our preview unit is ``(C, cell,
   cell)`` composited from every FOV of a region at ``_placement.cell_boxes`` geometry. A per-FOV
   cache would miss the composite entirely and, being a PNG, would flatten the channel axis,
   which is the axis the global-contrast recomposite on ``streamEnded`` depends on. The channel
   axis stays intact and native-dtype the whole way, exactly as the tile does in flight.
3. **The staleness token is derived from OUR paths and the reader's identity.** Identity is the
   acquisition PATH, following ``_mosaic_source._source_token``: explicitly not ``id(reader)``,
   because a cache entry outlives the reader that made it and CPython recycles ids, so a new
   reader over a different dataset could be served another acquisition's pixels.

MEASURED: what it buys
----------------------
Reopening a plate, cold cache versus warm, this machine, page cache warm both times (so these
are the CONSERVATIVE numbers: the colder the storage, the more the cache wins)::

    dataset                                        cold open   warm open   ratio   on disk
    real 10x tissue, 2 regions x 55 FOVs, 4 ch      0.670 s     0.022 s      31x   0.06 MB
    sim_1536wp, 1536 wells, 4 ch                   15.221 s     0.075 s     202x     91 MB

MEASURED: what the timepoint in the key costs and buys
------------------------------------------------------
Stepping the plate's bar t=0 -> t=1 -> t=0 on ``sim_5d_2x2_t3`` (4 regions x 4 FOVs x 2 channels,
256 px frames, 3 timepoints), ``tools/measure_plate_t_steps.py``, median of 9, a cold CELL cache
per repetition and the OS page cache warm, both columns run back to back on this machine::

    step               before                              after
    t=0 first visit    13.9 ms, 4 wells read               13.6 ms, 4 wells read
    t=1                 6.4 ms, 4 HITS -- frame 0's cells  13.9 ms, 4 wells read
    t=0 again           5.3 ms, 4 hits                      7.0 ms, 4 hits
    t=0 and t=1 differ  FALSE                               True

The before column IS the bug, and the fourth row is why the first three cannot be read as a
benchmark: the second step was the FASTEST of the three because it answered t=1 with t=0's cells.
Note what the fix does to that reading -- stepping to a NEW timepoint gets 7.5 ms SLOWER, because
it now does the work instead of answering the wrong question. What the key buys is the third row:
a revisited timepoint stays a hit, which a timepoint-in-the-TOKEN design would have thrown away
(``prune_stale`` deletes token directories, so t=1 would have deleted t=0's cells).

The warm 1536-well open is one memory-map of the compacted page plus 1536 slices and 1536 signal
emissions, and that is what 0.075 s buys; before compaction, when it was 1536 file opens, the
same replay measured 0.261 s. The tissue set replays TWO tiles, one per region, because a cached
mosaic is one cell and not 27 fields, so it skips 53 of the 55 round trips as well as all 55
reads. On disk that plate is ONE 91 MB ``.npy`` page plus a 35 KB index, not 1536 files.

MEASURED: why the token is plate-level and not per-FOV
------------------------------------------------------
record-zstack-viewer stats 2 to 4 paths per FOV to build its token. That is per-FOV invalidation
granularity, and it is not free at plate scale. Measured on ``sim_1536wp`` (1536 wells, 4
channels, 6144 image paths, symlinks into a real 20x scan), local APFS SSD:

    per-FOV, rzv style   6144 stats   12.8 ms   (2.1 us each; 10.8 ms warm)
    plate level, here       4 stats    0.013 ms (3 us for the whole token)

Local, the storm costs 13 ms and would be tolerable. The number that decides it is the one on the
store this product actually runs against: the same measurement over a FileProvider-backed Dropbox
mount ran 28.8 us per stat, 14x local, and a cold SMB or NFS round trip is ~1 ms. At 1 ms the
per-FOV token costs SIX SECONDS of stat before the first pixel, on the exact path this cache
exists to make fast, and it grows with the plate. So the token is one bounded set of stats for
the whole acquisition: the root, its metadata files and its timepoint directories.

What that costs us is invalidation GRANULARITY. One rewritten well invalidates the whole plate's
cells rather than that well's. That is the right trade for v1, which is post-acquisition by
design (``docs/SCOPE.md``): the acquisition is finished and on disk before we ever open it, so
partial rewrites are not the case being optimised, and a whole-plate rebuild is exactly the pass
the first open already runs. If live acquisition ever lands, the token grows a per-well tier and
the trade is revisited with a measurement, not with a preference.

NON-NEGOTIABLE: never under the experiment root
-----------------------------------------------
Computed cells go ONLY to ``platformdirs.user_cache_dir``, namespaced by a hash of the experiment
path. Squid experiments live on Dropbox, NAS and read-only mounts, and
``docs/hcs-viewer-quickstart.md`` promises the user, twice, "Read only. It never changes your
acquisition." ``docs/SCOPE.md`` says the same to us: "Datasets are READ ONLY. Never copy or
convert them -- a copy once filled the machine to 0 bytes."  :func:`_assert_outside_experiment`
re-checks that
on every publish rather than trusting a root computed once at construction, and a test asserts it
directly. The failure this guards against is not a write that fails on a read-only mount, which
is loud: it is a hidden sidecar written into somebody's data folder that then syncs to their
whole lab.

**If compute or storage ever moves off this workstation** (review section 6.5) the constraint
above is workstation shaped, and its REASONING, not its answer, is what ports. The reasoning is
"never write into a store you do not own". With a remote object store and shared compute the same
reasoning produces a shared server-side tier, and the mtime token is replaced by whatever
generation or etag that store exposes. The design ports; the location does not.

What is bounded, and what is not
--------------------------------
The RAM tier is bounded by bytes. The DISK tier is bounded per acquisition (one generation, the
rest pruned) and NOT across acquisitions: 96 MB per 1536-well plate, and opening fifty plates
leaves fifty of them. That is stated rather than hidden, and the README tells the user where the
folder is and that deleting it is safe.

**Trigger for building a total cap:** anyone reporting the cache folder as a disk problem, or a
deployment where $HOME is small or quota'd. The mechanism would be an LRU over the per-experiment
directories by mtime at startup, which is ~20 lines. It is not built now because a total cap
needs a number, the honest source of a number is a measurement of real usage, and there is none
yet -- inventing one here would be the same mistake ``_budget`` was written to stop.

Rejected here, for the record
-----------------------------
* **A sidecar under the experiment root.** Fastest to implement, and it would write into
  read-only mounts and Dropbox shares. See above.
* **``_recipe.ResultCache``.** RAM only (``max_entries=64``, an item count) and it dies with the
  process, which is precisely the property being asked for here. It is not dead code; its
  consumer is the merged-runs view, not this.
* **PNG, as record-zstack-viewer caches.** A cell is ``(C, 88, 88)`` uint16. PNG cannot hold the
  channel axis without splitting into C files or flattening to RGB, and both lose the thing the
  global recomposite needs. ``.npz`` holds the pixels and the content box in ONE file, which is
  what keeps the publish a single atomic replace.
* **Compressing the cells.** 62 KB of already-downsampled noise per cell; the decode would cost
  more than the read it saves.
"""
from __future__ import annotations

import hashlib
import json
import logging
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

#: Bumped when the on-disk record changes shape. It is part of the TOKEN rather than a field to
#: read back and compare, so entries from an older build are unreachable instead of misparsed --
#: the same property the mtime gives us, applied to our own format.
#:
#: 2 (2026-08-05): a cell is identified by its TIMEPOINT as well as its region. Version 1 cells
#: carry no timepoint and were all written by a producer that read frame 0, so under the new key
#: they are not "t=0 cells" -- they are cells whose timepoint is unknown, and the one thing that
#: must not happen is one of them being served under a label saying t=1. Bumping the version puts
#: every v1 entry behind a token nothing computes any more: unreachable by construction, then
#: DELETED by :meth:`PlateCellCache.prune_stale` on the first publish of the new generation. That
#: is chosen over reading v1 files as t=0 (which would be a guess about a producer's intent, and
#: the guess is only free while it is right) and over a migration pass (code that runs once,
#: is tested never, and re-derives in 0.7 s what it spends itself trying to keep).
FORMAT_VERSION = 2

#: Set to "0" to turn the cache off entirely. For a user who wants a cold read, and for any test
#: that must observe the uncached path.
ENV_ENABLED = "SQUIDXPLORER_PLATE_CACHE"

#: Overrides ``user_cache_dir``. For tests, which must never write into the developer's real
#: cache, and for a user who wants the cells on a faster disk than $HOME.
ENV_DIR = "SQUIDXPLORER_CACHE_DIR"

#: Files that describe the acquisition itself. Their mtime moves when it is rewritten or
#: re-parameterised, and they cost one stat each.
_TOKEN_FILES = ("acquisition.yaml", "coordinates.csv", "acquisition parameters.json",
                "configurations.xml")

#: Ceiling on stats spent building one token, whatever the layout. A plate with more timepoint
#: directories than this hashes their NAMES rather than their mtimes: weaker and bounded, instead
#: of strong and unbounded. The measurement above is the whole reason that trade is made.
MAX_TOKEN_STATS = 64

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


# --- where the cells live ----------------------------------------------------------------------

def cache_root() -> Path:
    """The cache directory for this user. NEVER inside an experiment; see the module docstring."""
    override = os.environ.get(ENV_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("squidxplorer", "cephla"))


def experiment_slug(path) -> str:
    """A short, stable, filesystem-safe namespace for one acquisition path.

    A hash rather than the path itself: an acquisition path holds separators, spaces and possibly
    a study or patient name, and none of those belong in a directory name under $HOME. The last
    path component is kept in front of the digest purely so a human can tell two cache
    directories apart by eye.
    """
    p = str(Path(path).expanduser())
    digest = hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
    stem = _UNSAFE.sub("_", Path(p).name)[:40] or "acq"
    return f"{stem}-{digest}"


def plate_token(path, extra: Sequence = ()) -> str:
    """A staleness token for the WHOLE acquisition, from a bounded number of stats.

    Any change to the acquisition yields a different token, so every entry written under the old
    one becomes unreachable. That is the ported design's central trick: no invalidation pass to
    get wrong, and no lock, because nothing is ever mutated in place.

    *extra* carries everything about the cached RENDERING that must invalidate independently of
    the store: cell size, dtype, channel list. Two windows disagreeing about the channel order
    must not read each other's cells.
    """
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

    # The timepoint directories. Adding a plane moves the CONTAINING directory's mtime, not the
    # root's, so stat-ing only the root would keep serving a plate that has since grown.
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
    """``(mtime_ns, size)`` of *p*, or a marker meaning absent -- absence is also a state.

    A missing file must hash differently from a present one, or deleting ``coordinates.csv``
    would leave a plate serving cells laid out by coordinates that no longer exist.
    """
    try:
        st = os.stat(p)
    except OSError:
        return b"\0absent\0"
    return f"\0{st.st_mtime_ns}:{st.st_size}\0".encode("utf-8")


def _assert_outside_experiment(cache_dir, experiment) -> None:
    """Refuse, loudly, to write anything under the acquisition folder.

    Checked on every publish rather than once at construction. The root comes from an environment
    variable a user can point anywhere, including at the experiment folder, and the promise in
    the README ("never writes into your acquisition folder") is not the kind of promise to keep
    only on the path that happened to be tested.
    """
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
    """The cell's pixels plus the sub-rectangle of the cell they actually cover.

    An ndarray subclass carrying one small value, for the same reason ``_placement.PlacedArray``
    is one: the geometry must not be able to arrive separately from the pixels. ``box`` is
    ``(top, left, h, w)`` in cell pixels, the same shape ``PlateOverview.add_tile`` takes.

    The box is what keeps the CONTRAST rule intact on a cache hit. ``add_tile``'s docstring is
    explicit that the running histogram is fed the TILE and never the store slice, because a
    mosaic cell is zero-padded wherever no FOV lands and those zeros would pin the 1st percentile
    at 0 and wash the whole plate out. Replaying a whole 88x88 cell as one tile would do exactly
    that; replaying ``cell[:, top:top+h, left:left+w]`` with its box does not.
    """

    def __new__(cls, array, box):
        obj = np.asarray(array).view(cls)
        obj.box = tuple(int(v) for v in box)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.box = getattr(obj, "box", None)


# --- the cache -----------------------------------------------------------------------------

#: The process-wide RAM tier, bounded by BYTES (adaptation 1 above). ONE instance, so the bound
#: is a bound on the whole preview path rather than per window: three windows on one plate would
#: otherwise each take the full budget.
_CELLS = MemoryBoundedLRUCache(cache_budget())


def enabled(env: Optional[dict] = None) -> bool:
    """Whether caching is on. Off is a supported state, not a broken one."""
    src = os.environ if env is None else env
    return str(src.get(ENV_ENABLED, "1")).strip().lower() not in ("0", "false", "no", "off")


class PlateCellCache:
    """Two tiers over one acquisition's plate cells: bytes-bounded RAM, then atomic files on disk.

    Layout::

        <user_cache_dir>/cells/<name>-<sha(path)>/<token>/t<t>-<region>-<sha>.npz

    The token directory IS the invalidation mechanism. A changed acquisition produces a new token,
    the old directory becomes unreachable, and :meth:`prune_stale` deletes it the next time a cell
    is published -- the one moment we know a new generation exists and the old one is dead.

    THE TIMEPOINT IS IN THE KEY AND NOT IN THE TOKEN, and that is the whole design of this class's
    2026-08-05 change rather than an implementation detail. In the key, a t=1 cell can never be
    served for t=0 -- the failure this exists to prevent, and the one the loupe's well-keyed coarse
    cache used to commit before it was re-keyed to ``(well, timepoint)``. In the TOKEN it would
    also have been correct, and it would have been useless: the token directory is what
    :meth:`prune_stale` deletes, so stepping t=0 -> t=1 would delete t=0's cells and stepping back
    would re-read the plate. Timepoints coexist under one generation; a changed ACQUISITION still
    invalidates all of them at once.

    Everything here is best effort at the I/O boundary, and loud about it in the log: a cache that
    raised on a full disk would turn a warm-start optimisation into a crash on open. It is NOT
    best effort about correctness. A wrong hit is impossible by construction (the token is in the
    key), and a write outside the cache root raises instead of being swallowed.
    """

    def __init__(self, experiment_path, *, cell_px: int, channels: Sequence[str], dtype,
                 time_point: int = 0, root: Optional[Path] = None) -> None:
        self.experiment = Path(experiment_path).expanduser()
        self.cell_px = int(cell_px)
        self.channels = [str(c) for c in channels]
        self.dtype = np.dtype(dtype)
        #: WHICH TIMEPOINT these cells are of. Fixed for the life of the instance because a
        #: preview pass reads exactly one timepoint; the plate builds a new cache (and a new
        #: worker) when the bar moves, and the RAM tier is process-wide, so stepping back to a
        #: timepoint already seen still HITS.
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
        """The cache for this reader's acquisition, or ``None`` with the reason logged.

        ``None`` rather than raising: every caller is on the window-open path, and an unusable
        cache must degrade to the uncached read the user had yesterday, not to a window that
        will not open. The reason is logged, so "it is not caching" is diagnosable rather than
        merely felt.
        """
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
            # os.replace makes a torn file impossible, so this is a damaged disk or a foreign
            # file. Drop it and read the acquisition instead: a cache must never be the reason
            # pixels cannot be shown.
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
        """Publish this region's cell. Atomic; returns whether it reached disk.

        The temp file is written BESIDE the destination (same directory, therefore the same
        filesystem) because ``os.replace`` is atomic only within one; a temp in ``/tmp`` would
        silently degrade to a copy across a mount boundary, which is the torn read this design
        exists to prevent.
        """
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
            # The temp goes with it. A publish that did not happen must leave NOTHING behind: a
            # stranded temp is invisible to `get` (it is not the destination name) but it is real
            # bytes under $HOME that nothing will ever collect.
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
    #
    # Adopted from ``ndviewer_hcs/plate_stack.py`` (maragall/ndviewer), which is the tool in the
    # reference video and therefore the shape whose feel is being chased. Its
    # ``PlateStackManager`` keeps "pre-computed Z x T plate assemblies stored as multi-page TIFF
    # ... memory-mapped for efficient random access", and ``get_page(t_idx, z_idx)`` is an index
    # into that memmap rather than a fuse. That is WHY scrubbing feels continuous there, and it is
    # a different granularity from one file per well.
    #
    # So this cache does both, in the order a log-structured store does: publish per WELL while
    # the preview streams, because a preview that is interrupted must still leave the wells it
    # finished; then COMPACT the generation into one memory-mapped page when the pass completes,
    # because that is what a reopen and a coarse tile actually read. Measured on sim_1536wp, the
    # per-well form costs 0.261 s to replay 1536 wells and 1.59 s to seed the coarse rungs, and
    # both are 1536 file opens. The pack turns each into one open plus slices.
    #
    # WHERE t SLOTTED IN (2026-08-05). This block used to say ours "cannot express t at all" and
    # describe the leading axis it would grow. It grew a FILE instead: one page per timepoint,
    # ``plate-cells-t{t}.npy``, rather than one ``(t, region, C, cell, cell)`` page. The producer
    # decides that. ``_PreviewWorker`` walks ONE timepoint per pass, so a single page over all t
    # could only be published complete by something that walked every timepoint -- which nothing
    # does, and which would make the plate's first paint wait on frames nobody asked to see. A
    # page per t is publishable the moment its own pass ends, which is the property ``pack`` is
    # built on ("only called when a preview pass finished"). The sidecar's ``"t"`` stopped being
    # the placeholder 0 and became this cache's timepoint, and ``_pack_index`` checks it.
    # z still needs nothing: the preview is one representative plane by definition, and a
    # z-resolved plate page is a different product (theirs), not a different cache.

    #: The compacted page and its sidecar, PER TIMEPOINT. The array is a plain ``.npy`` precisely
    #: so it can be memory-mapped; an ``.npz`` cannot be, and holding a 96 MB page in RAM to serve
    #: 88x88 slices out of it would be the opposite of the point.
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
        """Compact this generation into one memory-mapped page. False if it is not complete.

        Only called when a preview pass finished, so a partial plate is never compacted into a
        page that claims to be the plate. The per-well files are removed afterwards: the page
        supersedes them byte for byte, and keeping both would double 96 MB per 1536-well plate.

        Two files, published array first and index second, and the index is the commit marker.
        No lock: two windows compacting the same generation write the SAME bytes, because the
        content is a pure function of the token, so a torn interleaving is not a torn result.
        """
        regions = [str(r) for r in regions]
        if not regions:
            return False
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
        """One region's cell out of the memory-mapped page, or None if there is no page.

        The slice is a VIEW of the mapping, so a 96 MB page never enters this process's heap and
        the operating system's page cache does the bounding -- which is a better-informed bound
        than ours, because it can see the rest of the machine. The RAM tier above is deliberately
        NOT populated from here: that would copy the page back in, one cell at a time.
        """
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
            # The timepoint is already in the FILE NAME, so this can only fire on a hand-edited or
            # foreign sidecar. Checked anyway: serving another timepoint's page is the precise
            # failure this whole re-keying exists to make impossible, and a name is a convention
            # while the sidecar is the record.
            if int(raw.get("t", 0)) != self.time_point:
                return None
            raw["at"] = {r: i for i, r in enumerate(raw["regions"])}
            self._index = raw
        return self._index

    # ---- housekeeping -------------------------------------------------------------------
    def prune_stale(self) -> int:
        """Delete this acquisition's OTHER token directories. Returns how many went.

        Without this the token-in-the-key design never frees anything: a store that changes ten
        times leaves ten full generations of every cell under $HOME. Run on first publish, which
        is when a new generation is known to exist and the previous one is known to be dead.

        A GENERATION IS NOT A TIMEPOINT. This deletes other TOKEN directories, and the timepoint
        is deliberately not in the token, so every timepoint of the live acquisition lives in one
        directory and stepping t=0 -> t=1 keeps t=0's cells. That is what makes stepping back a
        cache hit instead of a re-read; see the class docstring. It also means this is the pass
        that finally removes the pre-2026-08-05 (FORMAT_VERSION 1) cells, which are under a token
        nothing computes any more.
        """
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
        """One file per region PER TIMEPOINT. The digest suffix makes any region id safe AND unique.

        Sanitising alone is not enough: ``A/1`` and ``A_1`` sanitise to the same name, and one
        well would then serve the other's pixels. The ``t`` prefix is the same argument one axis
        over: two timepoints of one well are two different pictures and must be two files.
        """
        r = str(region)
        clean = _UNSAFE.sub("_", r)[:40] or "region"
        digest = hashlib.sha256(r.encode("utf-8")).hexdigest()[:8]
        return self.dir / f"t{self.time_point}-{clean}-{digest}.npz"

    def _ram_key(self, region: str) -> tuple:
        """The process-wide RAM tier's key. ``(token, t, region)``: see the class docstring.

        One function rather than the tuple written at each site, because ``get`` and ``put`` are
        the two halves of one identity and a key that drifted between them would present as a
        cache that simply never hits -- slow, correct, and nearly invisible.
        """
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
