"""IMA-230 storage guard: stop before the disk overflows, and leave an honest artifact.

A large plate run can exhaust the output disk part-way through. Without a guard that surfaces as a
raw error thrown from inside a tensorstore writer thread, after a partial array is already on disk —
an unreadable traceback plus a plate that *looks* finished but silently isn't.

This module supplies the measurement primitives; ``_output.write_from_stream`` owns the policy.

Two-tier estimate
-----------------
Output size is NOT analytically predictable: the store is blosc-zstd (``_zarr_store.py``) and
microscopy compresses by wildly different factors — sparse fluorescence squeezes down hard, dense
brightfield barely at all. A fixed assumed ratio is wrong in both directions, and a guard that cries
wolf gets disabled by the operator.

Squid's acquisition software solves this by writing one synthetic test image to a tmpdir and
measuring the delta (``control/core/multi_point_controller.py``) because its data doesn't exist yet.
SquidMIP is a *reducer*, so it is in a better position: it can measure the bytes it actually wrote.

    pre-flight   uncompressed UPPER BOUND (pure arithmetic, no I/O)
                 -> reject only when even ONE field cannot fit
                 -> otherwise report bound vs free and proceed

    mid-stream   running byte counter (:class:`WrittenBytes`)
                 -> _write_one sums st_blocks*512 of the files it just created
                 -> est per field = total / fields_completed

Why a counter and not ``os.walk``: ``plate.ome.zarr`` and ``tiff/`` are SIBLINGS under *out_dir*
(see ``_output.write_from_stream``), so walking the plate directory would miss the uncompressed TIFF
copy entirely and under-reserve by 2-5x on exactly the ``--tiff`` configuration that eats the most
disk. A per-field walk would also be an O(n^2) stat storm — ~140 chunk files per field x 1536 wells
re-walked every field, competing with the writer threads for a nearly-full disk. ``_write_one``
already knows precisely which files it created, so summing them inline is exact and O(files written).

``st_blocks * 512`` rather than ``st_size`` because zarr chunks are many small compressed blobs and
block rounding makes real consumption exceed apparent size.

Out-of-space detection (measured, not assumed — IMA-230 T0 spike, 2026-07-20)
----------------------------------------------------------------------------
Filling a real 11 MB filesystem showed the two writers fail in genuinely different ways, and NEITHER
carries a usable ``errno``::

    tensorstore   ValueError            errno absent
                  "RESOURCE_EXHAUSTED: ... No space left on device ... os_error_code='28'"
    tifffile      OSError, errno=None   "9000000 requested and 6123392 written"

So a test injecting ``OSError(errno.ENOSPC)`` would confirm a translation that never fires in
production. :func:`is_out_of_space` matches on all three shapes: a genuine ``errno == ENOSPC`` (plain
file writes such as the metadata ``write_text`` do produce it), the tensorstore message, and
tifffile's short-write message.

Note also that "stop before the write that would fail" is not achievable at field granularity —
tensorstore commits chunk-by-chunk with no reservation primitive. The guard only reduces the
probability, which is why :func:`is_out_of_space` is load-bearing rather than a nicety.
"""

from __future__ import annotations

import errno as _errno
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# Mirrors _output._PYRAMID_MIN_YX / _PYRAMID_MAX_LEVELS. Duplicated rather than imported to keep this
# module free of a circular import (_output imports us); the pyramid test asserts they agree.
_PYRAMID_MIN_YX = 256
_PYRAMID_MAX_LEVELS = 6

_BYTES_PER_BLOCK = 512  # st_blocks is defined in 512-byte units regardless of the fs block size

# Non-image sidecars (zarr.json at plate/row/well/field level). Small and roughly constant; a little
# slack here is free insurance against many-well plates where the group JSON is not negligible.
_METADATA_SLACK_BYTES = 256 * 1024


class InsufficientDiskSpace(RuntimeError):
    """The output disk cannot safely hold (the rest of) this run.

    Raised by :func:`preflight` before anything is written, and by
    ``_output.write_from_stream`` mid-stream — either predicted by the guard or translated from a
    real out-of-space error that beat it (see :func:`is_out_of_space`).

    Deliberately NOT signalled through ``write_from_stream``'s ``stop=`` predicate: that parameter is
    already owned by the GUI (``_viewer._PlateWorker``) and means "the operator clicked cancel", and
    the viewer reads the same flag to decide the run ended normally. Reusing it would make a
    disk-full abort indistinguishable from a cancel — the exact silent failure this ticket exists to
    eliminate. A distinct exception type cannot be misread by any caller, present or future.
    """

    def __init__(
        self,
        *,
        bytes_free: int,
        bytes_needed: int,
        path,
        fields_written: int = 0,
        truncated: bool = False,
        detail: Optional[str] = None,
    ):
        self.bytes_free = int(bytes_free)
        self.bytes_needed = int(bytes_needed)
        self.path = str(path)
        self.fields_written = int(fields_written)
        self.truncated = bool(truncated)
        self.detail = detail
        super().__init__(self._message())

    def _message(self) -> str:
        need_mb = self.bytes_needed / 1024 / 1024
        free_mb = self.bytes_free / 1024 / 1024
        msg = (
            f"not enough disk space at '{self.path}': need ~{need_mb:,.0f} MB, "
            f"only {free_mb:,.0f} MB available"
        )
        if self.fields_written:
            msg += (
                f". Stopped after writing {self.fields_written:,} field(s); the output plate has "
                "been TRUNCATED to exactly what was written (its metadata is consistent with what "
                "is on disk)"
            )
        else:
            msg += ". Nothing was written"
        if self.detail:
            msg += f" [{self.detail}]"
        return msg


# --- out-of-space detection ------------------------------------------------------------------

# tensorstore surfaces an absl status as a ValueError whose text carries the OS error code; tifffile
# raises OSError with errno=None and a short-write message. Both measured in the T0 spike.
_TS_NO_SPACE_RE = re.compile(
    r"no space left on device|RESOURCE_EXHAUSTED|os_error_code='28'", re.IGNORECASE
)
_TIFF_SHORT_WRITE_RE = re.compile(r"(\d+) requested and (\d+) written")


def is_out_of_space(exc: BaseException) -> bool:
    """True if *exc* is a disk-full failure from any writer in this pipeline.

    Handles all three shapes measured in the T0 spike (see module docstring): a genuine
    ``errno == ENOSPC``, tensorstore's ``ValueError`` carrying ``os_error_code='28'``, and tifffile's
    short-write ``OSError``. Do NOT simplify this to an ``errno`` check — the two shapes that matter
    most in practice have no usable ``errno``.
    """
    if isinstance(exc, OSError) and exc.errno == _errno.ENOSPC:
        return True
    text = str(exc)
    if _TS_NO_SPACE_RE.search(text):
        return True
    m = _TIFF_SHORT_WRITE_RE.search(text)
    if m and isinstance(exc, OSError) and int(m.group(2)) < int(m.group(1)):
        return True
    return False


# --- measurement -----------------------------------------------------------------------------

def free_bytes(path) -> int:
    """Bytes available on the filesystem that would hold *path*, resolving to the nearest EXISTING
    ancestor.

    The output directory does not exist at pre-flight time — the CLI computes
    ``out_parent / f"{name}.hcs"`` and the writer creates it lazily. Squid's
    ``control/utils.get_available_disk_space`` *raises* on a missing directory, so it cannot be used
    as-is here; this walks up instead.

    Kept path-scoped on purpose: free space is a per-filesystem number, never a cached global.
    ``--output-folder`` can put the output on a different device from the input, and the deferred
    input-eviction work needs the figure for both.
    """
    p = Path(path).expanduser()
    for candidate in (p, *p.parents):
        if candidate.exists():
            return int(shutil.disk_usage(candidate).free)
    # Only reachable if even the filesystem root is missing (e.g. a bad Windows drive letter).
    raise ValueError(f"cannot determine free space for {path!r}: no existing ancestor directory")


def file_bytes(path) -> int:
    """Real disk consumption of one file (``st_blocks * 512``), 0 if it vanished.

    ``st_size`` understates: zarr chunks are many small compressed blobs and every one is rounded up
    to a filesystem block. Falls back to ``st_size`` on platforms without ``st_blocks`` (Windows).
    """
    try:
        st = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return 0
    blocks = getattr(st, "st_blocks", None)
    return int(blocks) * _BYTES_PER_BLOCK if blocks is not None else int(st.st_size)


def tree_bytes(path) -> int:
    """Real disk consumption of a directory tree, skipping symlinks.

    Test/diagnostic helper only — the guard uses :class:`WrittenBytes` instead. Walking the output
    per field would be an O(n^2) stat storm on a full plate (see module docstring).
    """
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return file_bytes(root)
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if not os.path.islink(fp):
                total += file_bytes(fp)
    return total


class WrittenBytes:
    """Thread-safe running total of bytes actually written, and how many fields produced them.

    ``_write_one`` runs on several writer threads at once, so both counters move under one lock.

    The mean is over COMPLETED FIELDS, not "the first one": ``write_from_stream`` harvests results
    with ``wait(FIRST_COMPLETED)``, which only fires once the pool is saturated, so several fields
    are already on disk when the first result arrives. Dividing by one would overestimate by up to
    the writer count — precisely the cry-wolf failure this design exists to avoid.
    """

    __slots__ = ("_lock", "_total", "_fields")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._fields = 0

    def record_field(self, paths) -> int:
        """Add the on-disk size of every path in *paths* as one completed field. Returns its bytes."""
        n = sum(file_bytes(p) for p in paths)
        with self._lock:
            self._total += n
            self._fields += 1
        return n

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @property
    def fields(self) -> int:
        with self._lock:
            return self._fields

    def per_field(self) -> Optional[int]:
        """Measured mean bytes per field, or None while no field has completed yet."""
        with self._lock:
            if self._fields <= 0:
                return None
            return int(self._total / self._fields)


# --- analytic upper bound --------------------------------------------------------------------

def pyramid_factor(frame_shape) -> float:
    """Total pyramid bytes as a multiple of level 0, mirroring ``_output._pyramid``.

    Each level halves Y and X (so a quarter of the pixels) until the coarsest fits
    ``_PYRAMID_MIN_YX`` or the level cap is hit. A small field yields exactly 1.0 (level 0 only),
    and a large one approaches 4/3.
    """
    y, x = int(frame_shape[-2]), int(frame_shape[-1])
    total, levels = 1.0, 1
    cy, cx = y, x
    while max(cy, cx) > _PYRAMID_MIN_YX and levels < _PYRAMID_MAX_LEVELS:
        cy = max(1, cy // 2) if cy >= 2 else cy
        cx = max(1, cx // 2) if cx >= 2 else cx
        total += (cy * cx) / (y * x)
        levels += 1
    return total


def estimate_field_bytes(metadata: dict, *, tiff: bool = False) -> int:
    """Uncompressed UPPER BOUND for one field ``(T, C, 1, Y, X)``, pyramid included.

    An upper bound by construction: the store is compressed, so the real figure is smaller — usually
    much smaller. Used for the pre-flight floor and as the conservative stand-in for the first few
    fields, before :class:`WrittenBytes` has any data.

    With *tiff*, adds the individual per-plane TIFF export: one uncompressed ``(Y, X)`` plane per
    channel per timepoint, with no pyramid. That tree lives beside the plate, not inside it.
    """
    frame = metadata.get("frame_shape") or (1, 1)
    dtype = np.dtype(metadata.get("dtype", np.uint16))
    n_t = max(1, int(metadata.get("n_t", 1) or 1))
    n_c = max(1, len(metadata.get("channels") or [1]))
    plane = int(frame[-2]) * int(frame[-1]) * dtype.itemsize
    level0 = plane * n_t * n_c
    total = level0 * pyramid_factor(frame)
    if tiff:
        total += level0  # uncompressed, level 0 only
    return int(total)


def estimate_total_bytes(metadata: dict, n_fields: int, *, tiff: bool = False) -> int:
    """Uncompressed upper bound for a whole run of *n_fields* fields, plus metadata slack."""
    return int(estimate_field_bytes(metadata, tiff=tiff) * max(0, int(n_fields))) + _METADATA_SLACK_BYTES


# --- pre-flight ------------------------------------------------------------------------------

def preflight(
    out_dir,
    metadata: dict,
    n_fields: int,
    *,
    tiff: bool = False,
    min_free_bytes: int = 0,
) -> dict:
    """Check the output disk before anything is written.

    Rejects ONLY when the disk cannot safely hold even a single field
    (``free < min_free_bytes + one_field_bound``). A failing *total* bound does NOT reject the run:
    the bound is uncompressed, and compression may well save it — refusing there would be the
    cry-wolf failure. The caller is expected to surface ``bound`` vs ``free`` prominently when
    ``fits_uncompressed`` is False, since that is the operator's most actionable signal.

    Returns a report dict; raises :class:`InsufficientDiskSpace` if even one field cannot fit.
    """
    free = free_bytes(out_dir)
    per_field = estimate_field_bytes(metadata, tiff=tiff)
    total = estimate_total_bytes(metadata, n_fields, tiff=tiff)
    floor = int(min_free_bytes) + per_field
    if free < floor:
        raise InsufficientDiskSpace(
            bytes_free=free,
            bytes_needed=floor,
            path=out_dir,
            detail="cannot fit even one field above the reserve",
        )
    return {
        "free_bytes": free,
        "per_field_bytes": per_field,
        "bound_bytes": total,
        "min_free_bytes": int(min_free_bytes),
        "fits_uncompressed": free - int(min_free_bytes) >= total,
    }
