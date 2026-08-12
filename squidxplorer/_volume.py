"""Big-array allocation for per-plane 3-D fusion (IMA-277).

WHY THIS EXISTS. Stitching used to fuse with ``z`` pinned to 1, so a region operator's result
was one fused plane: ~0.88 GB for the 10x tissue set's 27-FOV well at 4 channels. Per-plane
fusion multiplies that by the stack depth — 8.79 GB at 10 planes, measured on that same well —
and the machine this runs on has 16 GB. A plain ``np.zeros`` of the result therefore does not
merely use a lot of memory, it does not fit alongside the tiles being fused into it and the
pyramid the writer builds out of it.

The z-outer loop in :func:`squidxplorer._stitch.stitch_region` bounds the INPUT side (one z of tiles
resident, ~0.94 GB, exactly what it was before). This module bounds the OUTPUT side, and it does
it without changing anybody's contract: :func:`allocate` returns a real ``np.ndarray``, so the
writer, the viewer's ``_on_well``, ``_write_tiffs`` and every test are untouched. Above a
threshold that array is simply backed by a scratch FILE instead of anonymous memory.

WHAT THAT BUYS, precisely. A file-backed ``MAP_SHARED`` page can be written back and dropped; an
anonymous page can only be swapped. So after each z plane is fused, :func:`release` msyncs and
``MADV_DONTNEED``s the mapping and the pages go away for real — resident set stays at roughly
one plane's worth of tiles plus one plane of output, flat in stack depth. Without it the process
either swaps or dies. Both halves are measured in tests/test_stitch_zplanes.py.

WHERE THE SCRATCH FILE GOES, and where it must NOT. Never the acquisition folder: the owner's
invariant is that nothing writes into the data, "specially the tiffs". It goes to the system
temp directory (``SQUIDXPLORER_SCRATCH_DIR`` overrides), is unlinked the moment it is mapped on
POSIX so the kernel reclaims it even if this process is killed, and is closed with the array.
"""

from __future__ import annotations

import mmap
import os
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from squidxplorer._logpane import get_logger

_log = get_logger("volume")

# Above this, the result is backed by a scratch file rather than anonymous RAM. 2 GB is chosen
# against the numbers this path actually produces: a single fused plane of the 10x tissue set is
# 0.88 GB (stays in RAM, so every z-REDUCER — mip, reference, decon3d — is byte-for-byte and
# allocation-for-allocation what it has always been), while the 10-plane volume is 8.79 GB and
# cannot be. Overridable so a big-memory machine can turn spilling off with a huge value.
_SPILL_BYTES = 2 * 1024 ** 3
_SPILL_ENV = "SQUIDXPLORER_SPILL_BYTES"
_SCRATCH_ENV = "SQUIDXPLORER_SCRATCH_DIR"


def spill_threshold() -> int:
    """Bytes above which :func:`allocate` spills to a scratch file. Env-overridable."""
    raw = os.environ.get(_SPILL_ENV)
    if raw is None:
        return _SPILL_BYTES
    try:
        return int(float(raw))
    except ValueError:
        return _SPILL_BYTES


def scratch_dir() -> Path:
    """Directory the scratch file is created in. NEVER the acquisition folder — see the module
    docstring; the acquisition is read-only as far as this package is concerned."""
    return Path(os.environ.get(_SCRATCH_ENV) or tempfile.gettempdir())


class InsufficientScratchSpaceError(OSError):
    """Refusing to spill a fused volume that would not fit in the scratch directory."""


def allocate(shape: Sequence[int], dtype, *, threshold: Optional[int] = None,
             what: str = "fused volume") -> np.ndarray:
    """A zero-filled array of *shape*, spilled to a scratch file when it is too big for RAM.

    Returns a plain ``np.ndarray`` either way (a ``np.memmap`` IS one), so no caller can tell
    the difference except through :func:`release`, which is a no-op on the RAM case.
    """
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    limit = spill_threshold() if threshold is None else int(threshold)
    if nbytes <= limit:
        return np.zeros(shape, dtype=dtype)

    directory = scratch_dir()
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if nbytes >= free:
        raise InsufficientScratchSpaceError(
            f"refusing to start: the {what} is {nbytes / 1024 ** 3:.2f} GB, too big for RAM "
            f"(> {limit / 1024 ** 3:.2f} GB), and {directory} has only "
            f"{free / 1024 ** 3:.2f} GB free to spill it to. Free space, point "
            f"{_SCRATCH_ENV} at a roomier disk, or fuse fewer channels / z-planes."
        )

    fd, path = tempfile.mkstemp(prefix="squidxplorer-fuse-", suffix=".bin", dir=str(directory))
    try:
        handle = open(fd, "w+b")
    except BaseException:
        os.close(fd)
        os.unlink(path)
        raise
    try:
        # Unlink NOW on POSIX: the mapping keeps the inode alive, so the space is reclaimed by
        # the kernel the moment this process ends — including a kill -9, which a finaliser
        # cannot survive. Windows cannot unlink an open file, so it gets an explicit finaliser.
        posix_unlinked = os.name != "nt"
        if posix_unlinked:
            os.unlink(path)
        arr = np.memmap(handle, dtype=dtype, mode="w+", shape=shape)
    except BaseException:
        handle.close()
        if not posix_unlinked:
            os.unlink(path)
        raise
    _log.info("Fusion: %s is %.2f GB — backing it with a scratch file in %s (RAM limit %.2f GB); "
              "resident pages are released after every z plane.",
              what, nbytes / 1024 ** 3, directory, limit / 1024 ** 3)

    def _cleanup(_h=handle, _p=path, _unlinked=posix_unlinked):
        try:
            _h.close()
        finally:
            if not _unlinked:
                try:
                    os.unlink(_p)
                except OSError:
                    pass

    weakref.finalize(arr, _cleanup)
    return arr


def _backing_mmap(array) -> Optional[mmap.mmap]:
    """The ``mmap`` object behind *array*, walking ``.base`` (views, ndarray subclasses)."""
    seen = 0
    obj = array
    while obj is not None and seen < 8:
        m = getattr(obj, "_mmap", None)
        if isinstance(m, mmap.mmap):
            return m
        obj = getattr(obj, "base", None)
        seen += 1
    return None


def release(array) -> bool:
    """Flush *array*'s dirty pages and drop them from the resident set. True if it did anything.

    A no-op (returning False) for a RAM-backed array — there is nothing to release, and a caller
    should not have to know which kind it got. For a spilled one this is the whole point: msync
    then ``MADV_DONTNEED``, so a 10-plane fusion's resident set is one plane, not ten.
    """
    m = _backing_mmap(array)
    if m is None:
        return False
    m.flush()
    try:
        m.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError, ValueError):
        return False        # platform without MADV_DONTNEED: the flush still happened
    return True
