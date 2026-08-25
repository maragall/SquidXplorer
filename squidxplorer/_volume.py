"""Big-array allocation for per-plane 3-D fusion."""

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
    """Directory the scratch file is created in; never the acquisition folder, which is read-only."""
    return Path(os.environ.get(_SCRATCH_ENV) or tempfile.gettempdir())


class InsufficientScratchSpaceError(OSError):
    """Refusing to spill a fused volume that would not fit in the scratch directory."""


def allocate(shape: Sequence[int], dtype, *, threshold: Optional[int] = None,
             what: str = "fused volume") -> np.ndarray:
    """A zero-filled array of *shape*, spilled to a scratch file when it is too big for RAM."""
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
        posix_unlinked = os.name != "nt"
        if posix_unlinked:
            os.unlink(path)
        arr = np.memmap(handle, dtype=dtype, mode="w+", shape=shape)
    except BaseException:
        handle.close()
        if not posix_unlinked:
            os.unlink(path)
        raise
    _log.info("Fusion: %s is %.2f GB - backing it with a scratch file in %s (RAM limit %.2f GB); "
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
    """Flush *array*'s dirty pages and drop them from the resident set. True if it did anything."""
    m = _backing_mmap(array)
    if m is None:
        return False
    m.flush()
    try:
        m.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError, ValueError):
        return False
    return True
