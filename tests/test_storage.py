"""IMA-230 storage guard primitives — unit tests.

Free space is injected by monkeypatching ``shutil.disk_usage``; the guard's policy tests live in
tests/test_output.py. The out-of-space DETECTION tests are pinned to the exact strings measured in
the T0 spike against a real 11 MB filesystem (see squidmip/_storage module docstring) — that is the
whole point of the spike, so these must not be relaxed to a generic "errno" check.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import numpy as np
import pytest

from squidmip._storage import (
    InsufficientDiskSpace,
    WrittenBytes,
    estimate_field_bytes,
    estimate_total_bytes,
    file_bytes,
    free_bytes,
    is_out_of_space,
    preflight,
    pyramid_factor,
    tree_bytes,
)

# --- the exact exceptions a full disk produces (T0 spike, 2026-07-20) -------------------------

# tensorstore: a ValueError with NO errno attribute at all.
TS_REAL = (
    "RESOURCE_EXHAUSTED: Error writing local file \"/tmp/x/big.zarr/c/0/0/0/1/2\": "
    "Failed writing: \"/tmp/x/big.zarr/c/0/0/0/1/2.__lock\": Failed to write 950078 bytes to "
    "file 6: No space left on device [source locations='tensorstore/internal/os/"
    "file_util_posix.cc:380'] [os_error_code='28']"
)
# tifffile: an OSError whose errno is None.
TIFF_REAL = "9000000 requested and 6123392 written"


def test_detects_real_tensorstore_out_of_space():
    # A ValueError, not an OSError, and no errno — an errno-only check would miss this entirely.
    exc = ValueError(TS_REAL)
    assert not isinstance(exc, OSError)
    assert getattr(exc, "errno", None) is None
    assert is_out_of_space(exc)


def test_detects_real_tifffile_short_write():
    exc = OSError(TIFF_REAL)
    assert exc.errno is None  # tifffile does not populate errno
    assert is_out_of_space(exc)


def test_detects_genuine_enospc_errno():
    # Plain file writes (the metadata write_text) do produce a real errno.
    assert is_out_of_space(OSError(errno.ENOSPC, "No space left on device"))


def test_does_not_misfire_on_unrelated_errors():
    assert not is_out_of_space(ValueError("channel/axis mismatch"))
    assert not is_out_of_space(OSError(errno.EACCES, "Permission denied"))
    assert not is_out_of_space(RuntimeError("projector 'mip' is not callable"))
    # A COMPLETE tifffile write must not read as a failure.
    assert not is_out_of_space(OSError("9000000 requested and 9000000 written"))


# --- free_bytes ------------------------------------------------------------------------------

def test_free_bytes_resolves_nearest_existing_ancestor(tmp_path):
    """The output dir does not exist at pre-flight time — the CLI builds the path and the writer
    creates it lazily. Squid's helper raises here; ours must walk up."""
    missing = tmp_path / "acq.hcs" / "plate.ome.zarr" / "deeper"
    assert not missing.exists()
    assert free_bytes(missing) > 0
    assert free_bytes(missing) == free_bytes(tmp_path)


def test_free_bytes_on_existing_dir(tmp_path):
    assert free_bytes(tmp_path) > 0


def test_free_bytes_reflects_monkeypatched_disk_usage(tmp_path, monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(
        _shutil, "disk_usage", lambda p: type("U", (), {"total": 100, "used": 90, "free": 10})()
    )
    assert free_bytes(tmp_path) == 10


# --- file/tree measurement -------------------------------------------------------------------

def test_file_bytes_counts_block_allocation_not_apparent_size(tmp_path):
    """st_blocks*512, so many tiny zarr chunks are not undercounted."""
    p = tmp_path / "tiny.bin"
    p.write_bytes(b"x")
    n = file_bytes(p)
    assert n > 0
    if hasattr(os.stat(p), "st_blocks"):
        assert n >= 512  # one byte still occupies a whole block


def test_file_bytes_missing_file_is_zero(tmp_path):
    assert file_bytes(tmp_path / "nope") == 0


def test_tree_bytes_sums_and_skips_symlinks(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_bytes(b"a" * 4096)
    (tmp_path / "d" / "b").write_bytes(b"b" * 4096)
    real = tree_bytes(tmp_path)
    assert real >= 8192
    try:
        os.symlink(tmp_path / "d" / "a", tmp_path / "d" / "link")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert tree_bytes(tmp_path) == real  # the symlink adds nothing


def test_tree_bytes_empty_and_missing(tmp_path):
    assert tree_bytes(tmp_path / "nothing") == 0
    (tmp_path / "empty").mkdir()
    assert tree_bytes(tmp_path / "empty") == 0


# --- WrittenBytes ----------------------------------------------------------------------------

def test_written_bytes_has_no_estimate_before_any_field():
    """The first few submits happen before any field completes — the guard must know it has no
    measurement yet and fall back to the analytic bound."""
    assert WrittenBytes().per_field() is None


def test_written_bytes_means_over_completed_fields_not_over_one(tmp_path):
    """wait(FIRST_COMPLETED) only fires once the pool saturates, so several fields are on disk when
    the first result arrives. Dividing by 1 would overestimate by up to the writer count."""
    wb = WrittenBytes()
    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}"
        p.write_bytes(b"x" * 4096)
        paths.append(p)
    for p in paths:
        wb.record_field([p])
    assert wb.fields == 4
    assert wb.per_field() == pytest.approx(wb.total / 4, rel=0.01)


def test_written_bytes_is_thread_safe():
    import threading

    wb = WrittenBytes()

    def worker():
        for _ in range(200):
            wb.record_field([])  # 0 bytes, but still one field each

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wb.fields == 1600  # no lost updates


def test_written_bytes_spans_multiple_trees(tmp_path):
    """plate.ome.zarr and tiff/ are SIBLINGS; a field's cost is both, so record_field takes paths
    from both trees. This is the bug a plate-directory walk would have."""
    plate = tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "0"
    tiffs = tmp_path / "tiff" / "0"
    plate.parent.mkdir(parents=True)
    tiffs.mkdir(parents=True)
    a, b = plate.parent / "chunk", tiffs / "B2_0_0_ch.tiff"
    a.write_bytes(b"z" * 8192)
    b.write_bytes(b"t" * 8192)
    wb = WrittenBytes()
    n = wb.record_field([a, b])
    assert n == file_bytes(a) + file_bytes(b)
    assert n >= 16384


# --- analytic bound --------------------------------------------------------------------------

def _meta(y=8, x=8, n_t=1, n_c=2, dtype=np.uint16):
    return {
        "frame_shape": (y, x),
        "dtype": np.dtype(dtype),
        "n_t": n_t,
        "channels": [{"name": f"c{i}"} for i in range(n_c)],
    }


def test_pyramid_factor_is_one_for_small_fields():
    # <= _PYRAMID_MIN_YX means level 0 only, matching _output._pyramid.
    assert pyramid_factor((8, 8)) == 1.0
    assert pyramid_factor((256, 256)) == 1.0


def test_pyramid_factor_approaches_four_thirds_for_large_fields():
    f = pyramid_factor((4168, 4168))
    assert 1.3 < f < 1.34  # geometric 1 + 1/4 + 1/16 + ...


def test_estimate_field_bytes_matches_hand_arithmetic():
    # 8*8 px * 2 bytes * 1 t * 2 ch, no pyramid at this size
    assert estimate_field_bytes(_meta()) == 8 * 8 * 2 * 1 * 2


def test_estimate_scales_with_timepoints_channels_and_dtype():
    base = estimate_field_bytes(_meta())
    assert estimate_field_bytes(_meta(n_t=3)) == base * 3
    assert estimate_field_bytes(_meta(n_c=4)) == base * 2
    assert estimate_field_bytes(_meta(dtype=np.uint8)) == base // 2


def test_tiff_adds_an_uncompressed_level_zero_copy():
    """--tiff is a second uncompressed copy; the CLI docstring says it roughly doubles on-disk size.
    The estimate must reflect it or the guard under-reserves on the heaviest configuration."""
    without = estimate_field_bytes(_meta(), tiff=False)
    with_tiff = estimate_field_bytes(_meta(), tiff=True)
    assert with_tiff == without * 2  # no pyramid at this size, so exactly double


def test_estimate_total_scales_with_field_count():
    one = estimate_total_bytes(_meta(), 1)
    ten = estimate_total_bytes(_meta(), 10)
    assert ten > one
    assert ten - one == estimate_field_bytes(_meta()) * 9


def test_estimate_tolerates_metadata_without_shape_keys():
    """tests/test_output.py fabricates metadata with no frame_shape/dtype/n_t — must not explode."""
    assert estimate_field_bytes({"channels": [{"name": "a"}]}) > 0


# --- preflight -------------------------------------------------------------------------------

def _fake_free(monkeypatch, n):
    import shutil as _shutil

    monkeypatch.setattr(
        _shutil, "disk_usage", lambda p: type("U", (), {"total": n * 4, "used": n * 3, "free": n})()
    )


def test_preflight_rejects_when_one_field_cannot_fit(tmp_path, monkeypatch):
    _fake_free(monkeypatch, 10)  # 10 bytes free
    with pytest.raises(InsufficientDiskSpace) as ei:
        preflight(tmp_path / "out.hcs", _meta(), 100, min_free_bytes=0)
    assert ei.value.bytes_free == 10
    assert ei.value.fields_written == 0
    assert "Nothing was written" in str(ei.value)


def test_preflight_rejects_when_below_the_reserve(tmp_path, monkeypatch):
    per_field = estimate_field_bytes(_meta())
    _fake_free(monkeypatch, per_field + 5)  # room for the field but not the reserve
    with pytest.raises(InsufficientDiskSpace):
        preflight(tmp_path / "out.hcs", _meta(), 1, min_free_bytes=1_000)


def test_preflight_does_not_reject_merely_because_the_total_bound_exceeds_free(tmp_path, monkeypatch):
    """The bound is UNCOMPRESSED; compression may still save the run. Rejecting here would be the
    cry-wolf failure that gets the guard disabled. It must report, not refuse."""
    per_field = estimate_field_bytes(_meta())
    _fake_free(monkeypatch, per_field * 3)
    report = preflight(tmp_path / "out.hcs", _meta(), 1000, min_free_bytes=0)
    assert report["fits_uncompressed"] is False  # far too small for the bound...
    assert report["bound_bytes"] > report["free_bytes"]  # ...and it says so, without raising


def test_preflight_reports_fitting_run_as_provably_safe(tmp_path, monkeypatch):
    _fake_free(monkeypatch, 10 * 1024 * 1024 * 1024)
    report = preflight(tmp_path / "out.hcs", _meta(), 10, min_free_bytes=0)
    assert report["fits_uncompressed"] is True
    assert report["per_field_bytes"] > 0


def test_preflight_writes_nothing_on_rejection(tmp_path, monkeypatch):
    out = tmp_path / "out.hcs"
    _fake_free(monkeypatch, 10)
    with pytest.raises(InsufficientDiskSpace):
        preflight(out, _meta(), 100)
    assert not out.exists()  # acceptance #1


def test_preflight_accounts_for_tiff(tmp_path, monkeypatch):
    per_field = estimate_field_bytes(_meta(), tiff=False)
    _fake_free(monkeypatch, per_field + 100)  # fits WITHOUT tiff, not WITH it
    preflight(tmp_path / "out.hcs", _meta(), 1, tiff=False)  # no raise
    with pytest.raises(InsufficientDiskSpace):
        preflight(tmp_path / "out.hcs", _meta(), 1, tiff=True)


def test_exception_message_names_both_numbers(tmp_path, monkeypatch):
    """Squid's dialog names MB-required vs MB-available; ours must too, or the operator cannot act."""
    _fake_free(monkeypatch, 10)
    with pytest.raises(InsufficientDiskSpace) as ei:
        preflight(tmp_path / "out.hcs", _meta(), 100)
    msg = str(ei.value)
    assert "MB" in msg and "available" in msg and str(tmp_path / "out.hcs") in msg
