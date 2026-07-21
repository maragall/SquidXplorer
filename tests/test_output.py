"""IMA-184 output writer — clean-room unit tests (no reader, no data on disk).

Drives ``write_from_stream`` with a fabricated metadata dict + a hand-built
``(region, fov, image)`` stream, then reads the written store back with tensorstore + json
(the same v3 store ndviewer_light reads). The real-seam cross commit (``project_plate`` on
``sim_1536wp`` + hongquan) lives in tests/test_integration.py.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path

import numpy as np
import tensorstore as ts

from squidmip._output import (
    parse_well_id,
    plate_metadata,
    split_well,
    write_from_stream,
)

CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "Fluorescence 638 nm - Penta", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "Fluorescence 405 nm - Penta", "display_color": "#20ADF8"},
]
# B10 present so column sort is natural (2,3,10 not 10,2,3) and no zero-padding is exercised.
REGIONS = ["B2", "B3", "B10"]


def _meta():
    return {
        "regions": REGIONS,
        "fovs_per_region": {r: [0] for r in REGIONS},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(seed: int, t=1, c=2, y=8, x=8, dtype=np.uint16):
    # deterministic, unique-ish per (seed, t, c) plane, kept small so it also fits uint8
    base = np.arange(y * x).reshape(y, x)
    out = np.empty((t, c, 1, y, x), dtype=dtype)
    for ti in range(t):
        for ci in range(c):
            out[ti, ci, 0] = ((base + seed * 20 + ti * 7 + ci * 3) % 200).astype(dtype)
    return out


def _stream(images: dict):
    # completion order is arbitrary in reality; yield out of plate order on purpose
    for region in ("B3", "B10", "B2"):
        yield region, 0, images[region]


def _read_array(path: Path) -> np.ndarray:
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}, open=True
    ).result()
    return np.asarray(store[...].read().result())


# --- pure helpers ---------------------------------------------------------------------------

def test_parse_well_id_uppercases_no_padding_roundtrips():
    # vendored Squid semantics: uppercase, multi-letter rows, no zero-padding
    assert parse_well_id("B2") == ("B", "2")
    assert parse_well_id("aa3") == ("AA", "3")  # lowercase -> upper (Squid parse_well_id)
    assert split_well is parse_well_id  # back-compat alias
    for region in ("B2", "H12", "AA1", "AF48"):
        row, col = parse_well_id(region)
        assert row + col == region  # ndviewer reconstructs well_id = row + col


def test_parse_well_id_fails_loud_on_non_plate_region():
    import pytest

    for bad in ("region_1", "1A", "B2C"):  # not <letters><digits> -> refuse, don't mislabel
        with pytest.raises(ValueError):
            parse_well_id(bad)


def test_plate_metadata_natural_column_sort_and_well_paths():
    ome = plate_metadata(REGIONS, field_count=1)["plate"]
    assert [c["name"] for c in ome["columns"]] == ["2", "3", "10"]  # int sort, not lexicographic
    assert [c["name"] for c in ome["rows"]] == ["B"]
    paths = {w["path"] for w in ome["wells"]}
    assert paths == {"B/2", "B/3", "B/10"}  # no zero-padding


# --- full write via the stream --------------------------------------------------------------

def test_write_from_stream_layout_and_pixels(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert manifest["n_wells"] == 3 and manifest["n_fields_written"] == 3

    # plate group metadata
    plate_doc = json.loads((plate / "zarr.json").read_text())
    assert plate_doc["node_type"] == "group"
    assert plate_doc["attributes"]["ome"]["plate"]["field_count"] == 1

    # each well: group metadata + single-level field + omero + pixel-exact full-res array
    for region in REGIONS:
        row, col = parse_well_id(region)
        well_doc = json.loads((plate / row / col / "zarr.json").read_text())
        assert well_doc["attributes"]["ome"]["well"]["images"] == [{"path": "0"}]  # raw fov id 0

        field = plate / row / col / "0"
        field_doc = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]
        ds_paths = [d["path"] for d in field_doc["multiscales"][0]["datasets"]]
        assert ds_paths == ["0"]  # single level, Squid canonical (no pyramid)
        colors = [c["color"] for c in field_doc["omero"]["channels"]]
        assert colors == ["FF0000", "20ADF8"]  # hex without '#', in channel order

        assert np.array_equal(_read_array(field / "0"), images[region])  # full-res pixel-exact
        assert not (field / "1").exists()  # no pyramid level written

    # every group validates against the official OME-NGFF v0.5 pydantic models
    from tests.ngff_check import assert_valid_ngff_plate

    assert_valid_ngff_plate(plate)


def test_large_field_writes_pyramid(tmp_path):
    # A field larger than the pyramid floor (256 px) gets downsample LEVELS: 600 -> 300 -> 150.
    # Level 0 stays full-res pixel-exact; coarser levels are half-size area-averages. Small fields
    # (the other tests, 8x8) collapse to level 0 alone — canonical single-level output unchanged.
    big = {r: _image(i, y=600, x=600) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(big), tmp_path, n_fovs=1, tiff=False)
    assert manifest["levels"] == 3

    field = Path(manifest["plate"]) / "B" / "2" / "0"
    ds_paths = [d["path"] for d in
                json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]]
    assert ds_paths == ["0", "1", "2"]
    assert np.array_equal(_read_array(field / "0"), big["B2"])          # level 0 pixel-exact
    assert _read_array(field / "1").shape == (1, 2, 1, 300, 300)        # half-size
    assert _read_array(field / "2").shape == (1, 2, 1, 150, 150)
    # coarse-level scale reflects the real downsample factor (2x, 4x) in Y,X
    scales = [d["coordinateTransformations"][0]["scale"] for d in
              json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]]
    assert scales[1][-2:] == [0.325 * 2, 0.325 * 2]
    assert scales[2][-2:] == [0.325 * 4, 0.325 * 4]


def test_write_from_stream_individual_tiffs(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    import tifffile

    tiff_dir = tmp_path / "tiff" / "0"
    for region in REGIONS:
        for c_i, ch in enumerate(CH):
            f = tiff_dir / f"{region}_0_0_{ch['name']}.tiff"
            assert f.exists(), f
            plane = tifffile.imread(f)
            assert plane.dtype == np.uint16  # native dtype preserved
            assert np.array_equal(plane, images[region][0, c_i, 0])  # pixel-exact, z collapsed


def test_tiff_disabled(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    assert manifest["tiff"] is None
    assert not (tmp_path / "tiff").exists()


def test_uint8_dtype_preserved(tmp_path):
    images = {r: _image(i, dtype=np.uint8) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    arr = _read_array(tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "0")
    assert arr.dtype == np.uint8
    assert np.array_equal(arr, images["B2"])


def test_fails_loud_on_wrong_shape(tmp_path):
    import pytest

    # z not collapsed (Z=3) -> a seam bug; refuse rather than write a mislabelled field
    bad = np.zeros((1, 2, 3, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="T, C, 1, Y, X"):
        write_from_stream(_meta(), iter([("B2", 0, bad)]), tmp_path, n_fovs=1, tiff=False)


def test_fails_loud_on_channel_count_mismatch(tmp_path):
    import pytest

    # image says 3 channels, metadata lists 2 -> refuse (would mislabel omero)
    bad = np.zeros((1, 3, 1, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="channels"):
        write_from_stream(_meta(), iter([("B2", 0, bad)]), tmp_path, n_fovs=1, tiff=False)


def test_writer_memory_is_bounded_in_well_count(tmp_path):
    """The writer streams: peak RSS is flat in the number of wells (it never holds the plate).

    Feeds a lazy generator of N wells and checks 4x the wells does NOT ~4x the peak — each
    (region, fov, image) is written and released before the next is pulled.
    """
    import tracemalloc

    def stream(n):
        for i in range(n):
            yield f"B{i + 2}", 0, _image(i, y=256, x=256)  # ~256 KB/well, built lazily

    def peak_for(n, dest):
        meta = {
            **_meta(),
            "regions": [f"B{i + 2}" for i in range(n)],
            "fovs_per_region": {f"B{i + 2}": [0] for i in range(n)},
        }
        tracemalloc.start()
        write_from_stream(meta, stream(n), dest, n_fovs=1, tiff=True)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    p4 = peak_for(4, tmp_path / "a")
    p16 = peak_for(16, tmp_path / "b")
    assert p16 < p4 * 2  # 4x wells, <2x peak -> bounded/streaming, not proportional to plate size


# =================================================================================================
# IMA-230 storage guard: stop before overflow, and leave an artifact that tells the truth.
# =================================================================================================

import pytest

from squidmip._storage import InsufficientDiskSpace


def _cap_free(monkeypatch, free_bytes_value):
    """Pin the reported free space. A list lets a test shrink the disk mid-run."""
    box = {"free": free_bytes_value}
    import shutil as _sh

    monkeypatch.setattr(
        _sh, "disk_usage",
        lambda p: type("U", (), {"total": 1 << 40, "used": 0, "free": box["free"]})(),
    )
    return box


def _many(n):
    regions = [f"B{i + 2}" for i in range(n)]
    meta = {
        "regions": regions,
        "fovs_per_region": {r: [0] for r in regions},
        "channels": CH,
        "pixel_size_um": 0.325,
        "frame_shape": (8, 8),
        "dtype": np.uint16,
        "n_t": 1,
    }
    images = {r: _image(i) for i, r in enumerate(regions)}

    def stream():
        for r in regions:
            yield r, 0, images[r]

    return meta, stream, regions


# --- the guard is opt-in; existing callers are untouched -----------------------------------------

def test_guard_disabled_by_default_writes_whole_plate(tmp_path, monkeypatch):
    _cap_free(monkeypatch, 0)  # a disk with literally no space...
    meta, stream, regions = _many(4)
    m = write_from_stream(meta, stream(), tmp_path, n_fovs=1)  # ...and no min_free_bytes
    assert m["n_fields_written"] == 4  # guard never armed, behaviour unchanged
    assert m["truncated"] is False


# --- the guard fires -----------------------------------------------------------------------------

def test_guard_stops_before_overflow_and_raises_typed_error(tmp_path, monkeypatch):
    _cap_free(monkeypatch, 5_000)
    meta, stream, _ = _many(8)
    with pytest.raises(InsufficientDiskSpace) as ei:
        write_from_stream(meta, stream(), tmp_path, n_fovs=1,
                          min_free_bytes=4_000, write_workers=1)
    assert ei.value.truncated is True
    assert ei.value.bytes_free == 5_000
    assert "TRUNCATED" in str(ei.value)


def test_guard_consults_free_space_only_when_armed(tmp_path, monkeypatch):
    from squidmip import _output as O

    seen = []
    monkeypatch.setattr(O, "free_bytes", lambda p: (seen.append(p), 10**9)[1])

    meta, stream, _ = _many(4)
    O.write_from_stream(meta, stream(), tmp_path / "off", n_fovs=1)
    assert seen == [], "disarmed guard must not stat the filesystem at all"

    meta, stream, _ = _many(4)
    O.write_from_stream(meta, stream(), tmp_path / "on", n_fovs=1, min_free_bytes=1000)
    assert seen, "armed guard must consult free space before each submit"


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_guard_never_overshoots_the_reserve_at_any_concurrency(tmp_path, monkeypatch, workers):
    """The safety property: with room for K fields above the reserve, the run must never write more
    than K, whatever the writer count. Reserving only for the NEXT field would let the pool's
    in-flight work land together and blow straight through."""
    from squidmip._storage import estimate_field_bytes

    meta, stream, _ = _many(12)
    per_field = estimate_field_bytes(meta)
    room_for = 3
    _cap_free(monkeypatch, per_field * room_for)

    with pytest.raises(InsufficientDiskSpace) as ei:
        write_from_stream(meta, stream(), tmp_path / f"w{workers}", n_fovs=1,
                          min_free_bytes=0, write_workers=workers)
    assert ei.value.fields_written <= room_for
    assert ei.value.fields_written < 12  # it really did stop early


# --- honest partial output -----------------------------------------------------------------------

def test_abort_truncates_plate_and_well_metadata_to_what_landed(tmp_path, monkeypatch):
    """The layout is declared up front; after an abort that declaration must be rewritten or it
    names wells holding no arrays."""
    meta, stream, regions = _many(8)
    from squidmip._storage import estimate_field_bytes

    per_field = estimate_field_bytes(meta)
    _cap_free(monkeypatch, per_field * 4)
    with pytest.raises(InsufficientDiskSpace) as ei:
        write_from_stream(meta, stream(), tmp_path, n_fovs=1,
                          min_free_bytes=per_field, write_workers=1)

    plate = tmp_path / "plate.ome.zarr"
    ome = json.loads((plate / "zarr.json").read_text())["attributes"]["ome"]
    declared = {w["path"] for w in ome["plate"]["wells"]}
    on_disk = {
        f"{r}/{c}"
        for r in (p.name for p in plate.iterdir() if p.is_dir())
        for c in (p.name for p in (plate / r).iterdir() if p.is_dir())
    }
    assert declared == on_disk, "plate metadata must match the directories that exist"
    assert len(declared) == ei.value.fields_written


def test_truncated_plate_is_still_valid_ngff(tmp_path, monkeypatch):
    from tests.ngff_check import assert_valid_ngff_plate
    from squidmip._storage import estimate_field_bytes

    meta, stream, _ = _many(8)
    per_field = estimate_field_bytes(meta)
    _cap_free(monkeypatch, per_field * 4)
    with pytest.raises(InsufficientDiskSpace):
        write_from_stream(meta, stream(), tmp_path, n_fovs=1,
                          min_free_bytes=per_field, write_workers=1)
    assert_valid_ngff_plate(tmp_path / "plate.ome.zarr")  # acceptance #4


def test_truncated_plate_preserves_intended_regions(tmp_path, monkeypatch):
    """plate_metadata() recomputes rows/cols from whatever it is given, destroying the record of
    what was MEANT to be written. Resume needs that back."""
    from squidmip._storage import estimate_field_bytes

    meta, stream, regions = _many(8)
    per_field = estimate_field_bytes(meta)
    _cap_free(monkeypatch, per_field * 4)
    with pytest.raises(InsufficientDiskSpace):
        write_from_stream(meta, stream(), tmp_path, n_fovs=1,
                          min_free_bytes=per_field, write_workers=1)
    ome = json.loads((tmp_path / "plate.ome.zarr" / "zarr.json").read_text())["attributes"]["ome"]
    sq = ome["squidmip"]
    assert sq["truncated"] is True
    assert sq["intended_regions"] == regions          # the full plate, not the survivors
    assert len(sq["written_regions"]) < len(regions)  # and we can tell them apart
    assert sq["n_fields_intended"] == len(regions)


def test_abort_leaves_no_chunkless_field_directory(tmp_path, monkeypatch):
    """create_array writes the array zarr.json BEFORE its chunks, so an interrupted field leaves a
    structurally valid, data-missing array that ndviewer would display as real."""
    from squidmip._storage import estimate_field_bytes

    meta, stream, _ = _many(8)
    per_field = estimate_field_bytes(meta)
    _cap_free(monkeypatch, per_field * 4)
    with pytest.raises(InsufficientDiskSpace):
        write_from_stream(meta, stream(), tmp_path, n_fovs=1,
                          min_free_bytes=per_field, write_workers=1)
    plate = tmp_path / "plate.ome.zarr"
    for zarr_json in plate.rglob("*/0/zarr.json"):
        arr = zarr_json.parent
        chunks = [p for p in arr.rglob("*") if p.is_file() and p.name != "zarr.json"]
        assert chunks, f"{arr} has metadata but no chunk data"


# --- a real out-of-space error that beats the guard ----------------------------------------------

def test_real_tensorstore_no_space_is_translated(tmp_path, monkeypatch):
    """T0 spike: tensorstore raises ValueError with NO errno. It must still land as the typed error
    with the same honest cleanup, not as a raw traceback."""
    from squidmip import _output as O

    real = O._write_field
    calls = {"n": 0}

    def flaky(field_dir, image, channels, *a, **k):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ValueError(
                "RESOURCE_EXHAUSTED: Error writing local file: No space left on device "
                "[os_error_code='28']"
            )
        return real(field_dir, image, channels, *a, **k)

    monkeypatch.setattr(O, "_write_field", flaky)
    meta, stream, _ = _many(8)
    with pytest.raises(InsufficientDiskSpace) as ei:
        O.write_from_stream(meta, stream(), tmp_path, n_fovs=1, write_workers=1)
    assert ei.value.truncated is True
    assert "os_error_code" in (ei.value.detail or "")


def test_real_tifffile_short_write_is_translated(tmp_path, monkeypatch):
    """The other measured shape: OSError with errno=None."""
    from squidmip import _output as O

    real = O._write_tiffs

    def flaky(*a, **k):
        raise OSError("9000000 requested and 6123392 written")

    monkeypatch.setattr(O, "_write_tiffs", flaky)
    meta, stream, _ = _many(4)
    with pytest.raises(InsufficientDiskSpace):
        O.write_from_stream(meta, stream(), tmp_path, n_fovs=1, tiff=True, write_workers=1)


def test_unrelated_writer_error_still_propagates_raw(tmp_path, monkeypatch):
    """Only out-of-space is translated. A channel mismatch must NOT be disguised as a disk problem."""
    from squidmip import _output as O

    def boom(*a, **k):
        raise ValueError("channel/axis mismatch — refusing to mislabel omero")

    monkeypatch.setattr(O, "_write_field", boom)
    meta, stream, _ = _many(4)
    with pytest.raises(ValueError, match="mislabel omero"):
        O.write_from_stream(meta, stream(), tmp_path, n_fovs=1, write_workers=1)


# --- CRITICAL REGRESSIONS ------------------------------------------------------------------------

def test_regression_stop_predicate_is_still_a_clean_cancel(tmp_path, monkeypatch):
    """The viewer passes stop=self._stop.is_set and reads the same flag to decide the run ended
    normally. A cancel must stay a cancel: no exception, no truncation flag."""
    _cap_free(monkeypatch, 0)
    meta, stream, _ = _many(8)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    m = write_from_stream(meta, stream(), tmp_path, n_fovs=1, stop=stop, write_workers=1)
    assert m["truncated"] is False       # a cancel is not a storage abort
    assert m["n_fields_written"] < 8     # and it really did stop early


def test_regression_stop_and_guard_coexist(tmp_path, monkeypatch):
    """Both armed at once: the cancel still wins cleanly, with the guard also enabled."""
    _cap_free(monkeypatch, 10**9)
    meta, stream, _ = _many(8)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    m = write_from_stream(meta, stream(), tmp_path, n_fovs=1, stop=stop,
                          min_free_bytes=1000, write_workers=1)
    assert m["truncated"] is False


def test_regression_on_error_skipped_wells_no_longer_declared_but_absent(tmp_path):
    """Pre-existing bug fixed for free: a well skipped by on_error was still declared in the plate
    metadata with no arrays on disk. The completion tracking now makes the declaration honest."""
    meta, _, regions = _many(4)
    images = {r: _image(i) for i, r in enumerate(regions)}

    def stream():                      # B3 never arrives, exactly as on_error would drop it
        for r in regions:
            if r == "B3":
                continue
            yield r, 0, images[r]

    m = write_from_stream(meta, stream(), tmp_path, n_fovs=1)
    assert m["n_fields_written"] == 3
    plate = tmp_path / "plate.ome.zarr"
    assert not (plate / "B" / "3" / "0").exists()   # nothing on disk for the skipped well


# --- tiff accounting ------------------------------------------------------------------------------

def test_estimate_counts_the_tiff_tree_which_is_a_sibling_of_the_plate(tmp_path):
    """plate.ome.zarr and tiff/ are siblings, so a plate-directory walk would miss the uncompressed
    TIFF copy entirely and under-reserve on the heaviest configuration."""
    meta, stream, _ = _many(4)
    with_tiff = write_from_stream(meta, stream(), tmp_path / "a", n_fovs=1, tiff=True)
    meta2, stream2, _ = _many(4)
    without = write_from_stream(meta2, stream2(), tmp_path / "b", n_fovs=1, tiff=False)
    assert with_tiff["bytes_written"] > without["bytes_written"]
    # and the counter really spans both trees
    assert (tmp_path / "a" / "tiff").exists()


# --- real filesystem, real ENOSPC (macOS/Linux only) ---------------------------------------------

@contextlib.contextmanager
def _tiny_filesystem(mb=24):
    """A real, size-capped filesystem. Monkeypatched free space proves the POLICY; only a real disk
    proves the guard actually beats the writer to the punch and that the cleanup survives a genuine
    mid-field failure (the T0 spike showed tensorstore's error is a ValueError with no errno, so a
    synthetic OSError would prove nothing about production)."""
    import platform
    import subprocess
    import tempfile
    import uuid

    if platform.system() != "Darwin" or shutil.which("hdiutil") is None:
        pytest.skip("needs macOS hdiutil for a size-capped filesystem")
    tag = uuid.uuid4().hex[:8]
    dmg = Path(tempfile.gettempdir()) / f"squidmip-test-{tag}.dmg"
    mnt = Path(tempfile.gettempdir()) / f"squidmip-mnt-{tag}"
    subprocess.run(["hdiutil", "create", "-size", f"{mb}m", "-fs", "HFS+",
                    "-volname", "SQUIDMIPT", "-quiet", str(dmg)], check=True)
    subprocess.run(["hdiutil", "attach", str(dmg), "-quiet", "-mountpoint", str(mnt)], check=True)
    try:
        yield mnt
    finally:
        subprocess.run(["hdiutil", "detach", str(mnt), "-quiet"], check=False)
        dmg.unlink(missing_ok=True)


def _incompressible(n_wells, px=512):
    """Noise, so zstd cannot rescue the run and the disk really does fill."""
    regions = [f"B{i + 2}" for i in range(n_wells)]
    rng = np.random.default_rng(0)
    meta = {"regions": regions, "fovs_per_region": {r: [0] for r in regions}, "channels": CH[:1],
            "pixel_size_um": 0.325, "frame_shape": (px, px), "dtype": np.uint16, "n_t": 1}

    def stream():
        for r in regions:
            yield r, 0, rng.integers(0, 65535, size=(1, 1, 1, px, px), dtype=np.uint16)

    return meta, stream, regions


def _assert_honest_plate(plate: Path, expect_truncated: bool):
    from tests.ngff_check import assert_valid_ngff_plate

    ome = json.loads((plate / "zarr.json").read_text())["attributes"]["ome"]
    declared = {w["path"] for w in ome["plate"]["wells"]}
    on_disk = {f"{r.name}/{c.name}" for r in plate.iterdir() if r.is_dir()
               for c in r.iterdir() if c.is_dir()}
    assert declared == on_disk, "metadata must describe exactly the directories that exist"
    chunkless = [a.parent for a in plate.rglob("*/0/zarr.json")
                 if not [p for p in a.parent.rglob("*") if p.is_file() and p.name != "zarr.json"]]
    assert not chunkless, f"array metadata with no chunk data survived: {chunkless}"
    assert ome.get("squidmip", {}).get("truncated", False) is expect_truncated
    assert_valid_ngff_plate(plate)


@pytest.mark.integration
def test_real_disk_guard_stops_before_the_disk_actually_fills():
    with _tiny_filesystem() as mnt:
        meta, stream, regions = _incompressible(40)
        out = mnt / "guarded"
        with pytest.raises(InsufficientDiskSpace) as ei:
            write_from_stream(meta, stream(), out, n_fovs=1,
                              min_free_bytes=4 * 1024**2, write_workers=4)
        assert 0 < ei.value.fields_written < len(regions)
        # It stopped WITH room to spare — the whole point of a guard rather than a crash handler.
        assert shutil.disk_usage(mnt).free > 1024**2
        _assert_honest_plate(out / "plate.ome.zarr", expect_truncated=True)


@pytest.mark.integration
def test_real_disk_unguarded_enospc_is_translated_and_cleaned():
    """Guard disarmed, so a genuine tensorstore RESOURCE_EXHAUSTED lands mid-field. It must surface
    as the typed error, and the half-written field must be gone."""
    with _tiny_filesystem() as mnt:
        meta, stream, regions = _incompressible(40)
        out = mnt / "unguarded"
        with pytest.raises(InsufficientDiskSpace) as ei:
            write_from_stream(meta, stream(), out, n_fovs=1, write_workers=2)
        assert ei.value.truncated is True
        assert "RESOURCE_EXHAUSTED" in (ei.value.detail or "")
        _assert_honest_plate(out / "plate.ome.zarr", expect_truncated=True)
