"""IMA-233: the benchmark harness itself, on synthetic arrays and a synthetic reader.

These tests never touch a real acquisition — they lock the harness's CONTRACTS (the
quality metrics move in the documented direction, the guards refuse before allocating,
the read accounting is restored, the table renders every operator) so that a change to
the harness fails here rather than silently producing a plausible-looking wrong table.
The actual numbers come from ``tools/benchmark.py`` on real data; a benchmark asserted
against synthetic data would be measuring the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import _benchmark as bm


# --- quality metrics -------------------------------------------------------------------

def test_relative_gradient_energy_rises_with_structure():
    flat = np.full((64, 64), 100.0, dtype=np.float32)
    noisy = flat.copy()
    noisy[::2, :] += 50.0
    assert bm.relative_gradient_energy(flat) == pytest.approx(0.0)
    assert bm.relative_gradient_energy(noisy) > 0.1


def test_relative_gradient_energy_is_scale_invariant():
    """Normalising by the mean is the whole point: doubling the exposure must not read as
    a sharper image."""
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32) + 1.0
    assert bm.relative_gradient_energy(a * 2) == pytest.approx(
        bm.relative_gradient_energy(a), rel=1e-5)


def test_relative_gradient_energy_handles_empty_and_dark():
    assert np.isnan(bm.relative_gradient_energy(np.zeros((8, 8))))
    assert np.isnan(bm.relative_gradient_energy(np.zeros((0, 0))))


def test_block_uniformity_flat_is_one_vignetted_is_lower():
    flat = np.full((64, 64), 500.0, dtype=np.float32)
    y, x = np.mgrid[0:64, 0:64]
    vignette = flat * (1.0 - 0.6 * ((y - 32) ** 2 + (x - 32) ** 2) / (2 * 32 ** 2))
    assert bm.block_uniformity(flat) == pytest.approx(1.0)
    assert bm.block_uniformity(vignette) < 0.95


def test_block_uniformity_rejects_non_2d_and_tiny():
    assert np.isnan(bm.block_uniformity(np.zeros((4, 4))))
    assert np.isnan(bm.block_uniformity(np.zeros((8, 8, 3))))


def test_overlap_ncc_is_one_at_the_true_offset():
    tilefusion = pytest.importorskip("tilefusion.registration")
    assert tilefusion is not None
    rng = np.random.default_rng(1)
    full = rng.random((256, 400)).astype(np.float32)
    # Two tiles 256x256 sharing a 112 px strip: tile_j sits 144 px to the right of tile_i.
    tile_i, tile_j = full[:, :256], full[:, 144:400]
    assert bm.overlap_ncc(tile_i, tile_j, 0, 144) == pytest.approx(1.0, abs=1e-6)
    # A wrong placement scores near zero on white noise. This is the property that makes
    # the metric usable as a seam score at all.
    assert abs(bm.overlap_ncc(tile_i, tile_j, 0, 100)) < 0.3


# --- guards ----------------------------------------------------------------------------

_META = {
    "frame_shape": (2084, 2084),
    "dtype": "uint16",
    "channels": [{"name": "c0"}, {"name": "c1"}],
    "z_levels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "n_t": 1,
    "fovs_per_region": {"A1": list(range(27))},
    "pixel_size_um": 0.752,
}


def test_expected_output_bytes_counts_z_for_a_plane_op():
    """A plane-op keeps z at full depth; a z-reducer collapses it. The Nz factor between
    them is exactly the term whose omission fills memory."""
    reducer = bm.expected_output_bytes(_META, kind="fov", regions=["A1"], n_fovs=1,
                                       consumes_z=True)
    plane_op = bm.expected_output_bytes(_META, kind="fov", regions=["A1"], n_fovs=1,
                                        consumes_z=False)
    assert plane_op == reducer * len(_META["z_levels"])
    assert reducer == 2084 * 2084 * 2 * 2   # Y * X * n_channels * itemsize


def test_expected_output_bytes_returns_zero_without_a_frame_shape():
    assert bm.expected_output_bytes({}, kind="fov", regions=[], n_fovs=1,
                                    consumes_z=True) == 0


def test_guard_memory_refuses_an_impossible_run():
    with pytest.raises(bm.BenchmarkGuardError) as exc:
        bm.guard_memory(1 << 60, what="a preposterous run")
    assert "preposterous" in str(exc.value)


def test_guard_memory_allows_a_small_run():
    assert bm.guard_memory(1024, what="a small run")["checked"] in (True, False)


def test_persist_estimate_is_overlap_aware_for_region_operators():
    """A 27-FOV well fused into one mosaic must cost less than 27 separate frames — that
    is the whole reason ``estimate_write_bytes`` grew ``region_operator``."""
    per_fov = bm.persist_estimate(_META | {"fov_positions_um": _positions()},
                                  kind="fov", regions=["A1"], n_fovs=None)
    region = bm.persist_estimate(_META | {"fov_positions_um": _positions()},
                                 kind="region", regions=["A1"], n_fovs=None)
    assert 0 < region < per_fov


def _positions():
    """A 27-FOV 5x6-ish grid at the 10x acquisition's measured 1410.45 um step."""
    step = 1410.45
    return {("A1", i): ((i % 6) * step, (i // 6) * step) for i in range(27)}


# --- read accounting -------------------------------------------------------------------

class _FakeReader:
    def __init__(self):
        self.calls = 0

    def read(self, *_args, **_kwargs):
        self.calls += 1
        return np.zeros((4, 4), dtype=np.uint16)


def test_read_recorder_accumulates_and_restores():
    reader = _FakeReader()
    original = reader.read
    rec = bm._ReadRecorder()
    with rec.wrap(reader):
        reader.read("A1", 0, "c0", 0, 0)
        reader.read("A1", 0, "c0", 1, 0)
    assert rec.calls == 2
    assert rec.nbytes == 2 * 4 * 4 * 2
    assert rec.ms >= 0.0
    assert reader.calls == 2
    # The wrapper must come off: a reader left permanently instrumented would make every
    # later measurement include this run's bookkeeping.
    assert reader.read == original or reader.read.__func__ is original.__func__


def test_read_recorder_restores_after_an_exception():
    reader = _FakeReader()
    with pytest.raises(ValueError):
        with bm._ReadRecorder().wrap(reader):
            raise ValueError("boom")
    assert "read" not in vars(reader)


# --- reporting -------------------------------------------------------------------------

def _result(op="mip", **kw):
    r = bm.OperatorResult(operator=op, kind=kw.pop("kind", "fov"), dataset="/tmp/ds")
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_as_row_and_derived_rates():
    r = _result(wall_ms=2000.0, read_ms=500.0, out_megapixels=100.0)
    assert r.compute_ms == pytest.approx(1500.0)
    assert r.mpix_per_s == pytest.approx(50.0)
    assert r.as_row()["operator"] == "mip"


def test_compute_ms_never_goes_negative():
    """Read time is accumulated across threads, so at workers > 1 it can exceed the wall
    clock. A negative 'compute' column would be nonsense, not a measurement."""
    assert _result(wall_ms=100.0, read_ms=400.0).compute_ms == 0.0


def test_format_table_lists_every_operator_including_failures():
    results = [_result("mip", wall_ms=10.0, wells=1),
               _result("stitch", kind="region", error="KeyError: nope")]
    table = bm.format_table(results)
    assert "mip" in table and "stitch" in table
    assert "! stitch: KeyError: nope" in table
    assert "seam_ncc" in bm.QUALITY_NOTES["stitch"]


def test_format_stages_reports_the_unattributed_residual():
    """Stages never sum to the wall clock; the harness must SAY so rather than let the
    reader misattribute the gap."""
    r = _result("stitch", kind="region", wall_ms=1000.0,
                stage_ms={"project": 400.0, "fuse": 500.0})
    out = bm.format_stages([r])
    assert "(other)" in out and "100.0 ms" in out


def test_write_csv_and_json_round_trip(tmp_path):
    import csv
    import json

    results = [_result("mip", wall_ms=1.0, wells=1, quality={"sharp_gain": 1.5})]
    csv_path = bm.write_csv(results, tmp_path / "b.csv")
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["operator"] == "mip"

    json_path = bm.write_json(results, tmp_path / "b.json", meta={"dataset": "x"})
    payload = json.loads(json_path.read_text())
    assert payload["results"][0]["operator"] == "mip"
    assert payload["machine"]["machine"]          # the machine is part of the measurement
    assert payload["meta"]["dataset"] == "x"


def test_default_operators_are_all_real_registry_entries():
    from squidmip import available_projectors, available_region_operators

    known = set(available_projectors()) | set(available_region_operators())
    assert set(bm.DEFAULT_OPERATORS) <= known


def test_every_default_operator_documents_its_quality_direction():
    """A quality number whose desired direction the reader has to guess is not a
    measurement."""
    assert set(bm.DEFAULT_OPERATORS) <= set(bm.QUALITY_NOTES)
