"""Ruling z (Julio, 2026-08-25): "do we have sub-fov decon, so that we can try it and get
results really fast?" An ROI-scoped preview used to solve the WHOLE touched FOV and crop only
the output. Now it reads and solves the ROI window PLUS A HALO (the PSF's lateral support) and
trims the halo, so the interior equals the whole-field solve; z is kept whole; a save is
unchanged (whole fields).

Measured before the rule was written (nz 5 and nz 15, CPU and MPS, 25 synthetic points blurred
with the real PSF at G7's optics): the 99.9%-energy lateral radius of the modelled PSF is
9.9 px; halo 0 differs from the whole-field solve by up to 4-5 counts; every halo >= 4 px is
within 1 count. The rule takes ceil(radius) with a floor of HALO_MIN_PX.

Measured on G7 (FOV 1, 2050^2, 3 channels, 15 z, MPS, a 529x724 box): sub-FOV 1.69 s / 0.87 GB
peak against whole-field 10.07 s / 3.37 GB (one plane: 0.56 s vs 0.67 s). The interior equals
the whole-field solve within 1 count at 2 iterations (max 1, mean 0.002); at 3 iterations it
differs (max 7 at one plane, 216 over the stack, mean 0.72) because the Biggs-Andrews
acceleration step's lambda is ONE scalar over the whole solved volume, so a window's step is not
the field's. That is the solver's global step, not the halo, and it is stated, not hidden.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from squidxplorer import projection
from squidxplorer._engine import run_plate
from squidxplorer._mosaic_source import fov_windows_px, mosaic_bbox_um
from squidxplorer._op_result import RegionResultAccumulator
from tests.conftest import FakeReader

REGION = "A1"
CHANNELS = ("488",)
NZ = 5
FRAME = (160, 192)
PX = 0.376
DZ = 1.031
#: The ROI window inside FOV 0, (r0, r1, c0, c1), and its size.
WINDOW = (48, 112, 60, 140)


def _meta(fovs=(0,), positions=None) -> dict:
    return {
        "regions": [REGION],
        "channels": [{"name": c} for c in CHANNELS],
        "z_levels": list(range(NZ)),
        "n_z": NZ, "n_t": 1, "dtype": "uint16",
        "frame_shape": FRAME, "pixel_size_um": PX, "dz_um": DZ,
        "fovs_per_region": {REGION: list(fovs)},
        "fov_positions_um": positions or {(REGION, f): (0.0, 0.0) for f in fovs},
    }


def _optics():
    from squidxplorer._decon import OpticsParams

    return OpticsParams(na=0.8, wavelength_um=0.525, dxy_um=PX, dz_um=DZ, nz=NZ, ni=1.0)


def _blurred_stack(optics) -> np.ndarray:
    """A few point sources blurred with the SAME PSF the solve uses, plus noise: the
    ordinary picture, not a flat field on which any halo is trivially exact."""
    from squidxplorer._decon import make_psf

    psf = make_psf(optics)
    rng = np.random.default_rng(0)
    h, w = FRAME
    truth = np.zeros((NZ, h, w), np.float32)
    for _ in range(25):
        z, y, x = rng.integers(0, NZ), rng.integers(10, h - 10), rng.integers(10, w - 10)
        truth[z, y, x] = rng.uniform(2000, 8000)
    shape = [max(a, b) for a, b in zip(truth.shape, psf.shape)]
    axes = (0, 1, 2)
    kern = np.fft.ifftshift(np.pad(psf, [(0, s - p) for s, p in zip(shape, psf.shape)]))
    blur = np.fft.irfftn(np.fft.rfftn(truth, s=shape, axes=axes)
                         * np.fft.rfftn(kern, s=shape, axes=axes), s=shape, axes=axes)
    blur = blur[:NZ, :h, :w] + 900 + rng.normal(0, 15, (NZ, h, w))
    return np.clip(blur, 0, 65535).astype(np.uint16)


def _reader(stack: np.ndarray, meta=None) -> FakeReader:
    return FakeReader(meta or _meta(), planes=lambda r, f, c, z, t: stack[z].copy())


# --- (a) parity: the interior of the windowed solve IS the whole-field solve ------------------

def test_the_halo_is_the_psfs_lateral_support_with_a_floor():
    pytest.importorskip("petakit")
    from squidxplorer._decon import HALO_MIN_PX, lateral_halo_px

    halo = lateral_halo_px(_optics())
    assert halo == 10, f"the 99.9% lateral radius of the G7-optics PSF measured 9.9 px; got {halo}"
    assert halo >= HALO_MIN_PX >= 4


def test_the_windowed_solve_matches_the_whole_field_solve_inside_the_window():
    pytest.importorskip("petakit")
    from squidxplorer._decon import decon_op, lateral_halo_px

    optics = _optics()
    stack = _blurred_stack(optics)
    op = decon_op(optics, iterations=3)
    assert projection.operator_halo_px(op, NZ) == lateral_halo_px(optics) == 10

    whole = projection.project_well(_reader(stack), REGION, 0, reduce=op)
    sub = projection.project_well(_reader(stack), REGION, 0, reduce=op, window=WINDOW)
    r0, r1, c0, c1 = WINDOW
    assert sub.shape == (1, 1, NZ, r1 - r0, c1 - c0), "z must be kept whole, xy is the window"
    diff = np.abs(sub[0, 0].astype(np.int64) - whole[0, 0, :, r0:r1, c0:c1].astype(np.int64))
    assert diff.max() <= 1, f"max abs diff {diff.max()} counts against the whole-field solve"


# --- (b) the read path hands the solve the window + halo, not the frame ----------------------

class _WindowReader(FakeReader):
    """A reader that can serve a window (the Zarr reader can); counts the pixels it reads."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pixels_read = 0
        self.windows: list = []

    def read_window(self, region, fov, channel, z_level, time_point, window):
        r0, r1, c0, c1 = window
        self.windows.append(tuple(window))
        plane = self.read(region, fov, channel, z_level, time_point)[r0:r1, c0:c1]
        self.pixels_read += int(plane.size)
        return plane


def test_a_window_capable_reader_reads_only_the_window_plus_halo():
    stack = np.arange(NZ * FRAME[0] * FRAME[1], dtype=np.uint16).reshape(NZ, *FRAME)
    reader = _WindowReader(_meta(), planes=lambda r, f, c, z, t: stack[z])

    op = projection.plane_op(lambda plane: plane)
    op.halo_px = 6
    out = projection.project_well(reader, REGION, 0, reduce=op, window=WINDOW)
    r0, r1, c0, c1 = WINDOW
    assert reader.windows and all(w == (r0 - 6, r1 + 6, c0 - 6, c1 + 6) for w in reader.windows)
    assert reader.pixels_read == NZ * (r1 - r0 + 12) * (c1 - c0 + 12)
    assert out.shape == (1, 1, NZ, r1 - r0, c1 - c0)
    np.testing.assert_array_equal(out[0, 0], stack[:, r0:r1, c0:c1])


def test_a_plain_reader_is_sliced_so_the_operator_never_sees_the_frame():
    stack = np.arange(NZ * FRAME[0] * FRAME[1], dtype=np.uint16).reshape(NZ, *FRAME)
    seen: list = []

    def _op(plane):
        seen.append(plane.shape)
        return plane

    op = projection.plane_op(_op)                  # no halo declared: 0
    out = projection.project_well(_reader(stack), REGION, 0, reduce=op, window=WINDOW)
    r0, r1, c0, c1 = WINDOW
    assert seen and set(seen) == {(r1 - r0, c1 - c0)}
    np.testing.assert_array_equal(out[0, 0], stack[:, r0:r1, c0:c1])


def test_a_halo_is_clamped_at_the_frame_edge_and_an_empty_window_is_refused():
    stack = np.ones((NZ, *FRAME), np.uint16)
    op = projection.plane_op(lambda p: p)
    op.halo_px = 50
    out = projection.project_well(_reader(stack), REGION, 0, reduce=op, window=(0, 20, 0, 30))
    assert out.shape[-2:] == (20, 30)
    with pytest.raises(ValueError, match="window"):
        projection.project_well(_reader(stack), REGION, 0, reduce=op, window=(10, 10, 0, 30))
    with pytest.raises(ValueError, match="window"):
        projection.project_well(_reader(stack), REGION, 0, reduce=op, window=(0, 5, 0, 999))


# --- the ROI box becomes FOV-local windows, top-left convention --------------------------------

def test_fov_windows_px_are_the_roi_in_each_touched_frames_own_pixels():
    h, w = FRAME
    meta = _meta(fovs=(0, 1), positions={(REGION, 0): (0.0, 0.0),
                                         (REGION, 1): (w * PX, 0.0)})
    # a box straddling the seam between FOV 0 and FOV 1, rows 10..50
    box = ((w - 20) * PX, 10 * PX, (w + 30) * PX, 50 * PX)
    windows = fov_windows_px(meta, REGION, box)
    assert windows == {0: (10, 50, w - 20, w), 1: (10, 50, 0, 30)}
    assert fov_windows_px(meta, REGION, (-5.0, -5.0, -1.0, -1.0)) == {}


def test_the_region_arm_refuses_windows_by_name():
    pytest.importorskip("tilefusion")
    with pytest.raises(ValueError, match="window"):
        next(iter(run_plate(_reader(np.ones((NZ, *FRAME), np.uint16)), operator="stitch",
                            windows={(REGION, 0): WINDOW})))


def test_a_save_refuses_windows_by_name(tmp_path):
    from squidxplorer._dispatch import run_operator_once

    with pytest.raises(ValueError, match="window"):
        run_operator_once(_reader(np.ones((NZ, *FRAME), np.uint16)), operator="mip", save=True,
                          owed=1, out_dir=str(tmp_path), windows={(REGION, 0): WINDOW})


# --- the accumulator places a windowed field at the window's own footprint --------------------

def test_windowed_fields_land_where_the_whole_fields_would_have():
    h, w = FRAME
    positions = {(REGION, 0): (0.0, 0.0), (REGION, 1): ((w - 8) * PX, 3 * PX)}
    meta = _meta(fovs=(0, 1), positions=positions)
    rng = np.random.default_rng(1)
    full = {f: rng.integers(1, 4000, (1, h, w), np.uint16) for f in (0, 1)}
    whole = RegionResultAccumulator("mip", REGION, meta, CHANNELS)
    for f in (0, 1):
        whole.add(f, full[f])
    ref = whole.result()
    # ONE box, rows 20..60 and cols w-30..w+32 of the region mosaic, as the view derives it
    box = ((w - 30) * PX, 20 * PX, (w + 32) * PX, 60 * PX)
    windows = fov_windows_px(meta, REGION, box)
    assert windows == {0: (20, 60, w - 30, w), 1: (17, 57, 0, 40)}, windows
    part = RegionResultAccumulator("mip", REGION, meta, CHANNELS, fovs=[0, 1], windows=windows)
    for f, (r0, r1, c0, c1) in windows.items():
        part.add(f, full[f][:, r0:r1, c0:c1])
    got = part.result()
    # the windowed mosaic's box sits inside the region's, at the windows' union
    x0, y0, x1, y1 = got.extent.bbox_um
    rx0, ry0, rx1, ry1 = ref.extent.bbox_um
    assert rx0 <= x0 < x1 <= rx1 and ry0 <= y0 < y1 <= ry1
    # and every pixel of it equals the whole mosaic at the same place
    px = float(meta["pixel_size_um"])
    oy, ox = int(round((y0 - ry0) / px)), int(round((x0 - rx0) / px))
    plane, refplane = got.plane("488"), ref.plane("488")
    np.testing.assert_array_equal(plane, refplane[oy:oy + plane.shape[0], ox:ox + plane.shape[1]])
    assert plane.shape == (40, 62)


def test_a_scoped_run_says_its_window_and_halo_once(caplog):
    from squidxplorer._workers import _OperatorWorker

    stack = np.ones((NZ, *FRAME), np.uint16)
    reader = _reader(stack)
    worker = _OperatorWorker("mip", reader, _meta(), {REGION: {"rc": (0, 0), "well_id": "A1"}},
                             "", regions={REGION: [0]}, save=False, n_fovs=None,
                             windows={(REGION, 0): WINDOW})
    with caplog.at_level(logging.INFO):
        worker._run_body(_NullMetrics())
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ROI mip")]
    assert lines == ["ROI mip: 80x64 px window + 0 px halo, 1 plane(s)"], caplog.text


class _NullMetrics:
    def finish(self, *a, **k):
        pass


# --- the view sends its box as windows; a save never does ------------------------------------

def test_a_preview_from_a_view_with_a_box_sends_the_windows_and_a_save_does_not(
        qapp, napari_pane_stub):
    from squidxplorer._region_viewer import RegionViewer

    h, w = FRAME
    meta = _meta(fovs=(0, 1), positions={(REGION, 0): (0.0, 0.0), (REGION, 1): (w * PX, 0.0)})
    calls: list = []
    win = RegionViewer(None, meta, [REGION], window_id=7,
                       operator_specs=[("mip", "Maximum Intensity Projection")],
                       run_operator=lambda key, **kw: calls.append((key, kw)))
    try:
        win._roi_bbox = ((w - 20) * PX, 10 * PX, (w + 30) * PX, 50 * PX)
        win._preview_view_operator()
        assert calls, "the Preview chip did not reach run_operator"
        _key, kw = calls[-1]
        assert kw["regions"] == {REGION: [0, 1]}
        assert kw["windows"] == {(REGION, 0): (10, 50, w - 20, w), (REGION, 1): (10, 50, 0, 30)}
        assert kw["save"] is False
        calls.clear()
        win._run_plate_operator()
        _key, kw = calls[-1]
        assert kw["save"] is True and not kw.get("windows"), "a save must run whole fields"
    finally:
        win.dispose()
