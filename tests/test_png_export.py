"""The high-resolution PNG export: data pixels through the one compositor, never a screenshot."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded — Qt binding conflict", allow_module_level=True)

from squidxplorer import _viewer as V  # noqa: E402
from squidxplorer._montage import composite  # noqa: E402
from squidxplorer._png import PNG_MAX_PX, PngChannel, render_view_png  # noqa: E402
from squidxplorer._workers import _full_res_plane  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_video import _make_5d  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixtures)


@pytest.fixture
def five_d_root(tmp_path):
    """A tiny 5-D acquisition on disk: 2 regions x 1 FOV x 2 z x 2 ch x 3 t, 64 px."""
    root = tmp_path / "acq5d"
    _make_5d().build(root, ["A1", "A2"], n_fovs=1, nz=2, nt=3, size=64)
    return root


def _open_window(qapp, root):
    """A real ``RegionViewer`` over a real acquisition, the way the plate opens one."""
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = win._viewer_manager.open(list(win._order))
    assert w is not None, "no window was opened"
    return win, w


@pytest.fixture
def save_dialog(monkeypatch, tmp_path):
    """``QFileDialog.getSaveFileName`` answering with a path, so nothing modal blocks the run."""
    from qtpy.QtWidgets import QFileDialog

    chosen = {"path": str(tmp_path / "view.png"), "calls": 0, "title": "", "suggested": ""}

    def _answer(_parent, title, *_a, **_k):
        chosen["calls"] += 1
        chosen["title"] = str(title)
        chosen["suggested"] = str(_a[0]) if _a else ""
        return chosen["path"], ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_answer))
    return chosen


def _said(monkeypatch, w) -> list:
    """Record every ``_say`` sentence the window speaks."""
    lines: list = []
    monkeypatch.setattr(w, "_say", lines.append)
    return lines


def _rgb01(layer) -> np.ndarray:
    from squidxplorer._napari_view import colormap_hue_rgb, colormap_mid_rgb

    rgb = colormap_hue_rgb(layer) or colormap_mid_rgb(layer) or (255, 255, 255)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _png_pixels(path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def test_the_png_is_the_mosaics_native_size_not_the_screens(
        qapp, napari_pane_stub, five_d_root, save_dialog):
    """The export writes the layer's full-resolution pixels: a 64 px mosaic is a 64 px PNG, whatever the (headless, tiny) canvas measures."""
    from squidxplorer._napari_view import full_res_level

    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: len(w._pane._viewer.layers) >= 2, timeout=20)
    out = Path(save_dialog["path"])

    w._save_png()
    assert _drain_until(qapp, lambda: out.exists() and w._png_worker is None, timeout=60), (
        "the PNG export never finished")

    ch0 = w._meta["channels"][0]["name"]
    layer = w._pane.mosaic.find("raw", ch0)
    lvl = full_res_level(layer.data)
    native_h, native_w = int(lvl.shape[-2]), int(lvl.shape[-1])
    pixels = _png_pixels(out)
    assert pixels.shape == (native_h, native_w, 3), (
        f"wrote {pixels.shape[:2]}, the mosaic's native shape is {(native_h, native_w)}")
    shutdown_plate_window(qapp, win)


def test_the_pixels_are_the_composite_of_the_visible_channels_under_the_on_screen_contrast(
        qapp, napari_pane_stub, five_d_root, save_dialog):
    """Hidden channel out, on-screen window in: the PNG equals the one compositor's answer."""
    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: len(w._pane._viewer.layers) >= 2, timeout=20)
    names = [c["name"] for c in w._meta["channels"]]
    assert len(names) == 2
    mosaic = w._pane.mosaic

    mosaic.find("raw", names[1]).visible = False       # out of the view is out of the PNG
    mosaic.set_contrast(names[0], 10.0, 123.0)         # a contrast nobody seeds by default
    out = Path(save_dialog["path"])

    w._save_png()
    assert _drain_until(qapp, lambda: out.exists() and w._png_worker is None, timeout=60), (
        "the PNG export never finished")

    layer = mosaic.find("raw", names[0])
    plane = np.asarray(_full_res_plane(layer.data, w._z_slider_index()))
    expected = composite(plane[None], _rgb01(layer)[None], [(10.0, 123.0)])
    np.testing.assert_array_equal(_png_pixels(out), expected)
    shutdown_plate_window(qapp, win)


def test_the_no_visible_layer_refusal_is_named(
        qapp, napari_pane_stub, five_d_root, save_dialog, monkeypatch):
    """Every layer hidden: a NAMED sentence, no dialog, no worker — never a silent no-op."""
    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: len(w._pane._viewer.layers) >= 2, timeout=20)
    for c in w._meta["channels"]:
        layer = w._pane.mosaic.find("raw", c["name"])
        if layer is not None:
            layer.visible = False
    said = _said(monkeypatch, w)

    w._save_png()

    assert said, "the refused export said nothing"
    assert "hidden" in said[-1], f"the refusal does not name the cause: {said[-1]!r}"
    assert save_dialog["calls"] == 0, "a refused export still opened the save dialog"
    assert w._png_worker is None, "a refused export still started a worker"
    shutdown_plate_window(qapp, win)


def test_a_3d_view_is_refused_with_a_reason(
        qapp, napari_pane_stub, five_d_root, save_dialog, monkeypatch):
    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: len(w._pane._viewer.layers) >= 2, timeout=20)
    w.set_render_mode("3d")
    said = _said(monkeypatch, w)

    w._save_png()

    assert said and "3D" in said[-1], f"the 3D refusal does not say why: {said!r}"
    assert save_dialog["calls"] == 0
    assert w._png_worker is None
    shutdown_plate_window(qapp, win)


def test_the_export_never_reads_a_plane_on_the_qt_thread(
        qapp, napari_pane_stub, five_d_root, save_dialog):
    """Every decode of the whole export, pinned by thread ident — the reader is wrapped BEFORE the window opens, so the raw pyramid's own lazy reads are the"""
    from squidxplorer._mosaic_source import plane_cache

    from .test_gallery import _RecordingReader

    win = V.PlateWindow(None)
    win.ingest(str(five_d_root))
    manager = win._viewer_manager
    recording = _RecordingReader(manager._reader)
    manager._reader = recording
    w = manager.open(list(win._order))
    assert w is not None
    _drain_until(qapp, lambda: len(w._pane._viewer.layers) >= 2, timeout=20)

    plane_cache().clear()          # a warm cache would make the export decode nothing
    before = recording.reads
    main = threading.get_ident()
    out = Path(save_dialog["path"])

    w._save_png()
    assert _drain_until(qapp, lambda: out.exists() and w._png_worker is None, timeout=60), (
        "the PNG export never finished")

    assert recording.reads > before, "the export read nothing — this assertion would be vacuous"
    on_ui = [t for t in recording.read_threads[before:] if t == main]
    assert not on_ui, (
        f"{len(on_ui)} of {recording.reads - before} plane reads happened on the Qt thread; "
        f"the export must decode only in _PngWorker")
    shutdown_plate_window(qapp, win)


def test_a_fovs_view_exports_the_current_field_not_the_whole_well(
        qapp, napari_pane_stub, tmp_path, save_dialog):
    """The png chip in a FOVs view crops to the field on screen and names it in the file."""
    from squidxplorer._mosaic_source import mosaic_bbox_um
    from squidxplorer._napari_view import full_res_level, pyramid_levels
    from squidxplorer._region_viewer import _crop_levels_to_bbox

    root = tmp_path / "acq4fov"
    _make_5d().build(root, ["A1"], n_fovs=4, nz=2, nt=1, size=64)
    win, w = _open_window(qapp, root)
    child = win._viewer_manager.open_child(["A1"], parent_id=w.window_id, fovs=True)
    assert child is not None
    assert _drain_until(
        qapp, lambda: child._shown_region == "A1" and bool(child._fov_boxes_cache), timeout=30)
    fov = child._fov_slider.fov
    assert fov is not None
    out = Path(save_dialog["path"])

    child._save_png()
    assert _drain_until(qapp, lambda: out.exists() and child._png_worker is None, timeout=60), (
        "the PNG export never finished")

    layer = child._pane.mosaic.find("raw", child._meta["channels"][0]["name"])
    full = tuple(int(v) for v in full_res_level(layer.data).shape[-2:])
    cut, _bbox = _crop_levels_to_bbox(pyramid_levels(layer.data),
                                      mosaic_bbox_um(child._meta, "A1"),
                                      child._fov_boxes_cache[int(fov)])
    want = tuple(int(v) for v in cut[0].shape[-2:])
    pixels = _png_pixels(out)
    assert pixels.shape[:2] != full, "the export is still the whole well"
    assert pixels.shape[:2] == want, f"wrote {pixels.shape[:2]}, the field's box is {want}"
    assert save_dialog["suggested"] == f"{root.name}_A1_fov{fov}_raw.png"
    shutdown_plate_window(qapp, win)


def test_the_renderer_caps_the_long_side_and_says_so_in_its_step():
    """Qt-free: a plane over the cap is decimated to fit, and the step reports the clip."""
    plane = (np.arange(100 * 40, dtype=np.uint16) % 251).reshape(100, 40)
    ch = PngChannel("DAPI", plane, (0.0, 250.0), (0, 0, 255), z_index=0)

    rgb, step = render_view_png([ch], max_px=40)

    assert step == 3, f"ceil(100 / 40) is 3, got {step}"
    assert rgb.shape == (34, 14, 3), f"decimated shape is {rgb.shape}"
    native, step1 = render_view_png([ch], max_px=PNG_MAX_PX)
    assert step1 == 1 and native.shape == (100, 40, 3)


def test_a_mixed_scene_exports_every_visible_channel_not_one_op(
        qapp, napari_pane_stub, five_d_root, save_dialog):
    """Raw lit on one channel beside an operator result on the other, which is what the
    screen composites (the one-lit-op rule is per channel): the PNG equals that composite.
    The old one-op walk silently dropped the raw channel from the export."""
    win, w = _open_window(qapp, five_d_root)
    _drain_until(qapp, lambda: len(w._pane._viewer.layers) >= 2, timeout=20)
    names = [c["name"] for c in w._meta["channels"]]
    mosaic = w._pane.mosaic

    raw0 = mosaic.find("raw", names[0])
    result = (np.asarray(_full_res_plane(raw0.data, 0)) // 2 + 7).astype(np.uint16)
    mosaic.add_mosaic("blur", names[1], result, contrast_limits=(5.0, 99.0))
    assert mosaic.top_visible_layer(names[0]) is raw0, "raw must stay lit on its own channel"
    assert mosaic.visible_op() == "blur"
    out = Path(save_dialog["path"])

    w._save_png()
    assert _drain_until(qapp, lambda: out.exists() and w._png_worker is None, timeout=60), (
        "the PNG export never finished")

    blur = mosaic.find("blur", names[1])
    z = w._z_slider_index()
    p0 = np.asarray(_full_res_plane(raw0.data, z))
    p1 = np.asarray(_full_res_plane(blur.data, z))
    expected = composite(
        np.stack([p0, p1]),
        np.stack([_rgb01(raw0), _rgb01(blur)]),
        [tuple(float(v) for v in raw0.contrast_limits), (5.0, 99.0)])
    np.testing.assert_array_equal(_png_pixels(out), expected)
    shutdown_plate_window(qapp, win)
