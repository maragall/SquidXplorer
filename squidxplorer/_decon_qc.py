"""The decon iteration QC window: tri-MIP inspection over ONE solve.

Julio, 2026-09-09, verbatim: "For looking at iterations we look at the mip only. So a mip
is part of the computation. Then when we choose the iterations after the previewing, we
run it on the whole stack. This means that there should be a button in the decon UI that
opens another window with the MIP, and the turbo colormap togle, and the xy and xz yz
bands by it's side, and then with the iterations slider. Then there's the preview button
that actually runs it on the whole stack and you can see the rsults in 3D. But the MIP is
more than enough for QC."

Reinstated in spirit from the shelved ``DeconQCResultView`` (bf982a2^): one solve captures
every iteration, a stepper revisits them for free, and "use k iterations" adopts the
DISPLAYED count into the run's own parameter. The layout is the new spec: the XY MIP
large, the XZ band below it and the YZ band beside it (both z-scaled by dz/pixel so the
bands are geometrically honest), ``_iter_nav``'s slider (turbo rides its bar), one channel
at a time through a combo. The QC solve is scoped EXACTLY like a Preview: the ROI window
plus the operator's halo when one is drawn, the in-view region's centre field otherwise,
every channel. A real separate window, on purpose; the one-column rule binds the view,
not this.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from squidxplorer._logpane import get_logger

log = get_logger("decon.qc")

#: The XY panel's display budget on its long side; the bands share its scale.
XY_MAX_PX = 640

#: The latched contrast window: percentiles of the FINAL iteration's XY MIP, one window
#: per channel for the whole inspection. Per-k windows would normalise away the very
#: change being inspected (the movie's latch rule).
CONTRAST_PCT = (1.0, 99.9)

PROJECTIONS = ("xy", "xz", "yz")


class QCWorker(QThread):
    """ONE scoped decon solve with capture armed; the engine's own project_well does the
    window + halo + per-channel binding and lands the tri-MIPs in the store."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, reader, region, fov, window, op, time_point, parent=None):
        super().__init__(parent)
        self._args = (reader, str(region), int(fov), window, op, int(time_point))

    def run(self):  # pragma: no cover - thread body; the sync path is tested directly
        try:
            run_qc_solve(*self._args)
            self.done.emit()
        except Exception as exc:                 # noqa: BLE001 - reported as a sentence
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_qc_solve(reader, region, fov, window, op, time_point) -> None:
    """The QC solve, synchronous: arm capture around ONE project_well of the scoped field."""
    from squidxplorer._decon import arm_capture, clear_captures
    from squidxplorer.projection import project_well

    clear_captures()
    arm_capture(True)
    try:
        project_well(reader, region, fov, op, time_point=time_point, window=window)
    finally:
        arm_capture(False)


def _rgba(arr: np.ndarray, lo: float, hi: float, turbo: bool) -> np.ndarray:
    """8-bit RGBA of *arr* under the latched window, gray or napari's own turbo."""
    from napari.utils.colormaps import ensure_colormap

    span = float(hi - lo) or 1.0
    normed = np.clip((arr.astype(np.float32) - float(lo)) / span, 0.0, 1.0)
    cmap = ensure_colormap("turbo" if turbo else "gray")
    return (cmap.map(normed.ravel()) * 255).astype(np.uint8).reshape(*arr.shape, 4)


def _pixmap(arr: np.ndarray, lo: float, hi: float, turbo: bool, w_px: int, h_px: int):
    from qtpy.QtGui import QImage, QPixmap

    rgba = np.ascontiguousarray(_rgba(arr, lo, hi, turbo))
    h, w = arr.shape
    image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(image).scaled(
        max(1, int(w_px)), max(1, int(h_px)),
        Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


class DeconQCWindow(QWidget):
    """The tri-MIP stepper window. OWNS its captures (taken whole from the store), so the
    store never outlives the run and closing this window frees the bytes."""

    def __init__(self, caps: dict, z_ratio: float, subject: str = "",
                 on_use: "Optional[Callable[[int], None]]" = None, parent=None):
        super().__init__(parent)
        from squidxplorer._iter_nav import IterationSlider

        self.setWindowTitle("decon iteration QC")
        self.setStyleSheet("background:#0d1117;color:#e6edf3;")
        self._caps = {str(c): {int(k): dict(tri) for k, tri in by_k.items()}
                      for c, by_k in caps.items()}
        self._z_ratio = max(float(z_ratio), 1e-6)
        self._on_use = on_use
        self._turbo = False
        self._ks = sorted(set.intersection(*(set(b) for b in self._caps.values()))) \
            if self._caps else []
        if not self._ks:
            raise ValueError("the QC solve captured no iterations to inspect.")
        self._k = self._ks[-1]
        # One latched window per channel: the FINAL iteration's XY percentiles.
        self._window_by_channel = {
            c: tuple(np.percentile(by_k[self._ks[-1]]["xy"], CONTRAST_PCT))
            for c, by_k in self._caps.items()}

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        title = QLabel(subject or "decon QC")
        title.setStyleSheet("color:#8b98ad;font-size:12px;")
        head.addWidget(title, 1)
        # ONE channel at a time: three tri-panels per channel would be a dashboard.
        self.channel_combo = QComboBox()
        for c in sorted(self._caps):
            self.channel_combo.addItem(c)
        self.channel_combo.setEnabled(len(self._caps) > 1)
        self.channel_combo.currentTextChanged.connect(lambda *_: self._repaint())
        head.addWidget(self.channel_combo)
        v.addLayout(head)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.xy_label, self.xz_label, self.yz_label = QLabel(), QLabel(), QLabel()
        for lab in (self.xy_label, self.xz_label, self.yz_label):
            lab.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(self.xy_label, 0, 0)          # the MIP, large
        grid.addWidget(self.yz_label, 0, 1)          # the YZ band beside it
        grid.addWidget(self.xz_label, 1, 0)          # the XZ band below it
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(2, 1)
        v.addLayout(grid)

        self.slider = IterationSlider(on_change=self._show_k, on_turbo=self._set_turbo)
        self.slider.set_iterations(self._ks)
        v.addWidget(self.slider)

        self.use_btn = QPushButton("")
        self.use_btn.setToolTip(
            "Adopt the displayed iteration count as the decon run's own parameter; "
            "Preview then runs it on the whole stack.")
        self.use_btn.clicked.connect(self._use)
        v.addWidget(self.use_btn)

        self._repaint()
        self.resize(self.sizeHint())

    # -- the testable surface -------------------------------------------------------------
    @property
    def shown_k(self) -> int:
        return int(self._k)

    @property
    def channel(self) -> str:
        return str(self.channel_combo.currentText())

    def projection(self, name: str) -> np.ndarray:
        """The array the *name* panel is currently showing (pre-render)."""
        return self._caps[self.channel][self._k][str(name)]

    # -- stepping (a repaint of held planes, never a re-solve) ----------------------------
    def _show_k(self, k: int) -> None:
        if int(k) in self._ks:
            self._k = int(k)
            self._repaint()

    def _set_turbo(self, on: bool) -> None:
        self._turbo = bool(on)
        self._repaint()

    def _use(self, *_):
        if self._on_use is not None:
            self._on_use(int(self._k))

    def _repaint(self) -> None:
        channel = self.channel
        tri = self._caps.get(channel, {}).get(self._k)
        if tri is None:
            return
        lo, hi = self._window_by_channel[channel]
        xy, xz, yz = tri["xy"], tri["xz"], tri["yz"]
        h, w = xy.shape
        scale = min(max(XY_MAX_PX / max(h, w), 0.1), 8.0)
        zpx = xz.shape[0] * self._z_ratio * scale    # a band's z extent, geometrically honest
        self.xy_label.setPixmap(_pixmap(xy, lo, hi, self._turbo,
                                        round(w * scale), round(h * scale)))
        self.xz_label.setPixmap(_pixmap(xz, lo, hi, self._turbo,
                                        round(w * scale), round(zpx)))
        self.yz_label.setPixmap(_pixmap(np.ascontiguousarray(yz.T), lo, hi, self._turbo,
                                        round(zpx), round(h * scale)))
        self.use_btn.setText(f"use {self._k} iteration" + ("s" if self._k != 1 else ""))

    def closeEvent(self, event):                     # noqa: N802 - Qt naming
        self._caps = {}                              # closing frees the held projections
        super().closeEvent(event)


def qc_scope(view) -> "tuple[str, int, Optional[tuple]]":
    """The QC solve's field, scoped EXACTLY like a Preview: the ROI's first windowed
    field when a box is drawn, else the in-view region's CENTRE field, whole."""
    regions, windows = view._run_scope()
    if windows:
        (region, fov), window = sorted(windows.items())[0]
        return str(region), int(fov), tuple(window)
    region = view.current_region()
    if not region:
        raise ValueError("no region is in view.")
    fovs = list(((view._meta or {}).get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        raise ValueError(f"region {region!r} has no fields.")
    return str(region), int(fovs[len(fovs) // 2]), None


def start_qc(panel, view) -> None:
    """Run the QC solve off-thread and open the window when its captures land.

    *panel* is the DeconPanel (owns the worker and the window; its ``shutdown`` joins
    them); *view* is the asking RegionViewer.
    """
    from squidxplorer._decon import take_captures
    from squidxplorer._engine import bind_operator

    worker = getattr(panel, "_qc_worker", None)
    if worker is not None and worker.isRunning():
        panel.say("iteration QC: a solve is already running.")
        return
    reader, meta = getattr(view, "_reader", None), getattr(view, "_meta", None)
    if reader is None or not meta:
        panel.say("iteration QC: open an acquisition first.")
        return
    try:
        region, fov, window = qc_scope(view)
        kwargs = {n: v for n, v in (panel.kwargs() or {}).items() if n == "iterations"}
        op = bind_operator("decon", kwargs)
    except Exception as exc:                     # noqa: BLE001 - a refusal, said not raised
        panel.say(f"iteration QC: {exc}")
        return
    z_ratio = float(meta.get("dz_um") or 1.0) / float(meta.get("pixel_size_um") or 1.0)
    what = (f"{region} field {fov}"
            + (f", {window[3] - window[2]}x{window[1] - window[0]} px window" if window
               else ", whole field"))
    log.info("iteration QC: solving %s, every channel, capturing each iteration", what)

    def _land():
        panel.inspect_btn.setEnabled(True)
        caps = take_captures()
        if not caps:
            panel.say("iteration QC: the solve captured nothing "
                      "(every channel copied through?).")
            return
        old = getattr(panel, "_qc_window", None)
        if old is not None:
            try:
                old.close()
            except RuntimeError:
                pass

        def _adopt(k: int) -> None:
            refusal = panel.set_param("iterations", int(k))
            panel.say(refusal or f"decon iterations set to {k}.")

        window_ = DeconQCWindow(caps, z_ratio, subject=what, on_use=_adopt)
        panel._qc_window = window_
        window_.show()

    def _fail(msg: str) -> None:
        panel.inspect_btn.setEnabled(True)
        panel.say(f"iteration QC: {msg}")

    qc = QCWorker(reader, region, fov, window, op,
                  int(getattr(view, "time_point", 0) or 0), parent=panel)
    qc.done.connect(_land)
    qc.failed.connect(_fail)
    panel._qc_worker = qc
    panel.inspect_btn.setEnabled(False)
    qc.start()
