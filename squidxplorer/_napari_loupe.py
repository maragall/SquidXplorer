"""The canvas loupe: shift-left-click a view window's napari canvas to magnify what is under it.

WHY A LOUPE ON THE CANVAS AT ALL, WHEN THE CANVAS ALREADY ZOOMS
----------------------------------------------------------------
Because on a sparse acquisition it cannot zoom far enough to be useful without losing the place.
Measured on the 40x AF-sweep sets: a 4168 px frame framed to fill an 860x720 window is a **6.1x
downsample**, and the fields sit on a pitch of 7x the field, so 97% of the mosaic is empty space.
Wheel-zooming into one field to check focus and back out to find the next is the workflow this
replaces -- ``RegionViewer``'s FOV walk answers *which field*, and this answers *is it sharp*.

EVERYTHING ABOUT PIXELS IS ``squidxplorer._loupe``'s, AND THAT IS THE POINT
-----------------------------------------------------------------------
This module owns the GESTURE and the OVERLAY. What magnification means, which pyramid level
answers it, how the crop is bounded, where the pixels come from, how the bar is drawn -- all of
that is ``_loupe``, shared with the plate's press-and-hold loupe. See that module's docstring, and
the IMA-242 note it carries, for what the last private copy of ``composite`` and ``_pct_window``
cost: three bugs that were invisible until somebody put the two surfaces side by side.

THE CONTRAST AND THE COLOURS ARE THE CANVAS'S, NOT THE SOURCE'S
-----------------------------------------------------------------
A loupe is a magnifier OF THE SURFACE IT SITS ON. The plate's ``_loupe_lut`` makes this argument
and it transfers unchanged: if the inset derived its own window from the source's percentiles, then
dragging a contrast slider would move the canvas and leave the inset showing the old window
forever -- which is one of the three IMA-242 bugs, exactly. So the window and the colormap come
from the napari LAYERS (``RegionViewer._per_channel_luts`` / ``_visible_channels``, the same two
methods the .mp4 export reads), and the source's own ``window()`` is the stated fallback for a
channel the canvas has no opinion about.

SUPPRESSING THE CAMERA IS SUPPORTED API, NOT A TRICK
------------------------------------------------------
Verified against the installed napari 0.8 (``napari/_vispy/canvas.py``):
``VispyCanvas._process_mouse_event`` hands callbacks a ``ReadOnlyWrapper(napari_event,
exceptions=('handled',))`` -- ``handled`` is the ONE field a callback is meant to write -- and then
copies it back onto the vispy event; ``NapariSceneCanvas._process_mouse_event``, which is what
forwards to the camera, early-returns ``if event.handled``. napari's own ``drag_to_zoom`` uses the
same mechanism.

Deliberately NOT ``viewer.camera.mouse_zoom = False``, which also works. That is a MODE: it has to
be restored, and any path that dismisses the loupe without restoring it leaves the user's canvas
permanently un-zoomable. ``event.handled`` is per-event and fails safe.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import QWidget

from squidxplorer._logpane import get_logger
from squidxplorer._loupe import (
    _LOUPE_PX,
    _LOUPE_MAG,
    canvas_scale,
    capped_at_native,
    loupe_clamp_crop,
    loupe_crop_px,
    loupe_inset_rect,
    loupe_label,
    loupe_level,
    loupe_scale_at,
    loupe_um_per_screen_px,
    paint_loupe_inset,
)
from squidxplorer._montage import composite
from squidxplorer._qthread_life import detach

log = get_logger("napari_loupe")

#: The magnification ladder the wheel steps through. CLAMPED at both ends, never wrapped: a
#: magnifier that jumps 32x -> 2x on one more click of the wheel is a magnifier that lies about
#: what you are looking at, and the inset gives you no way to tell.
_MAG_LADDER = (2.0, 4.0, 8.0, 16.0, 32.0)


#: The remembered factor, for the LIFE OF THE PROCESS. Without this the wheel would be re-taught
#: to every window, which is the opposite of a setting. A SESSION value and not a prefs file:
#: `_prefs` went with the 2026-08-13 kill list (the close-all checkbox made the same move, to
#: `PlateWindow._warn_close_all`), so the factor resets next launch — the same trade, taken for
#: the same reason. Deliberately NOT `ViewSettings`: that is per-window look (contrast,
#: channels), and a loupe factor is not a property of one region's picture.
_SESSION_MAG = float(_LOUPE_MAG)


def _default_mag_index() -> int:
    """Where the ladder starts: what the user last chose this session, else the engine's default.

    A value that is no longer on the ladder (the ladder changed) snaps to the nearest rung
    instead of being honoured as something the wheel could never reach again.
    """
    try:
        want = float(_SESSION_MAG)
    except (TypeError, ValueError):
        want = float(_LOUPE_MAG)
    return min(range(len(_MAG_LADDER)), key=lambda i: abs(_MAG_LADDER[i] - want))


class LoupeInset(QWidget):
    """The floating inset. Paints, and holds nothing but what it was last told to show.

    ``WA_TransparentForMouseEvents`` so it never steals input: every press, move and wheel over
    the inset passes straight through to the GL widget underneath, which is what keeps
    wheel-to-magnify working while the cursor is over the inset itself. That is the right choice
    BECAUSE THE INSET HAS NO CONTROLS. The moment one is added this attribute is wrong.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(_LOUPE_PX, _LOUPE_PX)
        self._image: Optional[QImage] = None
        self._note = ""
        self._label = ""
        self._um_px: Optional[float] = None
        self.hide()

    def show_at(self, anchor_x: int, anchor_y: int) -> None:
        """Place the inset for an anchor in CANVAS pixels — the space a napari event's ``pos`` is
        already in, so there is nothing to convert."""
        host = self.parentWidget()
        if host is None:
            return
        bx, by = loupe_inset_rect(int(anchor_x), int(anchor_y), host.width(), host.height())
        self.move(bx, by)
        self.show()
        self.raise_()

    def show_crop(self, image, note: str = "", label: str = "", um_per_screen_px=None) -> None:
        self._image, self._note, self._label, self._um_px = image, note, label, um_per_screen_px
        self.update()

    def paintEvent(self, _event):                    # noqa: N802 - Qt naming
        p = QPainter(self)
        try:
            paint_loupe_inset(p, 0, 0, image=self._image, note=self._note, label=self._label,
                              um_per_screen_px=self._um_px)
        finally:
            p.end()


class CanvasLoupe(QObject):
    """Shift-left-click raises it; the wheel magnifies; Esc, a plain click or a shift-click drops it.

    Built with plain callables rather than a ``RegionViewer`` so this module has no back-edge into
    the window that owns it — the same shape ``_gallery`` uses to stay testable without a plate.
    """

    def __init__(self, *, viewer, canvas_widget: QWidget, meta: dict,
                 source_for: "Callable[[str], Any]",
                 mosaic,
                 region_of: "Callable[[], str]",
                 time_point_of: "Callable[[], int]",
                 look_of: "Callable[[], tuple]",
                 say: "Callable[[str], None]",
                 parent=None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._canvas = canvas_widget
        self._meta = meta or {}
        self._source_for = source_for
        self._mosaic = mosaic
        self._region_of = region_of
        self._time_point_of = time_point_of
        self._look_of = look_of
        self._say = say

        self._inset = LoupeInset(canvas_widget)
        self._worker = None
        self._source = None
        self._up = False
        self._gen = 0
        self._mag_index = _default_mag_index()
        self._anchor = (0, 0)          # canvas px
        self._point_um = (0.0, 0.0)    # world (y, x)
        self._zoom = 1.0
        self._said_native = False
        self._done = False
        self._connections: list = []

        self._connect(viewer.mouse_drag_callbacks, self._on_drag)
        self._connect(viewer.mouse_wheel_callbacks, self._on_wheel)
        canvas_widget.installEventFilter(self)

    # -- wiring --------------------------------------------------------------------------
    def _connect(self, callbacks: list, fn) -> None:
        callbacks.append(fn)
        # Remembered so `shutdown` can take them OUT again: napari's callback lists hold a strong
        # reference, so a window that closed while still registered would keep itself alive and
        # keep receiving events against a dead canvas.
        self._connections.append((callbacks, fn))

    def eventFilter(self, obj, event):               # noqa: N802 - Qt naming
        if obj is self._canvas and event.type() in (QEvent.Resize, QEvent.Hide):
            if event.type() == QEvent.Hide:
                self.dismiss()
            elif self._up:
                self._inset.show_at(*self._anchor)   # re-clamp inside the new canvas size
        return False

    # -- the gestures --------------------------------------------------------------------
    def _on_drag(self, viewer, event):
        """Shift-left-press raises the loupe and SUPPRESSES the pan.

        Setting ``handled`` on the PRESS is what kills the pan, and it is enough on its own:
        vispy only assigns ``_mouse_handler`` on a press it actually delivered, so a press the
        camera never saw means the subsequent moves never reach it either. That matters because
        napari throttles ``_on_mouse_move`` (``qthrottled(..., timeout=5)``), so setting
        ``handled`` from a move callback would be a race; from a press it is not.
        """
        if getattr(event, "button", 1) != 1 or "Shift" not in (event.modifiers or ()):
            if self._up and getattr(event, "button", 1) == 1:
                # A plain click means the user is doing something else, and a 240 px opaque panel
                # over the thing they just clicked is in the way. NOT handled: the click still
                # pans and still selects an ROI, exactly as it would have.
                self.dismiss()
            return
        if int(getattr(viewer.dims, "ndisplay", 2)) != 2:
            self._say("the loupe reads a 2-D point; switch back to 2D to use it.")
            return
        event.handled = True
        if self._up:
            self.dismiss()               # the gesture is its own toggle
            return
        self._raise_at(event)
        yield
        while getattr(event, "type", "") == "mouse_move":
            event.handled = True
            self._raise_at(event)
            yield

    def _on_wheel(self, viewer, event):
        """While the loupe is up the wheel changes MAGNIFICATION and the camera does not move."""
        if not self._up:
            return                        # not handled: the canvas zooms exactly as before
        event.handled = True
        try:
            step = 1 if float(np.asarray(event.delta).ravel()[-1]) > 0 else -1
        except Exception:                 # noqa: BLE001 - a wheel we cannot read moves nothing
            return
        nxt = self._mag_index + step
        if not 0 <= nxt < len(_MAG_LADDER):
            self._say(f"the loupe is at {_MAG_LADDER[self._mag_index]:g}× — "
                      f"that is as far as it {'magnifies' if step > 0 else 'zooms out'}.")
            return
        self._mag_index = nxt
        self._said_native = False         # a new ask deserves a fresh answer about the cap
        # REMEMBERED for the session, so the next window opens where the user left off.
        global _SESSION_MAG
        _SESSION_MAG = float(_MAG_LADDER[nxt])
        self._request()

    def dismiss(self) -> None:
        self._up = False
        self._gen += 1                    # anything already in flight is now stale
        try:
            self._inset.hide()
        except RuntimeError:
            pass

    # -- doing it ------------------------------------------------------------------------
    def _raise_at(self, event) -> None:
        pos = tuple(getattr(event, "position", ()) or ())
        if len(pos) < 2:
            return
        self._point_um = (float(pos[-2]), float(pos[-1]))
        try:
            self._anchor = (int(event.pos[0]), int(event.pos[1]))
        except Exception:                 # noqa: BLE001 - no canvas pos: keep the last anchor
            pass
        self._zoom = float(getattr(event, "camera_zoom", 0) or
                           getattr(self._viewer.camera, "zoom", 1.0))
        self._up = True
        self._inset.show_at(*self._anchor)
        self._request()

    def _ensure_worker(self):
        """Build the source and the worker on FIRST USE, not at window construction.

        A window whose loupe is never raised costs one attribute. The SOURCE is shared per
        acquisition (it caches a whole field's planes — tens of MB — and a written-plate source is
        mutated as wells land); the WORKER is per window, because it has one pending slot and two
        consumers sharing it would cancel each other's reads.
        """
        op = None
        try:
            op = self._mosaic.visible_op()
        except Exception:                 # noqa: BLE001
            pass
        source = self._source_for(op) if self._source_for is not None else None
        if source is None:
            return None
        if source is not self._source:
            self._stop_worker()
            self._source = source
        if self._worker is None:
            from squidxplorer._loupe import _LoupeWorker

            self._worker = _LoupeWorker(source)
            self._worker.ready.connect(self._on_crop)
            self._worker.start()
        return self._worker

    def _request(self) -> None:
        from squidxplorer._mosaic_source import fov_pixel_at_point

        if not self._up:
            return
        region = self._region_of()
        y_um, x_um = self._point_um
        hit = fov_pixel_at_point(self._meta, region, x_um, y_um) if region else None
        if hit is None:
            self._inset.show_crop(None, note="no field here", label="")
            return
        fov, py, px = hit
        pitch = self._meta.get("pixel_size_um")
        frame = self._meta.get("frame_shape") or (1, 1)
        field_px = max(1, int(min(int(frame[-2]), int(frame[-1]))))

        s_screen = canvas_scale(self._zoom, pitch) if pitch else 1.0
        s_loupe, mag = loupe_scale_at(s_screen, field_px, mag=_MAG_LADDER[self._mag_index])

        worker = self._ensure_worker()
        if worker is None:
            self._inset.show_crop(None, note="no pixel source for this layer")
            self._say("the loupe reads what is on disk, and there is no source for the layer "
                      "this window is showing.")
            return

        level = loupe_level(s_loupe, getattr(self._source, "n_levels", 1))
        crop = loupe_crop_px(s_loupe, level)
        scaled = 1 << int(level)
        y0, x0, h, w = loupe_clamp_crop(int(py / scaled) - crop // 2, int(px / scaled) - crop // 2,
                                        crop, crop,
                                        max(1, int(frame[-2]) // scaled),
                                        max(1, int(frame[-1]) // scaled))
        self._pending = (region, fov, s_loupe, mag)
        self._gen += 1
        worker.request(self._gen, region, level, y0, x0, h, w, int(self._time_point_of()), fov)

        # THE CAP, SAID ONCE. `loupe_scale_at` clamps at 1:1, so on a canvas already near native
        # every rung of the ladder yields the same picture. Silence there is a control that moves
        # and does nothing; a message per wheel tick is noise. Once per ask.
        if capped_at_native(mag, _MAG_LADDER[self._mag_index]) and not self._said_native:
            self._said_native = True
            self._say(f"the loupe is at native resolution here — {mag:.1f}× is all "
                      f"{_MAG_LADDER[self._mag_index]:g}× can give without inventing pixels. "
                      f"Zoom the canvas out for it to gain more.")

    def _on_crop(self, gen: int, well: str, crop, window, err) -> None:
        """Composite ON THIS THREAD, from an array that is already in RAM at inset resolution.

        The same three lines the plate's ``_on_loupe_crop`` runs, against the same
        ``_montage.composite`` — not a second compositor, not a second percentile rule, and not a
        second answer to what colour a channel is.
        """
        if int(gen) != int(self._gen) or not self._up:
            return                                    # a late arrival for a stale position
        if err or crop is None:
            self._inset.show_crop(None, note=str(err or "no pixels here"))
            return
        region, fov, s_loupe, mag = getattr(self, "_pending", (well, None, 1.0, 1.0))
        try:
            names, colors, windows, mask = self._look_of()
            wins = [w if w is not None else (window[i] if window and i < len(window) else None)
                    for i, w in enumerate(windows)]
            rgb = composite(crop, colors, wins, mask)
            h, w_ = rgb.shape[:2]
            img = QImage(np.ascontiguousarray(rgb).data, w_, h, 3 * w_,
                         QImage.Format_RGB888).copy()
        except Exception as exc:                      # noqa: BLE001 - named, never fatal
            self._inset.show_crop(None, note=f"{type(exc).__name__}: {exc}")
            return
        subject = f"{region} fov {fov}" if fov is not None else str(region)
        self._inset.show_crop(
            img,
            label=loupe_label(subject, mag, requested=_MAG_LADDER[self._mag_index]),
            um_per_screen_px=loupe_um_per_screen_px(
                getattr(self._source, "pixel_size_um", None), s_loupe))

    # -- lifecycle -----------------------------------------------------------------------
    def retarget(self) -> None:
        """The region, the timepoint or the visible layer changed. Re-read at the same point."""
        if self._up:
            self._request()

    def clear_cache(self) -> None:
        """Drop the decoded crops without stopping the thread — for a tab going background."""
        worker = self._worker
        if worker is not None:
            try:
                worker.clear_cache()
            except Exception:                         # noqa: BLE001
                pass

    def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            worker.stop()
            if not worker.wait(2000):
                # Mid-decode. NEVER drop the last reference to a running QThread: Qt calls
                # qFatal and the process aborts with no traceback. See `_qthread_life.detach`.
                detach(worker)
        except RuntimeError:
            pass

    def shutdown(self) -> None:
        """Idempotent teardown: dismiss, unhook, and JOIN the worker."""
        if self._done:
            return
        self._done = True
        self.dismiss()
        try:
            self._canvas.removeEventFilter(self)
        except RuntimeError:
            pass
        for callbacks, fn in self._connections:
            try:
                callbacks.remove(fn)
            except ValueError:
                pass
        self._connections.clear()
        self._stop_worker()
        try:
            self._inset.setParent(None)
            self._inset.deleteLater()
        except RuntimeError:
            pass
