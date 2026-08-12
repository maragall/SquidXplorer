"""Post-acquisition movie export → .mp4.

WE ARE POST-ACQUISITION, AND THAT IS THE WHOLE ARGUMENT
------------------------------------------------------
The frames already exist on disk. "Recording" here means **assembling an already-acquired axis
into a movie**; it is not camera capture, and this module has never had a code path that talks to
a camera.

That distinction is why this file is back. It was deleted on 2026-07-31 (c25d84d, "remove
recording from viewer") with the rationale "users record at acquisition time, not here", and
``scripts/smoke_import.py`` recorded the same claim as settled ("Recording lives on in
maragall/SimpleXplorer … it is not coming back here"). Both statements are true **of camera
capture**, which belongs to Squid, and both were applied to a module whose first paragraph
explicitly disclaims camera capture. The deletion answered a question this module was not asking,
and it took the T and Z sweep exports with it. That line in ``smoke_import.py`` has been corrected
rather than left to be rediscovered.

WHICH AXIS, AND WHEN THE FEATURE IS OFFERED AT ALL
--------------------------------------------------
Offered when ``n_t > 1 or n_z > 1`` (:func:`can_record`); the axis defaults to **T when there is
a time series, else Z** (:func:`default_axis`).

Gating strictly on ``n_t > 1`` was considered and rejected. Every real acquisition on the machine
this was built against is ``n_t = 1`` — the 10x tissue set, the 20x scan, the 1536 plate — so a
T-only gate would make the feature invisible on every dataset a user actually has, and would drop
the Z focus sweep the original module shipped with. Z is not a consolation prize: sweeping focus
through a fused region is the one thing a still frame of an HCS mosaic cannot show.

WHAT IS ENCODED
---------------
One frame per index on the chosen axis, each frame the **fused region mosaic** the window is
showing — the same ``fuse_region_mosaic`` the region view is built from, so a movie cannot drift
from what was on screen. Frames are composited and encoded ONE AT A TIME (streamed to the ffmpeg
writer), so a long series never sits in RAM.

CONTRAST IS LATCHED ONCE, NOT RE-COMPUTED PER FRAME. The original module took the 1st/99.8th
percentile of every frame independently, which is exactly how you *hide* the thing being recorded:
a blob moving through a field, or a stack coming into focus, changes the percentiles, so
per-frame autoscaling normalises the change away and the movie looks static. One window for the
whole movie — the window the user is already looking at, passed in by the caller — keeps the
change visible. Falls back to the first frame's percentiles when the caller has no window to give.

NO SOURCE DATA IS TOUCHED. This reads planes and writes one .mp4 to a path the user chose. Never
into the acquisition folder unless that is literally where they pointed the save dialog.

Qt-FREE ON PURPOSE. Everything here runs and is tested headless; the window half is a chip in
``_region_viewer`` and a QThread in ``_workers``. See ``tests/test_layering.py``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from squidxplorer._montage import _hex_to_rgb01, composite

#: How big a movie frame is allowed to get, on its long side. A movie is a DISPLAY artifact, not a
#: dataset: ``fuse_region_mosaic``'s own cap is 8192 px, which is a 200 MB RGB frame and not
#: something any player will open. 1080 is the long side of an ordinary screen, and the decimation
#: is applied by the fuser as it pastes, so a smaller frame is also a cheaper one.
MOVIE_MAX_PX = 1080

#: Playback rate. Independent of the frame count (N frames at F fps is an N/F-second movie), so a
#: 3-timepoint fixture is a half-second movie at 6 and a 10-plane sweep is a 1.7-second one.
DEFAULT_FPS = 6

#: The percentile pair the fallback window uses, matching the one the viewer seeds mosaics with.
_LO_PCT, _HI_PCT = 1.0, 99.8


class MissingEncoder(RuntimeError):
    """No mp4 encoder on this machine. Raised BEFORE anything is written, and it names the fix."""


def encoder_problem() -> Optional[str]:
    """``None`` when this machine can encode an mp4, else a sentence naming what is missing.

    Probed rather than assumed, because the answer differs per platform and per install: the
    encoder is ``imageio`` driving ``imageio-ffmpeg``, whose wheel carries its own ffmpeg binary
    for Linux, Windows and macOS, so nothing here depends on a system ffmpeg being on PATH. But
    the binary is a separate download inside that wheel and can be absent from a stripped or
    frozen build, which is the case this function exists to name — ``scripts/hcs-viewer.spec``
    excluded both packages until this feature came back.

    The alternative to probing is what the CLAUDE.md ``requires=`` note is about: an ImportError
    one call deep, absorbed by somebody's error handling, and a green run that wrote nothing.
    """
    try:
        import imageio.v2  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - reported by name, never swallowed
        return f"imageio is not installed ({type(exc).__name__}: {exc}). Install squidxplorer[video]."
    try:
        import imageio_ffmpeg
    except Exception as exc:  # noqa: BLE001
        return (f"imageio-ffmpeg is not installed ({type(exc).__name__}: {exc}). "
                f"Install squidxplorer[video].")
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        return (f"imageio-ffmpeg has no ffmpeg binary on this machine "
                f"({type(exc).__name__}: {exc}). Install squidxplorer[video].")
    if not exe or not os.path.exists(exe):
        return f"imageio-ffmpeg reports an ffmpeg binary that is not there: {exe!r}."
    return None


def _pad_to_even(frame: np.ndarray) -> np.ndarray:
    """H.264's yuv420p subsamples chroma 2x, so an odd width or height is not encodable.

    Padded with one row/column of black rather than resized: a resize resamples every pixel of
    every frame to hide a one-pixel problem, and the movie is then no longer the pixels that were
    read. A black edge is honest and free.
    """
    h, w = frame.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return frame
    out = np.zeros((h + h % 2, w + w % 2) + frame.shape[2:], dtype=frame.dtype)
    out[:h, :w] = frame
    return out


def write_mp4(frames: Iterable[np.ndarray], out_path, fps: int = DEFAULT_FPS) -> tuple[str, int]:
    """Encode ``(H, W, 3)`` uint8 RGB *frames* to an H.264 .mp4 at *fps*. Returns ``(path, n)``.

    Streams frame-by-frame to the writer, so peak memory is one frame whatever the axis length.

    NO PATH HERE LEAVES A FILE THAT LOOKS LIKE A SUCCESSFUL EXPORT:

    * **no encoder** → :class:`MissingEncoder`, raised before the writer is even constructed.
    * **no frames** → ``ValueError``. imageio's ffmpeg writer is lazy — the subprocess starts on
      the first ``append_data`` — so an empty encode creates nothing on disk, verified rather than
      assumed. There is deliberately no ``os.remove`` guarding a file that cannot exist.
    * **the frame producer raised part way** → the exception propagates AND the partial .mp4 is
      deleted. This is the one that needs the delete: N frames really were written, so the file is
      there, playable, and shorter than the sweep the user asked for. A truncated movie sitting at
      the path they typed, after an error they may have dismissed, is indistinguishable from the
      real thing.

    A CANCEL IS NOT A FAILURE and keeps its file. ``region_movie_frames`` stops by RETURNING, so
    the iterator ends cleanly, ``n > 0``, and this returns normally: the user gets the part of the
    sweep that had already been encoded, which is what asking to stop early means.
    """
    problem = encoder_problem()
    if problem:
        raise MissingEncoder(f"cannot write {out_path}: {problem}")

    import imageio.v2 as imageio  # lazy: keeps the encoder off the headless-pipeline import path

    out_path = str(out_path)
    n = 0
    writer = imageio.get_writer(out_path, fps=max(1, int(fps)), codec="libx264",
                                macro_block_size=None, quality=8, pixelformat="yuv420p")
    try:
        for frame in frames:
            arr = _pad_to_even(np.ascontiguousarray(frame, dtype=np.uint8))
            writer.append_data(arr)
            n += 1
    except BaseException:
        writer.close()
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
    writer.close()
    if n == 0:
        raise ValueError(f"nothing to encode: no frames were produced for {out_path}")
    return out_path, n


def axis_length(meta: dict, axis: str) -> int:
    """How many frames *axis* is worth on this acquisition. 't' → n_t, 'z' → len(z_levels)."""
    if axis == "t":
        return max(1, int(meta.get("n_t", 1) or 1))
    if axis == "z":
        return max(1, len(meta.get("z_levels") or [0]))
    raise ValueError(f"axis must be 't' or 'z', not {axis!r}")


def can_record(meta: dict) -> bool:
    """Whether there is a movie to make: more than one index on either axis.

    This IS the enable condition of the button. See the module docstring for why it is not
    ``n_t > 1``.
    """
    return axis_length(meta, "t") > 1 or axis_length(meta, "z") > 1


def default_axis(meta: dict) -> str:
    """'t' when this acquisition is a time series, else 'z'. See the module docstring."""
    return "t" if axis_length(meta, "t") > 1 else "z"


def _channel_colors(meta: dict, channels: Sequence[str],
                    rgb_by_channel: "Optional[dict]" = None) -> np.ndarray:
    """One float RGB per channel, preferring the colour the LAYER is tinted with right now.

    *rgb_by_channel* is what the window knows (``colormap_hue_rgb`` off the napari layer, as
    ``_per_channel_luts`` already collects it) — a user who recoloured a channel gets a movie in
    the colour they are looking at. Absent that, the acquisition's own ``display_color``.
    """
    by_name = {c["name"]: c for c in (meta.get("channels") or [])}
    out = []
    for name in channels:
        rgb = (rgb_by_channel or {}).get(name)
        if rgb is not None:
            out.append(np.asarray(rgb, dtype=np.float32) / 255.0)
            continue
        out.append(_hex_to_rgb01(by_name[name].get("display_color") or "#FFFFFF"))
    return np.stack(out) if out else np.zeros((0, 3), np.float32)


def _percentile_windows(store: np.ndarray) -> list:
    """One ``(lo, hi)`` per channel of a ``(C, H, W)`` stack, at the viewer's own percentiles."""
    return [(float(np.percentile(p, _LO_PCT)), float(np.percentile(p, _HI_PCT))) for p in store]


def region_movie_frames(
    reader: Any,
    meta: dict,
    region: str,
    *,
    axis: str,
    channels: "Optional[Sequence[str]]" = None,
    windows: "Optional[Sequence[Sequence[float]]]" = None,
    rgb_by_channel: "Optional[dict]" = None,
    z: int = 0,
    t: int = 0,
    max_px: int = MOVIE_MAX_PX,
    on_frame: "Optional[Callable[[int, int], None]]" = None,
    should_stop: "Optional[Callable[[], bool]]" = None,
) -> Iterator[np.ndarray]:
    """Yield one ``(H, W, 3)`` uint8 RGB frame per index along *axis*, for one region.

    The frame is the fused region mosaic — the picture the region window shows — so the movie and
    the window are the same function of the same pixels. Along 't' the fixed *z* is held; along
    'z' the fixed *t* is held. *channels* defaults to every channel; pass the VISIBLE ones and a
    hidden channel stays out of the movie exactly as it is out of the view.

    *windows* is the latched per-channel ``(lo, hi)``; ``None`` derives it from the first frame
    and reuses it for every later frame (see the module docstring on why it is never per-frame).

    *on_frame* is called ``(done, total)`` after each frame; *should_stop* is polled before each
    one, so a cancel lands within one frame instead of at the end.
    """
    from squidxplorer._mosaic_source import fuse_region_mosaic

    names = list(channels) if channels is not None else [c["name"] for c in meta["channels"]]
    if not names:
        raise ValueError(f"{region}: no channels to record — every channel is hidden.")
    colors = _channel_colors(meta, names, rgb_by_channel)

    z_levels = list(meta.get("z_levels") or [0])
    if axis == "z":
        indices = [(int(zl), int(t)) for zl in z_levels]
    elif axis == "t":
        z_use = z_levels[min(int(z), len(z_levels) - 1)]
        indices = [(int(z_use), int(ti)) for ti in range(axis_length(meta, "t"))]
    else:
        raise ValueError(f"axis must be 't' or 'z', not {axis!r}")

    total = len(indices)
    latched = list(windows) if windows else None
    for done, (zi, ti) in enumerate(indices, start=1):
        if should_stop is not None and should_stop():
            return
        planes = []
        for ch in names:
            fused = fuse_region_mosaic(reader, meta, region, ch, z=zi, t=ti, max_px=max_px)
            if fused is None:
                # Same "not derivable, do not guess" signal the mosaic path returns. A movie of a
                # region with no stage positions would be a wrong picture, not a rough one.
                raise ValueError(
                    f"{region}: no stage positions / pixel size — there is no mosaic to record.")
            planes.append(fused[0])
        store = np.stack(planes)
        if latched is None:
            latched = _percentile_windows(store)
        yield composite(store, colors, latched)
        if on_frame is not None:
            on_frame(done, total)


def record_region(
    reader: Any,
    meta: dict,
    region: str,
    out_path,
    *,
    axis: str,
    fps: int = DEFAULT_FPS,
    **frame_kwargs,
) -> tuple[str, int]:
    """Fuse, composite and encode one region's *axis* into *out_path*. Returns ``(path, n)``.

    The whole export in one call, so the GUI worker and a headless test drive the identical path.
    """
    frames = region_movie_frames(reader, meta, region, axis=axis, **frame_kwargs)
    return write_mp4(frames, out_path, fps=fps)
