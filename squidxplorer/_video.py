"""Post-acquisition movie export -> .mp4: assemble an already-acquired axis (T, else Z)
into a movie of the fused region mosaic. Not camera capture. Qt-free on purpose.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from squidxplorer._montage import _hex_to_rgb01, composite

#: Frame cap on the long side: a movie is a display artifact, not a dataset.
MOVIE_MAX_PX = 1080

#: Playback rate, independent of the frame count.
DEFAULT_FPS = 6

#: The percentile pair the fallback window uses, matching the one the viewer seeds mosaics with.
_LO_PCT, _HI_PCT = 1.0, 99.8


class MissingEncoder(RuntimeError):
    """No mp4 encoder on this machine. Raised BEFORE anything is written, and it names the fix."""


def encoder_problem() -> Optional[str]:
    """``None`` when this machine can encode an mp4, else a sentence naming what is missing."""
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
    """Pad to even H/W with black (H.264's yuv420p cannot encode odd dimensions)."""
    h, w = frame.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return frame
    out = np.zeros((h + h % 2, w + w % 2) + frame.shape[2:], dtype=frame.dtype)
    out[:h, :w] = frame
    return out


def write_mp4(frames: Iterable[np.ndarray], out_path, fps: int = DEFAULT_FPS) -> tuple[str, int]:
    """Encode ``(H, W, 3)`` uint8 RGB *frames* to an H.264 .mp4 at *fps*. Returns ``(path, n)``.

    Streams frame-by-frame; a mid-stream error deletes the partial file, a clean early
    return (cancel) keeps it.
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
    """Whether there is a movie to make: more than one index on either axis (never t alone)."""
    return axis_length(meta, "t") > 1 or axis_length(meta, "z") > 1


def default_axis(meta: dict) -> str:
    """'t' when this acquisition is a time series, else 'z'."""
    return "t" if axis_length(meta, "t") > 1 else "z"


def _channel_colors(meta: dict, channels: Sequence[str],
                    rgb_by_channel: "Optional[dict]" = None) -> np.ndarray:
    """One float RGB per channel, preferring the colour the layer is tinted with right now."""
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
    z_level: int = 0,
    time_point: int = 0,
    max_px: int = MOVIE_MAX_PX,
    on_frame: "Optional[Callable[[int, int], None]]" = None,
    should_stop: "Optional[Callable[[], bool]]" = None,
) -> Iterator[np.ndarray]:
    """Yield one ``(H, W, 3)`` uint8 RGB fused-mosaic frame per index along *axis*.

    *windows* is the per-channel ``(lo, hi)``, latched ONCE for the whole movie; ``None``
    derives it from the first frame — never per frame, which hides the change being recorded.
    """
    from squidxplorer._mosaic_source import fuse_region_mosaic

    names = list(channels) if channels is not None else [c["name"] for c in meta["channels"]]
    if not names:
        raise ValueError(f"{region}: no channels to record - every channel is hidden.")
    colors = _channel_colors(meta, names, rgb_by_channel)

    z_levels = list(meta.get("z_levels") or [0])
    if axis == "z":
        indices = [(int(zl), int(time_point)) for zl in z_levels]
    elif axis == "t":
        z_use = z_levels[min(int(z_level), len(z_levels) - 1)]
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
            fused = fuse_region_mosaic(reader, meta, region, ch, z_level=zi, time_point=ti, max_px=max_px)
            if fused is None:
                # same "not derivable, do not guess" signal the mosaic path returns
                raise ValueError(
                    f"{region}: no stage positions / pixel size - there is no mosaic to record.")
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
    """Fuse, composite and encode one region's *axis* into *out_path*. Returns ``(path, n)``."""
    frames = region_movie_frames(reader, meta, region, axis=axis, **frame_kwargs)
    return write_mp4(frames, out_path, fps=fps)
