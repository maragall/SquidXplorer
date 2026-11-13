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
    bbox_um: "Optional[Sequence[float]]" = None,
    on_planes: "Optional[Callable[..., None]]" = None,
    on_frame: "Optional[Callable[[int, int], None]]" = None,
    should_stop: "Optional[Callable[[], bool]]" = None,
) -> Iterator[np.ndarray]:
    """Yield one ``(H, W, 3)`` uint8 RGB fused-mosaic frame per index along *axis*.

    *windows* is the per-channel ``(lo, hi)``, latched ONCE for the whole movie; ``None``
    derives it from the first frame — never per frame, which hides the change being recorded.

    *bbox_um* crops every frame to that world box (an ROI view records its ROI, not the
    region around it). *on_planes* receives ``(frame, z_level, time_point, names, store, step)``
    with the ``(C, H, W)`` RAW-COUNT planes a frame is composited from, before any windowing:
    the one seam a measurement reads, so its numbers describe the frame's own pixels.
    """
    from squidxplorer._mosaic_source import fuse_region_mosaic, mosaic_bbox_um

    region_bbox = mosaic_bbox_um(meta, region) if bbox_um is not None else None
    if bbox_um is not None and region_bbox is None:
        raise ValueError(f"{region}: no stage positions, so there is no box to crop a movie to.")

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
        step = 1.0
        for ch in names:
            fused = fuse_region_mosaic(reader, meta, region, ch, z_level=zi, time_point=ti, max_px=max_px)
            if fused is None:
                # same "not derivable, do not guess" signal the mosaic path returns
                raise ValueError(
                    f"{region}: no stage positions / pixel size - there is no mosaic to record.")
            planes.append(fused[0])
            step = float(fused[1])
        store = np.stack(planes)
        if region_bbox is not None:
            store = _crop_to_bbox(store, region_bbox, bbox_um, region)
        if on_planes is not None:
            on_planes(done - 1, zi, ti, names, store, step)
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
    measurements: bool = False,
    **frame_kwargs,
) -> tuple[str, int]:
    """Fuse, composite and encode one region's *axis* into *out_path*. Returns ``(path, n)``.

    *measurements* also writes the movie's numbers beside it (``_measurements``): per-frame
    statistics and x/y profiles of the same raw-count planes each frame shows.
    """
    if not measurements:
        frames = region_movie_frames(reader, meta, region, axis=axis, **frame_kwargs)
        return write_mp4(frames, out_path, fps=fps)

    from squidxplorer import _measurements as m

    dt_s = _declared_interval_s(reader)
    pixel_size = float(meta["pixel_size_um"])
    stats, profile = [], []

    def measure(frame: int, z_level: int, time_point: int, names, store, step: float) -> None:
        ident = dict(frame=frame, time_point=time_point, z_level=z_level,
                     time_s=None if dt_s is None else float(dt_s) * time_point,
                     pixel_um=pixel_size * step)
        stats.extend(m.frame_stats_rows(store, names, **ident))
        profile.extend(m.frame_profile_rows(store, names, **ident))

    frames = region_movie_frames(reader, meta, region, axis=axis, on_planes=measure,
                                 **frame_kwargs)
    result = write_mp4(frames, out_path, fps=fps)
    stats_path, profile_path = m.measurement_paths(out_path)
    m.write_rows(stats_path, m.STATS_COLUMNS, stats)
    m.write_rows(profile_path, m.PROFILE_COLUMNS, profile)
    return result


def _declared_interval_s(reader: Any) -> Optional[float]:
    """The acquisition's DECLARED seconds between timepoints, or ``None`` when it states none.

    Read off the acquisition's own record (the reader's metadata does not carry it). It is the
    interval the run was asked for, not a clock: a timepoint that overran started late.
    """
    from squidxplorer._acquisition import load_acquisition_metadata

    root = getattr(reader, "source_id", None)
    if root is None:
        return None
    try:
        return load_acquisition_metadata(root).get("dt_s_declared")
    except (OSError, ValueError, KeyError):
        return None


def _crop_to_bbox(store: np.ndarray, region_bbox, bbox_um, region: str) -> np.ndarray:
    """``store[..., r0:r1, c0:c1]`` for the part of *bbox_um* inside the region's own box.

    The same proportional crop a view's ROI display takes (``_region_viewer._crop_levels_to_bbox``),
    on one fused plane: no placement of its own, only a slice of what the fuser placed.
    """
    x0, y0, x1, y1 = (float(v) for v in region_bbox)
    rx0, ry0, rx1, ry1 = (float(v) for v in bbox_um)
    rx0, rx1 = max(min(rx0, rx1), x0), min(max(rx0, rx1), x1)
    ry0, ry1 = max(min(ry0, ry1), y0), min(max(ry0, ry1), y1)
    if rx1 <= rx0 or ry1 <= ry0:
        raise ValueError(f"{region}: the ROI lies outside this region, so there is nothing to record.")
    h, w = store.shape[-2:]
    sx, sy = w / (x1 - x0), h / (y1 - y0)
    c0 = int(max(0, min(w - 1, round((rx0 - x0) * sx))))
    c1 = int(max(c0 + 1, min(w, round((rx1 - x0) * sx))))
    r0 = int(max(0, min(h - 1, round((ry0 - y0) * sy))))
    r1 = int(max(r0 + 1, min(h, round((ry1 - y0) * sy))))
    return store[..., r0:r1, c0:c1]
