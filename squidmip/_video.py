"""Post-acquisition video export → .mp4 (IMA-177 / Record).

We are post-acquisition: the frames already exist on disk. "Recording" here means **assembling an
already-acquired axis into a movie**, not capturing from a camera. Default axis is **T** (time-lapse);
**Z** is a fallback (a focus sweep) only when there is no time series — in HCS you normally MIP or pick
a reference plane rather than scrub Z. One well = one movie (a condition).

Bounded: frames are composited + encoded one at a time (streamed to the ffmpeg writer), so a long
series never sits in RAM. Playback `fps` is a display/encode rate, independent of the frame count
(N frames at F fps → an N/F-second movie).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from squidmip._montage import _hex_to_rgb01, _window


def write_mp4(frames: Iterable[np.ndarray], out_path, fps: int) -> str:
    """Encode an iterable of ``(H, W, 3)`` uint8 RGB frames to an H.264 ``.mp4`` at *fps*.

    Streams frame-by-frame to the ffmpeg writer (bundled via imageio-ffmpeg — no system install), so
    memory stays at one frame. Raises ValueError if no frames are produced.
    """
    import imageio.v2 as imageio  # lazy: keeps the encoder dep off the headless-pipeline import path

    out_path = str(out_path)
    n = 0
    writer = imageio.get_writer(out_path, fps=max(1, int(fps)), codec="libx264",
                                macro_block_size=None, quality=8)
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame, dtype=np.uint8))
            n += 1
    finally:
        writer.close()
    if n == 0:
        raise ValueError("write_mp4 got no frames to encode")
    return out_path


def _composite(planes: list[np.ndarray], colors: np.ndarray, dmax: float) -> np.ndarray:
    """Composite one timepoint's per-channel planes into an (H, W, 3) uint8 RGB frame (per-frame
    percentile contrast; display colours from the acquisition)."""
    h, w = planes[0].shape
    rgb = np.zeros((h, w, 3), np.float32)
    for c_i, plane in enumerate(planes):
        p = plane.astype(np.float32)
        lo, hi = float(np.percentile(p, 1.0)), float(np.percentile(p, 99.8))
        rgb += _window(p, lo, hi)[:, :, None] * colors[c_i][None, None, :]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def well_movie_frames(reader, region: str, fov: int, *, axis: str = "t", z: int = 0) -> Iterator[np.ndarray]:
    """Yield composited RGB frames for one well along *axis* ('t' = time-lapse at z, 't' default;
    't' uses the given z / mid-stack; 'z' = a focus sweep at t=0). One frame per index on that axis."""
    meta = reader.metadata
    channels = [c["name"] for c in meta["channels"]]
    colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in meta["channels"]])
    dmax = float(np.iinfo(np.dtype(meta["dtype"])).max)
    zs = meta["z_levels"]
    z_use = zs[len(zs) // 2] if z is None else zs[min(z, len(zs) - 1)]
    if axis == "z":
        for zi in zs:
            yield _composite([reader.read(region, fov, ch, zi, 0) for ch in channels], colors, dmax)
    else:  # time-lapse
        for t in range(meta["n_t"]):
            yield _composite([reader.read(region, fov, ch, z_use, t) for ch in channels], colors, dmax)


def default_axis(meta: dict, record_z: bool = False) -> str:
    """Pick the video axis. Default T (time-lapse): in HCS you project Z (MIP / reference plane) and
    the movie runs on T. ``record_z=True`` is the opt-in to record the Z focus sweep instead. Falls
    back to Z when there is no time series."""
    if record_z:
        return "z"
    return "t" if meta.get("n_t", 1) > 1 else "z"
