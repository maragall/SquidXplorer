"""Scriptable 3D camera: a declarative pose sequence over an open volume tab, optionally
recorded to .mp4 ("XY 3 s, snap to 45 degrees across the diagonal, XZ 3 s, YZ").

The planning half (poses, interpolation, dwell bookkeeping) is Qt-free and pure. The executor
drives :func:`squidxplorer._volume_view.snap_camera` for EVERY camera write - it stays the
app's one writer of ``camera.angles`` - and a step's dwell counts only after the volume's
brick loader reports idle, so a recording never shows half-loaded bricks. The canvas capture
is a seam (``capture=``): the default reads ``viewer.screenshot(canvas_only=True)``, which
needs a live GL canvas, so headless tests stub it and the real look is a hand check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import numpy as np

#: Recording rate: a camera pan needs more than _video's 6 fps axis-sweep default.
DEFAULT_FPS = 12

#: Seconds of interpolated pan between two poses of a recording.
DEFAULT_TRANSITION_S = 1.0

#: How long the residency wait may pump before giving up (the dwell still runs).
BRICK_WAIT_S = 30.0

Pose = Union[str, tuple]


@dataclass(frozen=True)
class Step:
    """One pose of the script: where the camera snaps, and how long it holds there.

    ``pose`` is a preset name ("xy", "xz", "yz", "fit") or an explicit (rx, ry, rz) degrees
    triple. While recording, ``dwell_s`` is seconds OF MOVIE at the stated fps (at least one
    frame, so a dwell-less final pose still appears); live, it is a wall-clock wait.
    ``transition_s`` overrides the script-wide pan time into this pose; the first step never
    pans in - the movie opens at its pose.
    """

    pose: Pose
    dwell_s: float = 0.0
    transition_s: Optional[float] = None


@dataclass(frozen=True)
class ScriptResult:
    path: Optional[str]
    n_frames: int
    poses: tuple


def as_step(spec) -> Step:
    """A ``Step``, a pose, an angles triple, or a ``(pose, dwell_s)`` pair, normalised."""
    if isinstance(spec, Step):
        return spec
    if isinstance(spec, str):
        return Step(spec)
    if isinstance(spec, (tuple, list)):
        if len(spec) == 3 and all(isinstance(v, (int, float)) for v in spec):
            return Step(tuple(float(v) for v in spec))
        if len(spec) == 2:
            return Step(spec[0] if isinstance(spec[0], str) else tuple(spec[0]),
                        float(spec[1]))
    raise ValueError(f"camera script: {spec!r} is not a step. Use Step(pose, dwell_s), "
                     f"a pose name, an (rx, ry, rz) triple, or a (pose, dwell_s) pair.")


def pose_angles(pose: Pose) -> Optional[tuple]:
    """The (rx, ry, rz) a pose means; ``None`` for "fit" (framing only, no rotation).

    Refuses an unknown pose BY NAME, so a script is validated whole before the camera moves.
    """
    from squidxplorer._volume_view import SNAP_ANGLES

    if isinstance(pose, str):
        if pose == "fit":
            return None
        angles = SNAP_ANGLES.get(pose)
        if angles is None:
            raise ValueError(f"camera script: unknown pose '{pose}'. "
                             f"Use one of {sorted(SNAP_ANGLES)}, 'fit', or an angles triple.")
        return angles
    try:
        angles = tuple(float(v) for v in pose)
    except (TypeError, ValueError):
        angles = ()
    if len(angles) != 3:
        raise ValueError(f"camera script: a pose must be a name or an (rx, ry, rz) degrees "
                         f"triple, not {pose!r}.")
    return angles


def shortest_arc(a: float, b: float) -> float:
    """The signed degrees from *a* to *b* along the shorter way round."""
    return ((float(b) - float(a) + 180.0) % 360.0) - 180.0


def transition_angles(start: tuple, end: tuple, n: int) -> list:
    """The INTERMEDIATE angle triples of an *n*-frame pan (the step's own snap writes *end*)."""
    deltas = [shortest_arc(a, b) for a, b in zip(start, end)]
    return [tuple(float(a) + d * i / n for a, d in zip(start, deltas))
            for i in range(1, max(0, int(n)))]


def dwell_frames(dwell_s: float, fps: int) -> int:
    """Frames a recorded dwell is worth: at least one, so every pose reaches the movie."""
    return max(1, int(round(float(dwell_s) * int(fps))))


def _pump_sleep(seconds: float) -> None:
    """Wall-clock wait that keeps the Qt event loop draining (brick arrivals are queued)."""
    from qtpy.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    end = time.monotonic() + float(seconds)
    while time.monotonic() < end:
        if app is not None:
            app.processEvents()
        time.sleep(0.01)


def wait_bricks_resident(win, timeout_s: float = BRICK_WAIT_S) -> bool:
    """Block (pumping Qt) until the volume's loader reports idle for the CURRENT epoch."""
    vol = getattr(win, "_native3d", None)
    loader = getattr(vol, "_loader", None)
    idle = getattr(loader, "idle", None)
    if idle is None:
        return True
    settled: list = []

    def _mark(epoch):
        if int(epoch) == int(getattr(vol, "_epoch", epoch)):
            settled.append(True)

    try:
        idle.connect(_mark)
    except Exception:                        # noqa: BLE001 - a loader without a live signal
        return True
    try:
        end = time.monotonic() + float(timeout_s)
        while not settled and time.monotonic() < end:
            _pump_sleep(0.02)
        return bool(settled)
    finally:
        try:
            idle.disconnect(_mark)
        except Exception:                    # noqa: BLE001 - already torn down
            pass


def _canvas_rgb(vol) -> np.ndarray:
    """One (H, W, 3) uint8 frame of the volume's own canvas. GUI thread, live GL only."""
    shot = vol._viewer.screenshot(canvas_only=True, flash=False)
    return np.ascontiguousarray(np.asarray(shot)[..., :3])


def _execute(win, vol, parsed: Sequence[Step], *, fps: int, transition_s: float,
             record: bool, capture: Optional[Callable], sleep: Callable,
             wait_ready: Callable) -> Iterator[np.ndarray]:
    """Walk the script; yields recorded frames (nothing when not recording)."""
    from squidxplorer import _volume_view

    prev: Optional[tuple] = None
    for step in parsed:
        target = pose_angles(step.pose)
        if record and target is not None and prev is not None:
            t = transition_s if step.transition_s is None else float(step.transition_s)
            for angles in transition_angles(prev, target, int(round(t * fps))):
                # settle=False: a pan frame turns the camera alone; the step's own snap
                # below re-frames and refines the bricks.
                _volume_view.snap_camera(win, angles, settle=False)
                yield capture()
        _volume_view.snap_camera(win, step.pose if isinstance(step.pose, str) else target)
        wait_ready(win)
        if record:
            for _ in range(dwell_frames(step.dwell_s, fps)):
                yield capture()
        elif step.dwell_s > 0:
            sleep(float(step.dwell_s))
        if target is not None:
            prev = target


def run_camera_script(
    win,
    steps: Sequence,
    *,
    out_path=None,
    fps: int = DEFAULT_FPS,
    transition_s: float = DEFAULT_TRANSITION_S,
    capture: Optional[Callable[[], np.ndarray]] = None,
    sleep: Callable[[float], None] = _pump_sleep,
    wait_ready: Callable[..., bool] = wait_bricks_resident,
    writer: Optional[Callable] = None,
) -> ScriptResult:
    """Run a camera script over *win*'s open 3D volume; record it when *out_path* is given.

    Every pose goes through ``snap_camera`` (the one writer of ``camera.angles``); each
    step waits for brick residency before its dwell counts. A recording latches contrast
    ONCE at the start (the ``_video`` rule) and pans between poses over *transition_s*.

    Usage, Julio's sentence::

        run_camera_script(view, [
            Step("xy", dwell_s=3),
            Step((45.0, 45.0, 45.0), dwell_s=3),   # 45 degrees across the diagonal
            Step("xz", dwell_s=3),
            Step("yz"),
        ], out_path="figure.mp4")
    """
    vol = getattr(win, "_native3d", None)
    if vol is None or not callable(getattr(vol, "frame", None)):
        raise ValueError("camera script: no 3D volume is up in this view. Open 3D first.")
    parsed = [as_step(s) for s in steps]
    if not parsed:
        raise ValueError("camera script: no steps.")
    for step in parsed:                      # the whole script validates before the camera moves
        pose_angles(step.pose)
    record = out_path is not None
    poses = tuple(step.pose for step in parsed)
    if not record:
        for _ in _execute(win, vol, parsed, fps=fps, transition_s=transition_s,
                          record=False, capture=None, sleep=sleep, wait_ready=wait_ready):
            pass
        return ScriptResult(path=None, n_frames=0, poses=poses)
    if capture is None:
        if not callable(getattr(vol._viewer, "screenshot", None)):
            raise ValueError("camera script: this viewer has no canvas to screenshot; "
                             "recording needs a live window.")
        capture = lambda: _canvas_rgb(vol)   # noqa: E731 - the one-line default seam
    latch = getattr(vol, "latch_contrast", None)
    if callable(latch):
        latch()
    if writer is None:
        from squidxplorer._video import write_mp4 as writer
    frames = _execute(win, vol, parsed, fps=fps, transition_s=transition_s,
                      record=True, capture=capture, sleep=sleep, wait_ready=wait_ready)
    path, n = writer(frames, out_path, fps=fps)
    return ScriptResult(path=str(path), n_frames=int(n), poses=poses)


def record_camera_script(win, steps: Sequence, out_path, **kwargs) -> ScriptResult:
    """The console entry: record *steps* over *win*'s open 3D volume into *out_path*."""
    return run_camera_script(win, steps, out_path=out_path, **kwargs)
