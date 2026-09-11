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

_RAW_OP = "raw"

Pose = Union[str, tuple]


@dataclass(frozen=True)
class Step:
    """One pose of the script: where the camera snaps, and how long it holds there.

    ``pose`` is a preset name ("xy", "xz", "yz", "fit") or an explicit (rx, ry, rz) degrees
    triple. While recording, ``dwell_s`` is seconds OF MOVIE at the stated fps (at least one
    frame, so a dwell-less final pose still appears); live, it is a wall-clock wait.
    ``transition_s`` overrides the script-wide pan time into this pose; the first step never
    pans in - the movie opens at its pose. ``zoom`` multiplies the framed zoom after the
    snap (zoom is not an angle, so the one-writer rule does not cover it).

    A ``glide=True`` step never re-frames: its transition interpolates the camera CENTER
    toward ``center`` (world (z, y, x); None holds it) and the zoom geometrically to the
    current zoom times ``zoom``, per frame alongside the angles, and its final write is
    angles-only through ``snap_camera`` plus the exact center/zoom - the zoom chapter's
    move. Center and zoom writes stay inside this module; angles still have ONE writer.
    """

    pose: Pose
    dwell_s: float = 0.0
    transition_s: Optional[float] = None
    zoom: Optional[float] = None
    center: Optional[tuple] = None
    glide: bool = False


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
        if step.glide:
            cam = vol._viewer.camera
            c0, z0 = tuple(float(v) for v in cam.center), float(cam.zoom)
            c1 = tuple(float(v) for v in step.center) if step.center is not None else c0
            z1 = z0 * float(step.zoom) if step.zoom is not None else z0
            t = transition_s if step.transition_s is None else float(step.transition_s)
            n = max(1, int(round(t * fps)))
            arc = (transition_angles(prev, target, n)
                   if (target is not None and prev is not None) else [])
            for i in range(1, n):
                if i - 1 < len(arc):
                    _volume_view.snap_camera(win, arc[i - 1], settle=False)
                f = i / n
                cam.center = tuple(a + (b - a) * f for a, b in zip(c0, c1))
                cam.zoom = z0 * (z1 / z0) ** f
                if record:
                    yield capture()
                else:
                    sleep(1.0 / fps)
            if target is not None:
                _volume_view.snap_camera(win, target, settle=False)
            cam.center, cam.zoom = c1, z1
            _volume_view.refresh_bricks(win)  # center and stride both moved
            wait_ready(win)
            if record:
                for _ in range(dwell_frames(step.dwell_s, fps)):
                    yield capture()
            elif step.dwell_s > 0:
                sleep(float(step.dwell_s))
            if target is not None:
                prev = target
            continue
        if target is not None and prev is not None:
            t = transition_s if step.transition_s is None else float(step.transition_s)
            for angles in transition_angles(prev, target, int(round(t * fps))):
                # settle=False: a pan frame turns the camera alone; the step's own snap
                # below re-frames and refines the bricks.
                _volume_view.snap_camera(win, angles, settle=False)
                if record:
                    yield capture()
                else:
                    sleep(1.0 / fps)         # live: the pan plays at the recording's own pace
        _volume_view.snap_camera(win, step.pose if isinstance(step.pose, str) else target)
        if step.zoom is not None:
            cam = vol._viewer.camera
            cam.zoom = float(cam.zoom) * float(step.zoom)
            _volume_view.refresh_bricks(win)     # the zoom moved the stride the camera needs
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


#: The demo's oblique hero angle: a 35 degree tilt off the XY top-down, yawed to azimuth 30
#: (tuned on a real ViewerModel: tilt 35.0, azimuth 30.1 by ``camera.view_direction``).
OBLIQUE_HERO = (0.0, 16.7, 58.8)

#: Fill factor on the oblique poses; axial views stay at the frame's own fit.
DEMO_ZOOM = 1.15


def demo_orbit_steps() -> list:
    """The canned SOP fly-around, ~22 s recorded at the default fps.

    XY top-down fit 2.5 s; slow 2 s pan to the oblique hero (35 degrees off XY, azimuth
    30), 3 s; an orbit - the azimuth sweeps 30 -> 80 -> 130 on the same 35 degree cone
    over 6 s of pan (each waypoint tuned on a real ViewerModel to hold tilt 35.0 +/- 0.1);
    XZ 2.5 s (the axial view); YZ 2.5 s; back to the hero for 2 s.
    """
    return [
        Step("xy", dwell_s=2.5),
        Step(OBLIQUE_HERO, dwell_s=3.0, transition_s=2.0, zoom=DEMO_ZOOM),
        Step((0.0, 34.4, 82.9), transition_s=3.0, zoom=DEMO_ZOOM),     # azimuth 80
        Step((0.0, 26.1, 114.3), transition_s=3.0, zoom=DEMO_ZOOM),    # azimuth 130
        Step("xz", dwell_s=2.5),
        Step("yz", dwell_s=2.5),
        Step(OBLIQUE_HERO, dwell_s=2.0, transition_s=2.0, zoom=DEMO_ZOOM),
    ]


def pose_for(tilt_deg: float, azimuth_deg: float) -> tuple:
    """The (0, ry, rz) triple whose ``camera.view_direction`` tilts *tilt_deg* off the z
    axis at *azimuth_deg* in the (y, x) plane: napari's rx=0 forward vector is
    (-cos(ry)sin(rz), cos(ry)cos(rz), -sin(ry)), inverted exactly (the demo waypoints'
    hand-tuned mapping, closed form)."""
    tilt = np.radians(float(tilt_deg))
    az = np.radians(float(azimuth_deg))
    ry = np.arcsin(np.sin(tilt) * np.sin(az))
    rz = np.arccos(np.clip(np.sin(tilt) * np.cos(az) / np.cos(ry), -1.0, 1.0))
    return (0.0, round(float(np.degrees(ry)), 2), round(float(np.degrees(rz)), 2))


#: The seeded orbit's envelope: the proven demo_orbit_steps shape with drawn numbers.
ORBIT_TILT_DEG = (30.0, 40.0)
ORBIT_START_AZ_DEG = (20.0, 45.0)
ORBIT_SWEEP_DEG = (80.0, 140.0)
ORBIT_ZOOM = (1.1, 1.2)
ORBIT_DWELL_S = (2.5, 3.5)
ORBIT_WAYPOINT_DEG = 50.0

#: The zoom chapter: glide to ~2.2x the orbit zoom (~2.5x fit), drift, pull back.
ZOOM_GLIDE_FACTOR = 2.2
ZOOM_DRIFT_DEG = 20.0


def random_orbit_steps(seed: int, zoom_center: Optional[tuple] = None,
                       home_center: Optional[tuple] = None) -> list:
    """A seeded orbit inside demo_orbit_steps' proven envelope: XY 2.5 s open; a hero pose
    (tilt 30-40 off XY, azimuth 20-45, zoom 1.1-1.2, dwell 2.5-3.5 s); an 80-140 degree
    orbit (direction random) as waypoints every ~50 degrees, 2.5-3.5 s pans, all on the
    drawn tilt; XZ 2.5 s, YZ 2.5 s, back to the hero.

    Sync across comparison passes is STRUCTURAL - one list runs both passes - so the seed
    buys reproducibility and variety across videos, not the sync. One seed, one list.

    *zoom_center* (world (z, y, x)) inserts the ZOOM CHAPTER after the orbit: a 3 s glide
    of center to the target with zoom rising ZOOM_GLIDE_FACTOR-fold, a 2.5 s zoomed
    ~20 degree azimuth drift (parallax) plus a 1 s hold, and a 2 s pull-back toward
    *home_center*. The DATA chooses the target, never the seed: the caller computes it
    (the brightest structure's center of mass) and passes it in; with zoom_center=None
    the list is exactly the chapterless storyboard.
    """
    rng = np.random.default_rng(int(seed))
    tilt = rng.uniform(*ORBIT_TILT_DEG)
    az0 = rng.uniform(*ORBIT_START_AZ_DEG)
    zoom = round(float(rng.uniform(*ORBIT_ZOOM)), 3)
    hero_dwell = round(float(rng.uniform(*ORBIT_DWELL_S)), 2)
    sweep = rng.uniform(*ORBIT_SWEEP_DEG)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    n_way = max(1, int(round(sweep / ORBIT_WAYPOINT_DEG)))
    hero = pose_for(tilt, az0)
    steps = [
        Step("xy", dwell_s=2.5),
        Step(hero, dwell_s=hero_dwell, transition_s=2.0, zoom=zoom),
    ]
    for i in range(1, n_way + 1):
        pan = round(float(rng.uniform(*ORBIT_DWELL_S)), 2)
        steps.append(Step(pose_for(tilt, az0 + direction * sweep * i / n_way),
                          transition_s=pan, zoom=zoom))
    if zoom_center is not None:
        az_end = az0 + direction * sweep
        drift = pose_for(tilt, az_end + direction * ZOOM_DRIFT_DEG)
        steps += [
            Step(pose_for(tilt, az_end), transition_s=3.0, glide=True,
                 center=tuple(float(v) for v in zoom_center), zoom=ZOOM_GLIDE_FACTOR),
            Step(drift, dwell_s=1.0, transition_s=2.5, glide=True),
            Step(drift, transition_s=2.0, glide=True, zoom=1.0 / ZOOM_GLIDE_FACTOR,
                 center=(tuple(float(v) for v in home_center)
                         if home_center is not None else None)),
        ]
    steps += [
        Step("xz", dwell_s=2.5),
        Step("yz", dwell_s=2.5),
        Step(hero, dwell_s=2.0, transition_s=2.0, zoom=zoom),
    ]
    return steps


#: Width of the gray bar between the two halves of a comparison frame.
DIVIDER_PX = 4


def _show_only(mosaic, op: str) -> None:
    """Light every layer of *op*'s identities and darken every other op's, programmatically."""
    with mosaic.programmatic():
        for other in mosaic.ops():
            want = str(other) == str(op)
            for ch in mosaic.channels(other):
                for ly in mosaic.layers_for(other, ch):
                    ly.visible = want


def _side_by_side(a: np.ndarray, b: np.ndarray, divider_px: int) -> np.ndarray:
    h = min(int(a.shape[0]), int(b.shape[0]))
    divider = np.full((h, int(divider_px), 3), 64, np.uint8)
    return np.concatenate([a[:h], divider, b[:h]], axis=1)


def _rebuild_volume(win, mosaic, op, scene=None):
    """Reopen the view's volume over *op*'s identity: the ``_show_result_volume`` chain
    (close, relight, ``open_3d``), never a second open path. Returns the 2D scene's
    ``(layer, visible)`` pairs as they stood between the close and the relight - the
    user's own underlying look; pass it back as *scene* to restore it instead of *op*.
    """
    from squidxplorer import _volume_view

    _volume_view.close_native3d(win)             # gives the parked layers their identities back
    out = [(ly, bool(getattr(ly, "visible", False))) for ly in mosaic.ours()]
    if scene is not None:
        with mosaic.programmatic():
            for ly, was in scene:
                try:
                    ly.visible = was
                except Exception:                # noqa: BLE001 - the layer may be gone
                    pass
    elif op is not None:
        mosaic.show_op(str(op))
    _volume_view.open_3d(win)
    return out


def record_comparison(
    win,
    out_path,
    steps: Optional[Sequence] = None,
    *,
    fps: int = DEFAULT_FPS,
    transition_s: float = DEFAULT_TRANSITION_S,
    divider_px: int = DIVIDER_PX,
    capture: Optional[Callable[[], np.ndarray]] = None,
    sleep: Callable[[float], None] = _pump_sleep,
    wait_ready: Callable[..., bool] = wait_bricks_resident,
    writer: Optional[Callable] = None,
    rebuild: Optional[Callable] = None,
) -> ScriptResult:
    """Record raw beside the operator result: two passes of the SAME steps over the SAME
    view, raw on the left, the operator identity on the right, composed frame by frame
    into one .mp4. Camera and zoom match by construction - both passes execute identical
    steps from a fresh frame over the same extent.

    When both identities hold live layers, a pass is a visibility flip; on a real volume
    tab (where the volume's bricks hold ONE identity and the other side is parked) each
    pass REBUILDS the volume over its identity through the existing close/relight/open_3d
    chain and waits for brick residency before the steps run. Contrast latches once PER
    PASS on that pass's own layers, so each modality is honestly windowed rather than
    sharing one window. The tab ends where it started, success or failure: the original
    volume's identity and the 2D layers' own visibility come back. *steps* defaults to
    :func:`demo_orbit_steps`.
    """
    vol = getattr(win, "_native3d", None)
    mosaic = getattr(vol, "_mosaic", None)
    if vol is None or mosaic is None:
        raise ValueError("comparison: no 3D volume is up in this view. Open 3D first.")
    live = [str(o) for o in mosaic.ops()]
    parked = [str(o) for o in mosaic.parked_ops()]
    present = list(dict.fromkeys([*live, *parked]))
    if _RAW_OP not in present:
        raise ValueError("comparison: no raw layer in this view to put beside the result.")
    others = [o for o in present if o != _RAW_OP]
    if not others:
        raise ValueError("comparison: no operator result in this view. Run an operator, "
                         "then record the comparison.")
    op = str(getattr(vol, "_op", "") or "")
    if op not in others:
        if len(others) > 1:
            raise ValueError(f"comparison: several operator layers here ({sorted(others)}); "
                             f"show the one to compare and click 3D again.")
        op = others[0]
    reduces = False
    try:
        reduces = bool(mosaic._reduces_z(op))
    except Exception:                            # noqa: BLE001 - undeclared: try to render
        reduces = False
    if reduces:
        # The 3D path's own declaration refusal, surfaced with the comparison's name on it.
        raise ValueError(f"comparison: '{op}' reduces z to a single plane, so it has no "
                         f"volume to render. Show a z-preserving operator and try again.")
    steps = list(steps) if steps is not None else demo_orbit_steps()

    passes: dict = {}

    def _collector(name):
        def _write(frames, _path, fps):          # noqa: ARG001 - the writer seam's signature
            passes[name] = [np.asarray(f) for f in frames]
            return "", len(passes[name])
        return _write

    def _run_pass(name):
        run_camera_script(win, steps, out_path=out_path, fps=fps,
                          transition_s=transition_s, capture=capture, sleep=sleep,
                          wait_ready=wait_ready, writer=_collector(name))

    if _RAW_OP in live and op in live:
        # Both sides live as layers: a pass is a visibility flip, no re-read.
        snapshot = [(ly, bool(getattr(ly, "visible", False))) for ly in mosaic.ours()]
        try:
            for name, show in (("raw", _RAW_OP), ("op", op)):
                _show_only(mosaic, show)
                _run_pass(name)
        finally:
            with mosaic.programmatic():
                for ly, was in snapshot:
                    try:
                        ly.visible = was
                    except Exception:            # noqa: BLE001 - the layer may be gone
                        pass
    else:
        # A real volume tab: one identity is on the bricks, the other parked. Each pass
        # reopens the volume over its own identity through the existing chain.
        rebuild = rebuild or _rebuild_volume
        underlying = None                        # the user's own 2D look, seen at first close
        try:
            for name, show in (("raw", _RAW_OP), ("op", op)):
                scene = rebuild(win, mosaic, show)
                if underlying is None:
                    underlying = scene
                if getattr(win, "_native3d", None) is None:
                    raise ValueError(f"comparison: the volume could not be rebuilt over "
                                     f"'{show}'; see this view's own message.")
                wait_ready(win)
                _run_pass(name)
        finally:
            try:
                rebuild(win, mosaic, None, scene=underlying)
            except Exception as exc:             # noqa: BLE001 - never mask the pass's error
                say = getattr(win, "_say", None)
                if callable(say):
                    say(f"comparison: could not restore the original volume ({exc}).")
    a, b = passes["raw"], passes["op"]
    if len(a) != len(b):
        raise ValueError(f"comparison: the two passes disagree, {len(a)} raw frame(s) "
                         f"against {len(b)}; the canvas changed mid-recording.")
    if writer is None:
        from squidxplorer._video import write_mp4 as writer
    composed = (_side_by_side(fa, fb, divider_px) for fa, fb in zip(a, b))
    path, n = writer(composed, out_path, fps=fps)
    return ScriptResult(path=str(path), n_frames=int(n),
                        poses=tuple(as_step(s).pose for s in steps))
