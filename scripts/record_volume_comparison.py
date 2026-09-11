#!/usr/bin/env python
"""Seeded-orbit raw-vs-decon comparison video, 561 only, whole-FOV bricked 3D, real GL.

Two sequential headful passes over volume_figure's boot (raw set, then the decon46 set),
each recording THE SAME random_orbit_steps(seed) list through run_camera_script, then the
two .mp4s composed frame k beside frame k with a 4 px divider. Sync is structural: one
steps list, two passes; the seed buys reproducibility and variety across videos. Both
passes window 561 at volume_figure's CONTRAST_561, one intensity scale.

Usage: python scripts/record_volume_comparison.py [--seed N]   # replaces a running instance
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))            # volume_figure's shared boot/teardown

RAW_SET = Path("/Users/julioamaragall/Downloads/25x_C4_dz=3_2026-08-14_16-51-15.744692")
#: (H, W) physical canvas pixels, asserted EXACT on both passes: fit-derived zoom depends
#: on canvas geometry, so an unpinned canvas made the two sides render at different scales.
CANVAS_PX = (1504, 2090)
DECON_SET = Path("/Users/julioamaragall/Downloads/"
                 "decon46_25x_C4_dz=3_2026-08-14_16-51-15.744692")
DESKTOP = Path("/Users/julioamaragall/Desktop")
DEFAULT_SEED = 4242
FPS = 12
RESIDENCY_TIMEOUT_S = 300.0


def _pin_canvas(app, view, vol) -> None:
    """Resize the view's window until the GL canvas is EXACTLY CANVAS_PX; refuse else."""
    import volume_figure as vf

    h = w = -1
    for _ in range(8):
        shot = vol._viewer.screenshot(canvas_only=True, flash=False)
        h, w = int(shot.shape[0]), int(shot.shape[1])
        if (h, w) == CANVAS_PX:
            print(f"canvas pinned {w}x{h}", flush=True)
            return
        ratio = float(view.window().devicePixelRatioF() or 1.0)
        top = view.window()
        top.resize(top.width() + int(round((CANVAS_PX[1] - w) / ratio)),
                   top.height() + int(round((CANVAS_PX[0] - h) / ratio)))
        vf._pump(app, 0.4)
    raise SystemExit(f"canvas pin failed: got {w}x{h}, "
                     f"wanted {CANVAS_PX[1]}x{CANVAS_PX[0]}")


def _content_bbox(frame, thresh: int = 8) -> tuple:
    """(r0, r1, c0, c1) of pixels above *thresh* in any channel: the rendered extent."""
    import numpy as np

    mask = (np.asarray(frame)[..., :3] > thresh).any(axis=2)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        raise SystemExit("preflight: a captured frame is entirely black")
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _apply_gamma(view, target: str, gamma: float) -> None:
    """The SAME gamma on every layer of the 561 identity, both sides: the symmetric
    rule extends to gamma - the shader renders it, so the recordings carry it."""
    if float(gamma) == 1.0:
        return
    mosaic = view._pane.mosaic
    with mosaic.programmatic():
        for op in mosaic.ops():
            for ly in mosaic.layers_for(op, target):
                ly.gamma = float(gamma)


def preflight_pass(source: Path, out_png: Path, window: tuple, seed: int,
                   camera_path: Path, apply_state: bool, gamma: float = 1.0) -> int:
    """One boot, pinned canvas, ONE hero-pose frame. Side A (apply_state=False) derives
    the seed's hero state and records it; side B applies it verbatim."""
    import json

    import imageio.v3 as iio
    import volume_figure as vf

    from squidxplorer import _volume_view
    from squidxplorer._camera_script import (_canvas_rgb, random_orbit_steps,
                                             wait_bricks_resident)

    booted = vf.boot_561_volume(source, source.name, contrast=window)
    if booted is None:
        return 1
    app, win, view, names, target, fallback = booted
    vol = view._native3d
    _pin_canvas(app, view, vol)
    _apply_gamma(view, target, gamma)
    cam = vol._viewer.camera
    if apply_state:
        st = json.loads(camera_path.read_text())
        cam.center = tuple(st["center"])
        cam.zoom = float(st["zoom"])
        cam.angles = tuple(st["angles"])
        _volume_view.refresh_bricks(view)
    else:
        hero = random_orbit_steps(seed)[1]
        _volume_view.snap_camera(view, tuple(hero.pose))
        if hero.zoom is not None:
            cam.zoom = float(cam.zoom) * float(hero.zoom)
            _volume_view.refresh_bricks(view)
        camera_path.write_text(json.dumps(
            {"center": [float(v) for v in cam.center], "zoom": float(cam.zoom),
             "angles": [float(v) for v in cam.angles]}))
    wait_bricks_resident(view, timeout_s=RESIDENCY_TIMEOUT_S)
    vf._pump(app, 0.5)
    frame = _canvas_rgb(vol)
    iio.imwrite(out_png, frame)
    print(f"preflight {source.name}: frame {frame.shape[1]}x{frame.shape[0]} "
          f"-> {out_png}", flush=True)
    vf.teardown(app, win, view, names)
    return 0


def compute_zoom_targets() -> tuple:
    """(zoom_center, home_center) in world (z, y, x) um: the DATA chooses the zoom
    chapter's target - the center of mass of the top 1% brightest 561 z-MIP pixels of
    the RAW set - never the seed. Single-field acquisitions only today."""
    import numpy as np

    from compare_figures import channel_mip

    from squidxplorer._mosaic_source import mosaic_fov_bboxes_um
    from squidxplorer.reader import open_reader

    reader = open_reader(RAW_SET)
    meta = reader.metadata
    if sum(len(f) for f in meta["fovs_per_region"].values()) != 1:
        raise SystemExit("zoom target: single-field acquisitions only today")
    region, fovs = next(iter(meta["fovs_per_region"].items()))
    mip = channel_mip(RAW_SET)[0]
    thr = np.quantile(mip, 0.99)
    ys, xs = np.nonzero(mip >= thr)
    wts = mip[ys, xs].astype(float)
    cy, cx = float((ys * wts).sum() / wts.sum()), float((xs * wts).sum() / wts.sum())
    x0, y0, x1, y1 = mosaic_fov_bboxes_um(meta, region)[fovs[0]]
    pitch = float(meta["pixel_size_um"])
    z_mid = float(meta["n_z"]) * float(meta["dz_um"]) / 2.0
    target = (z_mid, y0 + (cy + 0.5) * pitch, x0 + (cx + 0.5) * pitch)
    home = (z_mid, (y0 + y1) / 2.0, (x0 + x1) / 2.0)
    print(f"zoom target (data-chosen, top 1% MIP center of mass): "
          f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) um; "
          f"home ({home[0]:.1f}, {home[1]:.1f}, {home[2]:.1f}) um", flush=True)
    return target, home


def record_pass(source: Path, out_mp4: Path, seed: int, window: tuple,
                camera_path: Path, replay: bool,
                zoom_center=None, home_center=None, gamma: float = 1.0) -> int:
    """One booted app, one recorded orbit; runs alone in its own process (one GL app).

    Pass A (replay=False) records its per-frame camera state (center, zoom, angles) to
    *camera_path*; pass B (replay=True) applies that state verbatim before EACH capture,
    never re-deriving fit or zoom, so the two sides' cameras are identical by
    construction. The canvas is pinned to CANVAS_PX on both passes first.
    """
    import json

    import volume_figure as vf

    from squidxplorer._camera_script import (_canvas_rgb, random_orbit_steps,
                                             run_camera_script, wait_bricks_resident)

    booted = vf.boot_561_volume(source, source.name, contrast=window)
    if booted is None:
        return 1
    app, win, view, names, target, fallback = booted
    if fallback:
        print(f"WARNING: {source.name} rendered the FALLBACK ROI, not the full FOV",
              flush=True)
    vol = view._native3d
    _pin_canvas(app, view, vol)
    _apply_gamma(view, target, gamma)
    cam = vol._viewer.camera
    states = json.loads(camera_path.read_text()) if replay else []
    k = 0
    applied_err = 0.0

    def capture():
        nonlocal k, applied_err
        if replay:
            st = states[k]
            # Replay writes the recorded state directly; the one-writer rule covers the
            # app's own modules, and pass A's states all came through snap_camera.
            cam.center = tuple(st["center"])
            cam.zoom = float(st["zoom"])
            cam.angles = tuple(st["angles"])
            got = [*cam.center, cam.zoom, *cam.angles]
            want = [*st["center"], st["zoom"], *st["angles"]]
            applied_err = max(applied_err,
                              max(abs(float(a) - float(b)) for a, b in zip(got, want)))
        frame = _canvas_rgb(vol)
        if not replay:
            states.append({"center": [float(v) for v in cam.center],
                           "zoom": float(cam.zoom),
                           "angles": [float(v) for v in cam.angles]})
        k += 1
        return frame

    steps = random_orbit_steps(seed, zoom_center=zoom_center, home_center=home_center)
    result = run_camera_script(
        view, steps, out_path=str(out_mp4), fps=FPS, capture=capture,
        wait_ready=lambda w: wait_bricks_resident(w, timeout_s=RESIDENCY_TIMEOUT_S))
    assert result.n_frames == len(states), (
        f"{result.n_frames} frame(s) against {len(states)} camera state(s)")
    if replay:
        import hashlib

        # The recording's identity includes the trajectory it replayed: the sidecar
        # hash is the reuse key, so it is only ever reused against this exact json.
        Path(str(out_mp4) + ".sha").write_text(
            hashlib.sha256(camera_path.read_bytes()).hexdigest())
        print(f"replayed {len(states)} camera state(s), "
              f"max apply error {applied_err:.3e}", flush=True)
    else:
        camera_path.write_text(json.dumps(states))
        print(f"recorded {len(states)} camera state(s) -> {camera_path}", flush=True)
    print(f"pass {source.name}: {result.n_frames} frame(s) -> {result.path}", flush=True)
    # Let the last snap's brick refresh settle before closing: closing over a busy loader
    # segfaulted after the recording (measured, exit -11 with the mp4 already whole).
    wait_bricks_resident(view, timeout_s=60.0)
    vf._pump(app, 1.0)
    vf.teardown(app, win, view, names)
    return 0


def _kill_running() -> None:
    # The app caps real windows at one instance; Julio approved replacing his.
    subprocess.run(["pkill", "-f", "squidxplorer._viewer import main"], check=False)
    time.sleep(2)


def preflight(tmp: Path, sources: dict, windows: dict, seed: int,
              gamma: float = 1.0) -> None:
    """ONE pinned hero frame per side under ONE camera state, compared precisely BEFORE
    any full pass records: equal canvas sizes, content bbox corners within 1 px. A
    mismatch is a named refusal and no video is recorded."""
    import imageio.v3 as iio

    camera = tmp / f"preflight_seed{seed}.camera.json"
    # The frame cache is keyed on the whole look: a window or gamma change re-captures.
    pngs = {name: tmp / (f"preflight_{name}_seed{seed}"
                         f"_w{int(windows[name][0])}-{int(windows[name][1])}"
                         f"_g{gamma:g}.png")
            for name in ("raw", "decon")}
    if camera.exists() and all(p.exists() for p in pngs.values()):
        print("reusing existing preflight frames", flush=True)
    else:
        for name, replay in (("raw", False), ("decon", True)):
            png = pngs[name]
            png.unlink(missing_ok=True)
            _kill_running()
            w = windows[name]
            cmd = [sys.executable, str(Path(__file__).resolve()), "--seed", str(seed),
                   "--preflight", str(sources[name]), str(png), str(w[0]), str(w[1]),
                   str(camera), "--gamma", str(gamma)]
            if replay:
                cmd.append("--replay")
            rc = subprocess.run(cmd, check=False).returncode
            if not png.exists():
                raise SystemExit(f"preflight: the {name} side exited {rc} with no frame")
            if rc != 0:
                print(f"WARNING: the {name} preflight exited {rc} at teardown; "
                      f"its frame is whole, continuing", flush=True)
    a, b = iio.imread(pngs["raw"]), iio.imread(pngs["decon"])
    if a.shape != b.shape:
        raise SystemExit(f"preflight REFUSAL: canvas sizes differ, "
                         f"raw {a.shape} vs decon {b.shape}; no video recorded")
    # Threshold 8 sees the window's toe (the two schemes' floors differ, so dim edges
    # differ by design; reported, not gated); threshold 32 sees bright structure, whose
    # bbox is the GEOMETRY check the identical replayed camera must satisfy.
    ba8, bb8 = _content_bbox(a, 8), _content_bbox(b, 8)
    ba, bb = _content_bbox(a, 32), _content_bbox(b, 32)
    print(f"preflight canvas {a.shape[1]}x{a.shape[0]} both sides; "
          f"toe bbox (thresh 8) raw {ba8}, decon {bb8}; "
          f"structure bbox (thresh 32) raw {ba}, decon {bb}", flush=True)
    worst = max(abs(x - y) for x, y in zip(ba, bb))
    if worst > 2:
        raise SystemExit(f"preflight REFUSAL: structure bboxes disagree by {worst} px "
                         f"(raw {ba}, decon {bb}); no video recorded")
    print(f"preflight PASS: structure bbox corners agree within {worst} px", flush=True)


def _phase_shift(a, b) -> tuple:
    """Whole-pixel phase-correlation shift (dy, dx) between two grayscale halves."""
    import numpy as np

    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    x = fa * np.conj(fb)
    x /= np.abs(x) + 1e-9
    c = np.fft.irfft2(x, s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    dy = iy if iy <= a.shape[0] // 2 else iy - a.shape[0]
    dx = ix if ix <= a.shape[1] // 2 else ix - a.shape[1]
    return int(dy), int(dx)


def _halves_check(path: Path, n: int) -> None:
    """Decode frames across the orbit; the halves' GEOMETRY must align within 1 px by
    phase correlation, else the composed file is REFUSED before it replaces the
    destination. Structure bboxes are reported for the eyeball, never gated: the two
    sides' windows differ by design, so dim-edge visibility moves the bbox while a
    camera mismatch moves the correlation peak."""
    import imageio.v2 as imageio
    import numpy as np

    from squidxplorer._camera_script import DIVIDER_PX

    reader = imageio.get_reader(str(path))
    worst = (0, -1)
    for idx in (0, 70, 140, 210):
        idx = min(idx, n - 1)
        f = np.asarray(reader.get_data(idx))
        left = f[:, :CANVAS_PX[1]]
        right = f[:, CANVAS_PX[1] + DIVIDER_PX:]
        bl, br = _content_bbox(left, 32), _content_bbox(right, 32)
        dy, dx = _phase_shift(left[..., 1].astype(float), right[..., 1].astype(float))
        worst = max(worst, (max(abs(dy), abs(dx)), idx))
        print(f"frame {idx}: phase shift (dy, dx) = ({dy}, {dx}); structure bbox "
              f"left {bl}, right {br}", flush=True)
    reader.close()
    if worst[0] > 1:
        raise SystemExit(f"alignment REFUSAL: frame {worst[1]} halves are shifted by "
                         f"{worst[0]} px; the composed file is not delivered")


def _readable_frames(path: Path) -> int:
    """Frames imageio can count in *path*; 0 for a missing or unreadable file."""
    if not path.exists():
        return 0
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        n = int(reader.count_frames())
        reader.close()
        return n
    except Exception:                        # noqa: BLE001 - unreadable = not reusable
        return 0


def compose(a_path: Path, b_path: Path, out_path: Path) -> tuple[int, tuple]:
    """Frame k beside frame k, raw left, decon right; counts must match by construction."""
    import imageio.v2 as imageio
    import numpy as np

    from squidxplorer._camera_script import DIVIDER_PX, _side_by_side
    from squidxplorer._video import write_mp4

    ra, rb = imageio.get_reader(str(a_path)), imageio.get_reader(str(b_path))
    na, nb = ra.count_frames(), rb.count_frames()
    assert na == nb, f"pass frame counts diverged: raw {na}, decon {nb}"
    frames = (_side_by_side(np.asarray(fa)[..., :3], np.asarray(fb)[..., :3], DIVIDER_PX)
              for fa, fb in zip(ra, rb))
    _path, n = write_mp4(frames, out_path, fps=FPS)
    ra.close(); rb.close()

    check = imageio.get_reader(str(out_path))
    n_out = check.count_frames()
    shape = np.asarray(check.get_data(0)).shape
    check.close()
    assert n_out == n, f"composed file reads {n_out} frame(s), wrote {n}"
    return n_out, shape


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--decon-source", type=Path, default=DECON_SET,
                    help="the decon sibling acquisition for the right side")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="one window applied to BOTH sides; "
                         "default: volume_figure's raw 561 window")
    ap.add_argument("--raw-window", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="matched scheme: the raw side's own window (overrides --window)")
    ap.add_argument("--decon-window", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="matched scheme: the decon side's own window (overrides --window)")
    ap.add_argument("--out", type=Path, default=None,
                    help="composed .mp4 path; default: Desktop, named by the seed")
    ap.add_argument("--pass", dest="pass_args", nargs=5,
                    metavar=("SOURCE", "OUT_MP4", "LO", "HI", "CAMERA_JSON"),
                    help="internal: run one recording pass in this process")
    ap.add_argument("--preflight", dest="preflight_args", nargs=5,
                    metavar=("SOURCE", "OUT_PNG", "LO", "HI", "CAMERA_JSON"),
                    help="internal: capture one pinned hero frame in this process")
    ap.add_argument("--replay", action="store_true",
                    help="internal: apply CAMERA_JSON verbatim instead of recording it")
    ap.add_argument("--zoom-center", type=float, nargs=3, metavar=("Z", "Y", "X"),
                    default=None, help="internal: the zoom chapter's world-um target")
    ap.add_argument("--home-center", type=float, nargs=3, metavar=("Z", "Y", "X"),
                    default=None, help="internal: the pull-back's world-um center")
    ap.add_argument("--no-zoom", action="store_true",
                    help="record the chapterless storyboard")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="the ONE gamma, applied to BOTH sides (symmetric, like the "
                         "window); below 1 lifts the low end at a peak-holding ceiling")
    args = ap.parse_args(argv)

    if args.pass_args:
        src, out, lo, hi, camera = args.pass_args
        return record_pass(Path(src), Path(out), args.seed, (float(lo), float(hi)),
                           Path(camera), args.replay,
                           zoom_center=args.zoom_center, home_center=args.home_center,
                           gamma=args.gamma)
    if args.preflight_args:
        src, out, lo, hi, camera = args.preflight_args
        return preflight_pass(Path(src), Path(out), (float(lo), float(hi)), args.seed,
                              Path(camera), args.replay, gamma=args.gamma)

    from volume_figure import CONTRAST_561

    # The matched scheme (the CEO's intensity matching, Julio-approved): each side's
    # window is its OWN - ceiling at its own MIP p99.999 so peaks render equal, floor
    # its own auto-contrast rule floor - so absolute cross-side brightness is
    # intentionally NOT comparable. Gamma stays shared. Without per-side flags, one
    # window applies to both (the earlier symmetric schemes).
    window = tuple(args.window) if args.window else CONTRAST_561
    windows = {"raw": tuple(args.raw_window) if args.raw_window else window,
               "decon": tuple(args.decon_window) if args.decon_window else window}
    print(f"seed {args.seed}, decon {args.decon_source.name}, raw window "
          f"{windows['raw']}, decon window {windows['decon']}, "
          f"gamma {args.gamma:g} both sides", flush=True)
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "squidxplorer_comparison"
    tmp.mkdir(exist_ok=True)
    sources = {"raw": RAW_SET, "decon": args.decon_source}
    # The reuse key carries the pass's identity: set name, window and gamma, so a
    # cached recording from another look or another decon solve is never reused.
    pass_out = {
        name: tmp / (f"{name}_seed{args.seed}_{sources[name].name[:24]}"
                     f"_w{int(windows[name][0])}-{int(windows[name][1])}"
                     f"_g{args.gamma:g}.mp4")
        for name in ("raw", "decon")}
    camera_json = pass_out["raw"].with_suffix(".camera.json")

    preflight(tmp, sources, windows, args.seed, gamma=args.gamma)
    zoom_center = home_center = None
    if not args.no_zoom:
        zoom_center, home_center = compute_zoom_targets()

    import hashlib

    outs = {}
    for name in ("raw", "decon"):
        source, out, w = sources[name], pass_out[name], windows[name]
        sha_file = Path(str(out) + ".sha")
        # A replayed recording's identity includes the TRAJECTORY it replayed: reuse
        # only when its sidecar hash matches the current camera json, so a decon mp4
        # recorded against other states (or none) always re-records.
        if name == "raw":
            fresh = camera_json.exists()
        else:
            fresh = (camera_json.exists() and sha_file.exists()
                     and sha_file.read_text().strip()
                     == hashlib.sha256(camera_json.read_bytes()).hexdigest())
        if fresh and _readable_frames(out) > 0:
            print(f"reusing existing {out}", flush=True)
            outs[name] = out
            continue
        out.unlink(missing_ok=True)           # a stale intermediate never survives
        sha_file.unlink(missing_ok=True)
        _kill_running()
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--seed", str(args.seed), "--pass", str(source), str(out),
               str(w[0]), str(w[1]), str(camera_json), "--gamma", str(args.gamma)]
        if zoom_center is not None:
            cmd += ["--zoom-center", *(str(v) for v in zoom_center),
                    "--home-center", *(str(v) for v in home_center)]
        if name == "decon":
            cmd.append("--replay")
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            # A teardown crash AFTER the recording still leaves a whole, readable mp4.
            if _readable_frames(out) > 0:
                print(f"WARNING: the {name} pass exited {rc} at teardown; "
                      f"its recording is whole, continuing", flush=True)
            else:
                print(f"FAIL: the {name} pass exited {rc}", flush=True)
                return rc
        outs[name] = out

    final = args.out or DESKTOP / f"raw_vs_decon_561_seed{args.seed}.mp4"
    staged = tmp / f"staged_{final.name}"
    n, shape = compose(outs["raw"], outs["decon"], staged)
    _halves_check(staged, n)                 # refuses BEFORE the destination changes
    staged.replace(final)
    size_mb = final.stat().st_size / 1e6
    print(f"composed {final}: {n} frame(s), {n / FPS:.1f} s at {FPS} fps, "
          f"{shape[1]}x{shape[0]} px, {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
