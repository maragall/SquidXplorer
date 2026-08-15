"""Minerva Author export: fused region mosaic -> OME-TIFF + .story.json -> launch.

Minerva's unit is ONE fused mosaic per region (its layout is a hardcoded 1x1 grid and it
reads only ``series[0]``), so a FOV subset becomes a crop, never N files.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np
import tifffile

from squidxplorer._engine import _resolve_operator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from squidxplorer.reader import SquidReader

__all__ = [
    "export_selection",
    "group_selection",
    "write_ome_tiff",
    "auto_groups",
    "write_story",
    "launch_minerva",
    "render_exhibit",
    "render_script",
    "open_exhibit",
    "minerva_home",
    "MINERVA_PORT",
    "MINERVA_URL",
    "NEEDS_INTERNET_NOTE",
]

# minerva-author binds this port in its own app.py; it is not configurable there.
MINERVA_PORT = 2020
MINERVA_URL = f"http://localhost:{MINERVA_PORT}/"

#: Env var pointing at an ``explorer`` checkout holding minerva-author and its venv.
MINERVA_HOME_ENV = "SQUIDXPLORER_MINERVA_HOME"

_OME_SUFFIXES = (".ome.tiff", ".ome.tif")


# --- helpers ---------------------------------------------------------------------------------

def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """``"#FF0000"`` / ``"ff0000"`` -> ``(255, 0, 0)``. Falls back to grey on anything odd."""
    h = str(value or "").lstrip("#").strip()
    if len(h) != 6:
        return (200, 200, 200)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (200, 200, 200)


def _ome_color(rgb: tuple[int, int, int]) -> int:
    """OME ``Channel.Color`` is a signed int32 RGBA."""
    r, g, b = (int(v) & 255 for v in rgb)
    v = (r << 24) | (g << 16) | (b << 8) | 255
    return v - (1 << 32) if v >= (1 << 31) else v


def _safe(name: str) -> str:
    """Filesystem-safe token for a region/acquisition name."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name)) or "x"


def _require_pixel_size(metadata: dict) -> float:
    """Return the acquisition's pixel size, or refuse the export (Minerva 500s without it)."""
    px = metadata.get("pixel_size_um")
    if not px:
        raise ValueError(
            "cannot export to Minerva: this acquisition has no objective pixel size "
            "(acquisition.yaml -> objective.pixel_size_um). Minerva Author rejects an "
            "OME-TIFF without PhysicalSizeX, and substituting a placeholder would put a "
            "wrong physical scale into every measurement made from it."
        )
    return float(px)


# --- writers ---------------------------------------------------------------------------------

def write_ome_tiff(
    img_cyx: np.ndarray,
    path,
    channel_names: Sequence[str],
    pixel_um: float,
    channel_colors: Optional[Sequence[tuple[int, int, int]]] = None,
):
    """Write a 2D multichannel OME-TIFF that Minerva Author ingests."""
    path = Path(path)
    if not str(path).lower().endswith(_OME_SUFFIXES):
        raise ValueError(
            f"Minerva requires an OME-TIFF path ending in {' or '.join(_OME_SUFFIXES)}; got {path.name!r}. "
            "Its reader takes the last two extension components and rejects the file otherwise."
        )
    img = np.asarray(img_cyx)
    if img.ndim != 3:
        raise ValueError(f"expected a (C, Y, X) array, got shape {img.shape}")
    if img.shape[0] != len(channel_names):
        raise ValueError(
            f"image has C={img.shape[0]} but {len(channel_names)} channel names "
            f"({list(channel_names)}) — refusing to mislabel the OME-XML."
        )

    meta = {
        "axes": "CYX",
        "Channel": {"Name": list(channel_names)},
        "PhysicalSizeX": float(pixel_um), "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": float(pixel_um), "PhysicalSizeYUnit": "µm",
    }
    if channel_colors:
        meta["Channel"]["Color"] = [_ome_color(c) for c in channel_colors]

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Flat, single-resolution, OME inferred from the extension: a pyramid or ome=True
        # flips Minerva's OME-version branch.
        tifffile.imwrite(str(path), img, photometric="minisblack", metadata=meta)
    except Exception:
        meta["Channel"].pop("Color", None)   # older tifffile rejects Channel.Color
        tifffile.imwrite(str(path), img, photometric="minisblack", metadata=meta)
    return path


def auto_groups(
    img_cyx: np.ndarray,
    channel_names: Sequence[str],
    channel_colors: Sequence[tuple[int, int, int]],
    label: str = "All channels",
    luts: Optional[dict] = None,
) -> list[dict]:
    """One Minerva group over all channels: colour + contrast, per-channel LUT overrides."""
    img = np.asarray(img_cyx)
    dtype_max = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    luts = luts or {}
    channels = []
    for i, name in enumerate(channel_names):
        lut = luts.get(name) or {}
        clim = lut.get("clim")
        if clim is not None and len(tuple(clim)) == 2:
            lo, hi = float(tuple(clim)[0]), float(tuple(clim)[1])
        else:
            plane = img[i].astype(np.float32, copy=False).ravel()
            lo = float(np.percentile(plane, 1.0))
            hi = float(np.percentile(plane, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        r, g, b = lut.get("rgb") or channel_colors[i]
        channels.append({
            "id": i,
            "label": name,
            "color": "%02x%02x%02x" % (int(r) & 255, int(g) & 255, int(b) & 255),
            "min": round(max(0.0, lo / dtype_max), 6),
            "max": round(min(1.0, hi / dtype_max), 6),
        })
    return [{"label": label, "channels": channels}]


def write_story(story_path, ome_path, groups: list[dict], pixels_per_micron: float = 0.0,
                provenance: str = ""):
    """Write a Minerva Author saved-story that pre-loads *groups* for *ome_path*."""
    story_path, ome_path = Path(story_path), Path(ome_path).resolve()
    dataset = ome_path.name
    for suffix in _OME_SUFFIXES:
        if dataset.lower().endswith(suffix):
            dataset = dataset[: -len(suffix)]
            break
    story = {
        "in_file": str(ome_path),
        "csv_file": "",
        "root_dir": str(ome_path.parent),
        "out_name": dataset,
        "sample_info": {
            "name": dataset,
            "rotation": 0,
            "text": provenance,
            "pixels_per_micron": float(pixels_per_micron),
        },
        "waypoints": [],
        "masks": [],
        "groups": groups,
    }
    story_path.parent.mkdir(parents=True, exist_ok=True)
    story_path.write_text(json.dumps(story, indent=2), encoding="utf-8")
    return story_path


def _provenance_text(image, z_operator: str, operator: str, *, region: str = "",
                     time_point: Optional[int] = None, fovs: Optional[Sequence[int]] = None,
                     n_fovs: Optional[int] = None) -> str:
    """One line saying what produced these pixels, for the story's ``sample_info.text``."""
    parts = [f"squidxplorer {operator}/{z_operator}"]
    if region:
        parts.append(f"region {region}")
    if time_point is not None:
        parts.append(f"timepoint t={int(time_point)}")
    if fovs is not None:
        got = len(list(fovs))
        parts.append(f"all {got} FOV(s)" if n_fovs in (None, got)
                     else f"CROPPED to {got} of {n_fovs} FOV(s): {sorted(fovs)}")
    p = getattr(image, "placement", None)
    if p is not None:
        if p.registered:
            parts.append(f"registered on {p.reg_channel} (z={p.reg_z}, t={p.reg_t})")
        else:
            parts.append("coordinate placement (no registration)")
        parts.append("flat-field corrected" if p.illumination_corrected
                     else "no flat-field correction")
    return "; ".join(parts)


# --- export ----------------------------------------------------------------------------------

def default_out_dir(reader: "SquidReader") -> Path:
    """Default export location, ``~/minerva_export/<acquisition>`` — never inside the acquisition."""
    name = _safe(Path(getattr(reader, "_path", "acquisition")).name)
    return Path.home() / "minerva_export" / name


def group_selection(selection: Iterable[tuple[str, int]]) -> "dict[str, list[int]]":
    """``[(region, fov), ...]`` -> ``{region: [fov, ...]}``, first-seen order, deduplicated."""
    grouped: dict[str, list[int]] = {}
    for region, fov in selection:
        fovs = grouped.setdefault(str(region), [])
        fov = int(fov)
        if fov not in fovs:
            fovs.append(fov)
    return grouped


def _channel_names(channels: Sequence[dict]) -> list[str]:
    """Channel display names, refusing a blank one (Minerva silently mislabels after a blank)."""
    names = [str(c.get("name") or "").strip() for c in channels]
    blank = [i for i, n in enumerate(names) if not n]
    if blank:
        raise ValueError(
            f"channel(s) at index {blank} have no name. Minerva Author drops unnamed channels "
            "from its label list but not from the image, which would put every later channel's "
            "name on the wrong one. Name them in acquisition_channels.yaml and re-export."
        )
    return names


def export_selection(
    reader: "SquidReader",
    selection: Iterable[tuple[str, int]],
    out_dir=None,
    *,
    time_point: int = 0,
    z_operator: str = "mip",
    operator: str = "stitch",
    on_progress=None,
    luts: Optional[dict] = None,
    **operator_kwargs,
) -> list[tuple[Path, Path]]:
    """Export the selected region(s) to Minerva-ingestable file pairs — one pair per region."""
    from squidxplorer._stitch import _stitch_plate   # local: avoids an import cycle at module load

    grouped = group_selection(selection)
    if not grouped:
        raise ValueError("nothing selected: export_selection needs at least one (region, fov)")

    meta = reader.metadata
    pixel_um = _require_pixel_size(meta)                  # refuse early — nothing written yet
    _resolve_operator(z_operator)    # unknown z operator: fail here, named, not mid-stitch

    fovs_per_region = meta.get("fovs_per_region", {})
    for region, fovs in grouped.items():
        if region not in fovs_per_region:
            raise ValueError(f"unknown region {region!r}; acquisition has {sorted(fovs_per_region)}")
        for fov in fovs:
            if fov not in fovs_per_region[region]:
                raise ValueError(
                    f"unknown fov {fov} for region {region!r}; available: {fovs_per_region[region]}"
                )

    n_t = int(meta.get("n_t", 1) or 1)
    if not 0 <= time_point < n_t:
        raise ValueError(f"t={time_point} is out of range: this acquisition has {n_t} timepoint(s)")

    channels = meta["channels"]
    names = _channel_names(channels)
    colors = [_hex_to_rgb(c.get("display_color")) for c in channels]
    ppm = 1.0 / pixel_um if pixel_um else 0.0

    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(reader)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_prefix = _safe(Path(getattr(reader, "source_id", None)
                             or getattr(reader, "_path", "acquisition")).name)

    # workers=1: peak memory is workers x one fused mosaic, and fusion is internally parallel.
    # z_operator is forwarded only when the RECORD takes it (declared or accepted) — the one
    # kwargs contract; an operator that cannot honour a non-default choice refuses by name here
    # rather than silently exporting under a label it ignored.
    from squidxplorer._engine import operator_accepts, operator_params

    takes_z = ("z_operator" in {p.name for p in operator_params(operator)}
               or "z_operator" in operator_accepts(operator))
    op_kwargs = dict(operator_kwargs)
    # export_selection's OWN signature default — comparing against it detects "the caller chose",
    # never a branch on an operator's name.
    z_default = export_selection.__kwdefaults__["z_operator"]
    if takes_z:
        op_kwargs.setdefault("z_operator", z_operator)
    elif z_operator != z_default:
        raise ValueError(
            f"operator {operator!r} takes no z_operator, so the export cannot honour "
            f"z_operator={z_operator!r}; it would run the operator's own z handling and "
            "label the file with a choice that never happened.")
    written: dict[str, tuple[Path, Path]] = {}
    stream = _stitch_plate(
        reader, regions=grouped, workers=1, operator=operator, **op_kwargs,
    )
    for region, _anchor_fov, image in stream:
        # Stream: fuse one region, write it, drop it.
        img_cyx = np.asarray(image[time_point, :, 0])
        fovs = grouped[region]
        whole = len(fovs) == len(fovs_per_region.get(region, []))
        stem = f"{stem_prefix}_{_safe(region)}_t{time_point}_{_safe(z_operator)}_{_safe(operator)}"
        if not whole:      # a crop, not the region — say so in the filename, not just the story
            stem += f"_{len(fovs)}fov"
        label = region if whole else f"{region} ({len(fovs)} FOVs)"
        ome_path = write_ome_tiff(img_cyx, out_dir / f"{stem}.ome.tiff", names, pixel_um, colors)
        story_path = write_story(
            out_dir / f"{stem}.story.json",
            ome_path,
            auto_groups(img_cyx, names, colors, label=label, luts=luts),
            pixels_per_micron=ppm,
            provenance=_provenance_text(
                image, z_operator, operator, region=region, time_point=time_point, fovs=fovs,
                n_fovs=len(fovs_per_region.get(region, []) or []) or None),
        )
        written[region] = (ome_path, story_path)
        del image, img_cyx
        if on_progress is not None:
            on_progress(len(written), len(grouped))
    # the region loop yields in COMPLETION order; the caller asked in selection order.
    return [written[r] for r in grouped if r in written]


# --- launch ----------------------------------------------------------------------------------

def minerva_home() -> Optional[Path]:
    """The ``explorer`` checkout that provides minerva-author, or ``None``."""
    candidates = []
    env = os.environ.get(MINERVA_HOME_ENV)
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.home() / "CEPHLA" / "projects" / "explorer")
    for root in candidates:
        if (root / "vendor" / "minerva-author" / "src" / "app.py").is_file():
            return root
    return None


def _minerva_parts(home: Path) -> tuple[Path, Optional[Path]]:
    """``(app.py, interpreter)`` for a checkout. Interpreter is ``None`` if its venv is absent."""
    app = home / "vendor" / "minerva-author" / "src" / "app.py"
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        py = home / rel
        if py.is_file():
            return app, py
    return app, None


def is_running(timeout: float = 1.0) -> bool:
    """Is something already answering on minerva-author's port?"""
    try:
        with urllib.request.urlopen(MINERVA_URL, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def launch_minerva(story_path=None, *, open_browser: bool = True, timeout: float = 90.0,
                   should_stop=None) -> bool:
    """Start minerva-author if it isn't up, then open the browser. Best-effort, never raises.

    Whoever started the server opens the tab: Author unconditionally opens its own on a
    cold start, so we only open one when the server was already answering.
    """
    import time
    import webbrowser

    stopped = should_stop if callable(should_stop) else (lambda: False)
    if stopped():
        return False

    we_started_it = False
    if not is_running():
        home = minerva_home()
        if home is None:
            return False
        app, python = _minerva_parts(home)
        if python is None or not app.is_file():
            return False
        try:
            log = open(home / "vendor" / "minerva-author" / "server.log", "ab")
        except OSError:
            log = subprocess.DEVNULL
        try:
            subprocess.Popen(
                [str(python), str(app)],
                cwd=str(app.parent.parent),
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            return False
        we_started_it = True     # ...and it is about to open its own tab
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stopped():        # the caller (a GUI closing) gave up — do not hold it for 90 s
                return False
            if is_running():
                break
            time.sleep(0.2)
        else:
            return False

    if stopped():
        return False
    if open_browser and not we_started_it:
        try:
            webbrowser.open(MINERVA_URL)
        except Exception:
            pass          # the server is up; a browser that won't open is not a failure
    return True


#: Both Minerva front ends load their viewer JavaScript from jsdelivr, so VIEWING needs
#: internet; the rendered tiles themselves are local.
NEEDS_INTERNET_NOTE = (
    "Minerva's viewer JavaScript is loaded from a CDN (jsdelivr), so viewing needs internet - "
    "the rendered tiles themselves are local."
)


def render_script(home: Path) -> Path:
    """``src/render.py`` inside a checkout. Beside :func:`_minerva_parts`' ``src/app.py``."""
    return Path(home) / "vendor" / "minerva-author" / "src" / "render.py"


def render_exhibit(ome_path, story_path, out_dir, *, threads: Optional[int] = None,
                   force: bool = True, should_stop=None, timeout: float = 3600.0) -> Path:
    """Render *ome_path* + *story_path* into a viewable Minerva exhibit; returns its ``index.html``."""
    import time

    stopped = should_stop if callable(should_stop) else (lambda: False)
    home = minerva_home()
    if home is None:
        raise FileNotFoundError(
            f"Minerva Author checkout not found (set ${MINERVA_HOME_ENV} to an explorer checkout)")
    _app, python = _minerva_parts(home)
    script = render_script(home)
    if python is None:
        raise FileNotFoundError(f"{home} has no .venv interpreter; render.py needs Minerva's own venv")
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found; this checkout has no renderer")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(threads) if threads else max(1, min(8, os.cpu_count() or 1))
    cmd = [str(python), str(script), str(Path(ome_path).resolve()),
           str(Path(story_path).resolve()), str(out_dir.resolve()), "--threads", str(n)]
    if force:
        cmd.append("--force")

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(script.parent.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"could not start render.py: {exc}") from exc

    # Poll rather than communicate(): a render is minutes long and the GUI thread that
    # closes the window must not be held for it.
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if stopped() or time.monotonic() > deadline:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:                    # noqa: BLE001 - best effort, then leave it
                proc.kill()
            raise RuntimeError("the Minerva render was stopped before it finished")
        time.sleep(0.2)

    output = ""
    try:
        output = proc.stdout.read() if proc.stdout is not None else ""
    except Exception:                            # noqa: BLE001 - the exit code is the fact
        pass
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    if proc.returncode != 0:
        tail = "\n".join([ln for ln in (output or "").splitlines() if ln.strip()][-6:])
        raise RuntimeError(
            f"render.py exited {proc.returncode}" + (f":\n{tail}" if tail else " with no output"))

    index = out_dir / "index.html"
    if not index.is_file():
        raise RuntimeError(f"render.py exited 0 but wrote no index.html in {out_dir}")
    return index


def open_exhibit(index_path) -> bool:
    """Open a rendered exhibit's ``index.html`` in the browser. Best-effort, never raises."""
    import webbrowser
    try:
        return bool(webbrowser.open(Path(index_path).resolve().as_uri()))
    except Exception:                            # noqa: BLE001
        return False


def reveal(path) -> None:
    """Show *path* in the OS file manager. Best-effort, never raises."""
    path = Path(path)
    try:
        if shutil.which("open"):                       # macOS
            subprocess.Popen(["open", "-R", str(path)])
        elif shutil.which("explorer.exe"):             # Windows
            subprocess.Popen(["explorer.exe", "/select,", str(path)])
        elif shutil.which("xdg-open"):                 # Linux — no per-file select
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError:
        pass
