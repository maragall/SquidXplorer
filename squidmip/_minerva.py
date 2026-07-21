"""Minerva Author export (IMA-228): fused region mosaic -> OME-TIFF + .story.json -> launch.

Hands the region(s) the user selected to `Minerva Author <https://github.com/labsyspharm/
minerva-author>`_ without leaving the viewer.

Minerva's unit is ONE FUSED MOSAIC PER REGION
---------------------------------------------
This is the fact the whole module is shaped around, and it was read out of minerva-author's
own source, not assumed:

* ``src/app.py`` emits ``"Layout": {"Grid": [["i0"]]}`` **unconditionally** — a single 1x1
  grid cell. There is no code path that lays out N images.
* ``Opener.__init__`` opens ``self.io.series[0]`` and nothing else.

So handing Minerva a set of per-FOV files cannot work: it silently renders only the first
one. A region is a MOSAIC containing an array of FOVs, and it is that mosaic — fused — that
Minerva ingests. The earlier version of this module exported one file per FOV; that was
provably wrong, not merely suboptimal.

Pipeline
--------
::

    [(region, fov), ...]  selection
         │
         │  group by region ──▶ {region: [fov, ...]}
         │  require pixel_size_um ──── missing ──▶ ValueError (see "Pixel size" below)
         ▼
    stitch_plate(reader, regions={region: [fov, ...]}, operator=..., projector=...)
         │        the IMA-222 region-operator seam. NOT project_plate: that is the
         │        z-reduction path and cannot fuse FOVs into a mosaic.
         │
         │  ONE (T, C, 1, H, W) per region — a FOV subset gives the CROP of the
         │  region spanned by those FOVs, still one mosaic, never N files.
         ▼
    [t, :, 0]  →  (C, H, W) native dtype
         ├──▶ write_ome_tiff()  →  <stem>.ome.tiff    pixels + names + PhysicalSize
         └──▶ auto_groups()     →  write_story()  →  <stem>.story.json    COLOUR + contrast
                                                          │
                                                          ▼
                                              launch_minerva()  (best-effort)
                                                          │
                                              user clicks "Select File"

Why we do not import ``squid2minerva``
--------------------------------------
That package (``~/CEPHLA/projects/explorer``) is not installable: it has no
``pyproject.toml``, its imports resolve only via a ``sys.path`` hack in its own ``run.py``,
and its ``requirements.txt`` hard-pins ``tifffile==2025.5.10`` / ``zarr==2.18.7`` against
SquidMIP's ``tifffile>=2023.1.0``. It has no git tags, so it cannot even be pinned by
version. The parts we need are ~60 lines of pure-array code and ``tifffile`` is already a
hard dependency of this package, so we write them here. See ``docs/ima-228-eng-review.md``.

Minerva Author's ingest contract (undocumented — read out of its ``src/app.py``)
-------------------------------------------------------------------------------
Four hard requirements, each of which fails in a way that is hard to diagnose from the
Minerva side:

* **Colour lives in the story, not the TIFF.** Minerva colours channels *by index* and
  ignores OME-TIFF channel colours outright. The only path for our per-channel colours is
  the ``groups`` block of the ``.story.json``. We still write ``Channel.Color`` into the
  OME-XML because it is correct and other tools read it, but nothing in Minerva does.
* **Pixel size is a gate.** Minerva reads ``PhysicalSizeX`` and returns HTTP 500
  ("Image is missing OME-XML pixel size") when it is absent. SquidMIP's ``pixel_size_um``
  is nullable, and elsewhere (``_output.py``) a missing value degrades to ``1.0`` — which
  is right for a zarr axis transform but *wrong* here: it would silently put a bogus
  physical scale into Minerva. So this module refuses the export instead.
* **The filename is a gate.** Minerva takes the last two extension components of the path;
  anything not ending ``.ome.tif`` / ``.ome.tiff`` is rejected as "Invalid tiff file".
* **Channel names are opaque labels — but an empty one shifts every channel after it.**
  Minerva does *not* parse channel names. ``Opener.load_xml_markers`` returns
  ``[c.name for c in metadata.images[0].pixels.channels if c.name]`` and ``make_channel_labels``
  yields them straight through as display text; there is no regex over them anywhere in
  ``app.py``, ``story.py``, ``render.py`` or ``storyexport.py``. So the failure mode petakit's
  OME-TIFF reader has — emitting a name like ``"488"`` that its own ``wavelength_from_channel``
  regex then cannot parse — has no counterpart here, and SquidMIP's names
  (``"Fluorescence_638_nm_-_Penta"``) are safe as-is.
  What *is* a live hazard is the ``if c.name`` filter: a channel whose name is empty is
  DROPPED from the list, so every later channel is labelled with its predecessor's name while
  the pixel data stays put — a silent mislabel, not an error. :func:`_channel_names` therefore
  refuses to write a blank name.
* **Write it flat.** ``imwrite(path, img, photometric="minisblack", metadata=...)`` — OME is
  inferred from the extension. Do not pass ``ome=True``: Minerva branches on an OME-version
  probe (SubIFDs tag 330) and re-opens the file down a different axis path when the tag is
  absent, which flat single-resolution output relies on. Adding a pyramid would flip that
  branch.
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

from squidmip._engine import _resolve_projector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from squidmip.reader import SquidReader

__all__ = [
    "export_selection",
    "group_selection",
    "write_ome_tiff",
    "auto_groups",
    "write_story",
    "launch_minerva",
    "minerva_home",
    "MINERVA_PORT",
    "MINERVA_URL",
]

# minerva-author binds this port in its own app.py; it is not configurable there.
MINERVA_PORT = 2020
MINERVA_URL = f"http://localhost:{MINERVA_PORT}/"

#: Env var pointing at an ``explorer`` checkout that has run its ``setup.py``. That checkout
#: holds *both* halves we need: ``vendor/minerva-author/src/app.py`` and the ``.venv`` whose
#: interpreter has minerva-author's dependencies (waitress, flask_cors, xsdata, ome-types,
#: openslide-bin, ...). minerva-author has no venv of its own.
MINERVA_HOME_ENV = "SQUIDMIP_MINERVA_HOME"

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
    """Return the acquisition's pixel size, or refuse the export.

    Minerva returns an opaque HTTP 500 when ``PhysicalSizeX`` is missing, so failing here —
    with a message naming the file the user should fix — is strictly kinder than exporting
    something that cannot be opened. Deliberately does *not* reuse ``_output.py``'s
    ``pixel_size_um or 1.0`` fallback: a fabricated scale would make Minerva's measurements
    silently wrong, which is worse than not exporting.
    """
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
    """Write a 2D multichannel OME-TIFF that Minerva Author ingests.

    *img_cyx* is ``(C, Y, X)`` in its native dtype — no rescale, no float cast. *path* must
    end ``.ome.tiff`` or ``.ome.tif`` (Minerva's own extension check rejects anything else).
    """
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
        # Flat, single-resolution, OME inferred from the extension. See the module docstring
        # for why this exact call shape matters to Minerva's OME-version branch.
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
) -> list[dict]:
    """One Minerva group over all channels: colour + auto-stretched contrast.

    Contrast is a 1st-percentile floor and 99.9th-percentile ceiling per channel, normalised
    to 0..1 against the dtype maximum — Minerva's own convention. This is the *only* place
    our channel colours reach Minerva.
    """
    img = np.asarray(img_cyx)
    dtype_max = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    channels = []
    for i, name in enumerate(channel_names):
        plane = img[i].astype(np.float32, copy=False).ravel()
        lo = float(np.percentile(plane, 1.0))
        hi = float(np.percentile(plane, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        r, g, b = channel_colors[i]
        channels.append({
            "id": i,
            "label": name,
            "color": "%02x%02x%02x" % (int(r) & 255, int(g) & 255, int(b) & 255),
            "min": round(max(0.0, lo / dtype_max), 6),
            "max": round(min(1.0, hi / dtype_max), 6),
        })
    return [{"label": label, "channels": channels}]


def write_story(story_path, ome_path, groups: list[dict], pixels_per_micron: float = 0.0):
    """Write a Minerva Author saved-story that pre-loads *groups* for *ome_path*.

    The user opens this file through Author's "Select File" and lands in the editor with our
    colours and contrast already applied — which is the only way to get them there, since
    Minerva ignores OME-TIFF channel colours. ``in_file`` must be absolute: Author resolves
    it from its own working directory, not ours.
    """
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
            "text": "",
            "pixels_per_micron": float(pixels_per_micron),
        },
        "waypoints": [],
        "masks": [],
        "groups": groups,
    }
    story_path.parent.mkdir(parents=True, exist_ok=True)
    story_path.write_text(json.dumps(story, indent=2), encoding="utf-8")
    return story_path


# --- export ----------------------------------------------------------------------------------

def default_out_dir(reader: "SquidReader") -> Path:
    """Where exports go when the caller doesn't say: ``~/minerva_export/<acquisition>``.

    NOT inside the acquisition folder. The tool's standing promise to users is that it never
    writes there (README, "Good to know"), and acquisition volumes are routinely read-only
    network shares — defaulting there would fail exactly where it is least expected. Also not
    a temp dir: Minerva is a separate, long-lived process, and OS sweeping can delete a story
    it still has open. The home directory is writable, discoverable and persistent.
    """
    name = _safe(Path(getattr(reader, "_path", "acquisition")).name)
    return Path.home() / "minerva_export" / name


def group_selection(selection: Iterable[tuple[str, int]]) -> "dict[str, list[int]]":
    """``[(region, fov), ...]`` -> ``{region: [fov, ...]}``, first-seen order, deduplicated.

    The selection the plate hands us is a flat list of pairs, but the EXPORT unit is a region.
    This is the one place that regrouping happens, and keeping it a named function is what
    stops "one file per pair" from creeping back in: everything downstream iterates regions.
    """
    grouped: dict[str, list[int]] = {}
    for region, fov in selection:
        fovs = grouped.setdefault(str(region), [])
        fov = int(fov)
        if fov not in fovs:
            fovs.append(fov)
    return grouped


def _channel_names(channels: Sequence[dict]) -> list[str]:
    """Channel display names, refusing a blank one.

    Minerva drops falsy channel names (``[c.name for c in ... if c.name]``) without shortening
    the pixel data, which silently shifts every later channel's label onto the wrong image. A
    blank name is therefore a mislabel waiting to happen, and we fail here instead.
    """
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
    t: int = 0,
    projector: str = "mip",
    operator: str = "stitch",
    on_progress=None,
    **operator_kwargs,
) -> list[tuple[Path, Path]]:
    """Export the selected region(s) to Minerva-ingestable file pairs — ONE PAIR PER REGION.

    *selection* is ``[(region, fov), ...]`` (what the plate emits). It is grouped by region,
    and each region is fused into a single mosaic through :func:`squidmip.stitch_plate` — the
    IMA-222 region-operator seam — then written as one OME-TIFF plus one ``.story.json``.

    A FOV subset within a region does NOT become N files. It becomes the crop of that region
    spanned by those FOVs: still one mosaic, because Minerva Author lays out exactly one image
    (``"Layout": {"Grid": [["i0"]]}``, hardcoded) and reads only ``series[0]``. Handing it N
    files would silently render the first and discard the rest.

    Returns ``[(ome_path, story_path), ...]``, one per region, in the order the regions first
    appear in *selection*.

    Parameters
    ----------
    t:
        Timepoint to export (default 0). The region operator returns every timepoint; this
        picks the plane written.
    projector:
        Z-reduction applied per FOV *before* fusion (``"mip"``, ``"reference"``, ...). Passed
        through to the region operator, which owns the z axis.
    operator:
        Region-operator name (default ``"stitch"``, i.e. registered fusion; ``"coordinate"``
        places by stage position only). Anything added via ``add_region_operator`` works here
        with no edit to this module — that is the point of the seam.
    on_progress:
        Optional ``fn(done, total)`` called after each REGION, for a GUI readout. ``total`` is
        the number of regions, not FOVs.
    **operator_kwargs:
        Forwarded to the region operator (``blend_px=``, ``channels=``, ``register=``, ...).

    Raises
    ------
    ValueError
        If the selection is empty, the acquisition has no pixel size, a ``(region, fov)`` is
        not in the acquisition, a channel has no name, or *t* is out of range. All are raised
        *before* anything is written.
    """
    from squidmip._stitch import stitch_plate   # local: avoids an import cycle at module load

    grouped = group_selection(selection)
    if not grouped:
        raise ValueError("nothing selected: export_selection needs at least one (region, fov)")

    meta = reader.metadata
    pixel_um = _require_pixel_size(meta)                  # refuse early — nothing written yet
    _resolve_projector(projector)     # unknown projector: fail here, named, not mid-stitch

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
    if not 0 <= t < n_t:
        raise ValueError(f"t={t} is out of range: this acquisition has {n_t} timepoint(s)")

    channels = meta["channels"]
    names = _channel_names(channels)
    colors = [_hex_to_rgb(c.get("display_color")) for c in channels]
    ppm = 1.0 / pixel_um if pixel_um else 0.0

    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(reader)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_prefix = _safe(Path(getattr(reader, "_path", "acquisition")).name)

    # workers=1: peak memory is `workers x one fused mosaic`, and a mosaic is orders of
    # magnitude larger than the single FOV this used to hold. Fusion is internally parallel
    # anyway, so one region in flight still saturates the CPU (see stitch_plate).
    written: dict[str, tuple[Path, Path]] = {}
    stream = stitch_plate(
        reader, regions=grouped, workers=1, operator=operator,
        projector=projector, **operator_kwargs,
    )
    for region, _anchor_fov, image in stream:
        # Stream: fuse one region, write it, drop it.
        img_cyx = np.asarray(image[t, :, 0])
        fovs = grouped[region]
        whole = len(fovs) == len(fovs_per_region.get(region, []))
        stem = f"{stem_prefix}_{_safe(region)}_t{t}_{_safe(projector)}_{_safe(operator)}"
        if not whole:      # a crop, not the region — say so in the filename, not just the story
            stem += f"_{len(fovs)}fov"
        label = region if whole else f"{region} ({len(fovs)} FOVs)"
        ome_path = write_ome_tiff(img_cyx, out_dir / f"{stem}.ome.tiff", names, pixel_um, colors)
        story_path = write_story(
            out_dir / f"{stem}.story.json",
            ome_path,
            auto_groups(img_cyx, names, colors, label=label),
            pixels_per_micron=ppm,
        )
        written[region] = (ome_path, story_path)
        del image, img_cyx
        if on_progress is not None:
            on_progress(len(written), len(grouped))
    # stitch_plate yields in COMPLETION order; the caller asked in selection order.
    return [written[r] for r in grouped if r in written]


# --- launch ----------------------------------------------------------------------------------

def minerva_home() -> Optional[Path]:
    """The ``explorer`` checkout that provides minerva-author, or ``None``.

    Read from ``$SQUIDMIP_MINERVA_HOME``, else the conventional sibling checkout. Returns a
    path only if it actually has *both* halves — the app and the venv interpreter — since
    minerva-author carries no venv of its own and cannot run under ours.
    """
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
    """Start minerva-author if it isn't up, then open the browser. Best-effort.

    Returns ``True`` when a server is answering. **Never raises** — the export has already
    succeeded by the time this is called, and a missing sibling checkout must not turn a
    successful export into a failure. The caller reports the outcome and always shows the
    user the story path, because Minerva has no deep link: the file is chosen by hand in
    Author's "Select File" dialog.

    Parameters
    ----------
    should_stop:
        Optional ``fn() -> bool`` polled while waiting for the server. The liveness wait is
        up to *timeout* seconds long, and a GUI that joins this thread on close (``closeEvent``
        -> ``QThread.wait()``) would freeze for the remainder of it — measured at 84 s. The
        viewer passes its worker's stop flag here so closing abandons the wait at once. The
        files are already on disk; only the wait is abandoned.
    """
    import time
    import webbrowser

    stopped = should_stop if callable(should_stop) else (lambda: False)
    if stopped():
        return False

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
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stopped():        # the caller (a GUI closing) gave up — do not hold it for 90 s
                return False
            if is_running():
                break
            # Short naps, not one long one: the stop flag is honoured within ~0.2 s instead of
            # up to a second, which is what makes closing the window feel immediate.
            time.sleep(0.2)
        else:
            return False

    if stopped():
        return False
    if open_browser:
        try:
            webbrowser.open(MINERVA_URL)
        except Exception:
            pass          # the server is up; a browser that won't open is not a failure
    return True


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
