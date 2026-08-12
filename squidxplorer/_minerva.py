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
         └──▶ auto_groups(luts=)  →  write_story()  →  <stem>.story.json   COLOUR + contrast
                                                          │
                              ┌───────────────────────────┴────────────────────────┐
                              ▼                                                    ▼
                   launch_minerva()  (best-effort)                        render_exhibit()
                              │                                                    │
                   user clicks "Select File"                          open_exhibit(index.html)
                    THE EDITOR. Needs the click.                    THE VIEWER. Needs no click.

TWO DESTINATIONS, AND ONLY ONE OF THEM NEEDS A CLICK
----------------------------------------------------
The click cannot be removed from Author. Its server reads ``sys.argv`` exactly once and only to
test for ``--dev`` (``app.py:2049``); its one image-opening route, ``POST /api/import``
(``app.py:1728``), returns the loaded project to the HTTP caller and not to the browser; and
``GET /`` unconditionally serves ``index.html`` (``app.py:865``). There is no route, flag or URL
parameter that puts a file into the editor UI. Verified against v1.21.0, commit ``c555515``.

``src/render.py`` is a different program with a real ``argparse`` (``render.py:271-323``). It
takes the OME-TIFF and the same ``.story.json`` we already write, honours every colour and
contrast in its ``groups``, and produces a viewable exhibit. That is the whole of the "no manual
step" answer: not a way into the editor, a way past it.

**Both front ends need internet to VIEW** - see :data:`NEEDS_INTERNET_NOTE`.

Why we do not import ``squid2minerva``
--------------------------------------
That package (``~/CEPHLA/projects/explorer``) is not installable: it has no
``pyproject.toml``, its imports resolve only via a ``sys.path`` hack in its own ``run.py``,
and its ``requirements.txt`` hard-pins ``tifffile==2025.5.10`` / ``zarr==2.18.7`` against
SquidXplorer's ``tifffile>=2023.1.0``. It has no git tags, so it cannot even be pinned by
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
  ("Image is missing OME-XML pixel size") when it is absent. SquidXplorer's ``pixel_size_um``
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
  regex then cannot parse — has no counterpart here, and SquidXplorer's names
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

#: Env var pointing at an ``explorer`` checkout that has run its ``setup.py``. That checkout
#: holds *both* halves we need: ``vendor/minerva-author/src/app.py`` and the ``.venv`` whose
#: interpreter has minerva-author's dependencies (waitress, flask_cors, xsdata, ome-types,
#: openslide-bin, ...). minerva-author has no venv of its own.
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
    luts: Optional[dict] = None,
) -> list[dict]:
    """One Minerva group over all channels: colour + contrast.

    Contrast defaults to a 1st-percentile floor and 99.9th-percentile ceiling per channel,
    normalised to 0..1 against the dtype maximum - Minerva's own convention. This is the *only*
    place our channel colours reach Minerva.

    *luts*, when given, is ``{channel_name: {"clim": (lo, hi) | None, "rgb": (r,g,b) | None}}``
    read off the napari layers the user is looking at (``RegionViewer._per_channel_luts``). It
    is applied **per channel and per field**, never all-or-nothing:

    * a channel NOT in *luts* keeps the percentile contrast and *channel_colors* entry, exactly
      as if *luts* had not been passed. A window that has three of the acquisition's four
      channels on screen must not blank the fourth;
    * ``"clim"`` of ``None`` keeps the percentiles for that channel;
    * ``"rgb"`` of ``None`` keeps *channel_colors* for that channel. ``None`` is what
      :func:`squidxplorer._napari_view.colormap_hue_rgb` returns for a colormap that is a ramp
      rather than one colour (``viridis``, ``turbo``), which Minerva's single ``"color"`` field
      cannot hold. Falling back is deliberate: emitting the ramp's brightest stop would put a
      colour into Minerva that is on no screen.

    ``clim`` is in RAW INTENSITY UNITS, the same units napari's contrast slider works in and the
    same units the percentiles are computed in, so both paths divide by ``dtype_max`` here and
    the arithmetic below is shared rather than duplicated.
    """
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
            # What was DONE to these pixels, travelling with them. An exported OME-TIFF outlives
            # this session and the log line that described it, and flat-field correction is on by
            # default -- so "these intensities were divided by a gain field" has to be legible
            # from the export itself, not reconstructed from whoever remembers the run.
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


def _provenance_text(image, projector: str, operator: str, *, region: str = "",
                     t: Optional[int] = None, fovs: Optional[Sequence[int]] = None,
                     n_fovs: Optional[int] = None) -> str:
    """One line saying what produced these pixels, for the story's ``sample_info.text``.

    Reads the :class:`~squidxplorer._placement.Placement` riding on the fused array rather than
    re-deriving anything, so it cannot drift from what actually ran. Degrades to just the
    operator names when the array carries no placement (a plain ndarray from a custom region
    operator), because a partial record is still better than none and a raise here would fail an
    export over a caption.

    WHICH REGION, WHICH TIMEPOINT AND WHICH FIELDS are named here as of 2026-08-05, because both
    of them became things a user can change and neither was recorded. The exported timepoint used
    to be hardcoded to 0 and a FOV subset was not expressible at all, so "the mosaic" was
    unambiguous; now the same acquisition yields a different file per timepoint and per box, and
    an OME-TIFF outlives the log line that described it. The filename carries ``_t1_`` and
    ``_2fov`` — this is the same two facts spelled out in the story, where Minerva shows them.

    ``p.reg_t`` stays and is NOT the same number: it is the timepoint the geometry was SOLVED on,
    which is deliberately fixed while the exported one moves. Naming only one of the two would
    make the story read as though the pixels came from the registration timepoint.
    """
    parts = [f"squidxplorer {operator}/{projector}"]
    if region:
        parts.append(f"region {region}")
    if t is not None:
        parts.append(f"timepoint t={int(t)}")
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
    luts: Optional[dict] = None,
    **operator_kwargs,
) -> list[tuple[Path, Path]]:
    """Export the selected region(s) to Minerva-ingestable file pairs — ONE PAIR PER REGION.

    *selection* is ``[(region, fov), ...]`` (what the plate emits). It is grouped by region,
    and each region is fused into a single mosaic through :func:`squidxplorer.stitch_plate` — the
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
    luts:
        Optional ``{channel_name: {"clim": (lo, hi) | None, "rgb": (r,g,b) | None}}`` - THE LUTS
        THE USER HAS ON SCREEN, from ``RegionViewer._per_channel_luts``. When given they beat the
        defaults per channel and per field; see :func:`auto_groups` for what each field replaces
        and what happens when a channel is missing from the dict.

        Optional, and the default has to stay the percentiles rather than become the screen. A
        plate-level export has no window and therefore no on-screen LUTs, and the CLI has none
        either; making the screen mandatory would leave those two paths with nothing to read.
        Every existing caller passing nothing keeps today's behaviour byte for byte.

        Only the STORY is affected. The OME-TIFF keeps the acquisition's ``display_color`` in its
        OME-XML, because that file is the pixels plus what the microscope recorded, and a screen
        setting is not a fact about the acquisition. Minerva reads colour from the story anyway
        (see the module docstring), so this is the field that decides what Minerva shows.
    **operator_kwargs:
        Forwarded to the region operator (``blend_px=``, ``channels=``, ``register=``, ...).

    Raises
    ------
    ValueError
        If the selection is empty, the acquisition has no pixel size, a ``(region, fov)`` is
        not in the acquisition, a channel has no name, or *t* is out of range. All are raised
        *before* anything is written.
    """
    from squidxplorer._stitch import stitch_plate   # local: avoids an import cycle at module load

    grouped = group_selection(selection)
    if not grouped:
        raise ValueError("nothing selected: export_selection needs at least one (region, fov)")

    meta = reader.metadata
    pixel_um = _require_pixel_size(meta)                  # refuse early — nothing written yet
    _resolve_operator(projector)     # unknown projector: fail here, named, not mid-stitch

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
            auto_groups(img_cyx, names, colors, label=label, luts=luts),
            pixels_per_micron=ppm,
            provenance=_provenance_text(
                image, projector, operator, region=region, t=t, fovs=fovs,
                n_fovs=len(fovs_per_region.get(region, []) or []) or None),
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

    Read from ``$SQUIDXPLORER_MINERVA_HOME``, else the conventional sibling checkout. Returns a
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

    ONE TAB, NOT TWO
    ----------------
    **minerva-author opens the browser itself.** ``src/app.py`` (v1.21.0, commit ``c555515``)
    defines ``open_browser()`` at ``:2033`` and calls it at ``:2050`` and ``:2053`` - once in
    each arm of ``if "--dev" in sys.argv``, so it is **unconditional and there is no flag that
    suppresses it**. That is the whole of the server's argv handling: ``sys.argv`` is read
    exactly once, at ``:2049``, and only to test for ``--dev``. Asking Author not to open a tab
    is therefore not available, and the only end of this we control is ours.

    So the rule is: **whoever started the server opens the tab.**

    * We started it -> Author opens its own tab. We do not, or the user gets two.
    * It was already answering -> nobody is about to open anything, so we do.

    Why dropping our call on a cold start is safe, stated as the failure rather than the happy
    path: if Author's ``webbrowser.open_new`` cannot find a browser it returns ``False`` and the
    server still serves, so the user gets no tab and a ``True`` from here. That is why the
    caller's success line names the URL - a user with no tab has an address to paste rather than
    a dead end. If instead that call *raises*, it raises before ``serve()`` on the next line, so
    the server never binds the port, ``is_running()`` never becomes true, this returns ``False``,
    and the caller already reports a launch failure. Neither case is silent.

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
        we_started_it = True     # ...and it is about to open its own tab. See "ONE TAB, NOT TWO".
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
    if open_browser and not we_started_it:
        try:
            webbrowser.open(MINERVA_URL)
        except Exception:
            pass          # the server is up; a browser that won't open is not a failure
    return True


#: Said in the GUI, in the module docstring and here, because it is recorded nowhere else and a
#: user meets it as a blank page rather than as an error. BOTH Minerva front ends load their
#: JavaScript from jsdelivr, so both need a working internet connection at VIEWING time:
#:
#: * Minerva Author's own UI - ``static/index.html`` ends with
#:   ``<script src="https://cdn.jsdelivr.net/npm/minerva-author-ui@1.10.2/build/bundle...js">``;
#: * the exhibit :func:`render_exhibit` writes - ``src/exhibit.py:30`` emits
#:   ``<script src="https://cdn.jsdelivr.net/npm/minerva-browser@3.20.0/build/bundle.js">``.
#:
#: Verified against the sibling checkout at v1.21.0, commit ``c555515``. The rendered tiles ARE
#: local and complete; it is only the viewer program that is fetched. There is a stale 4.2 MB
#: ``static/bundle.*.js`` in that checkout that nothing references, so "it worked offline once"
#: is not a thing this can fall back on.
NEEDS_INTERNET_NOTE = (
    "Minerva's viewer JavaScript is loaded from a CDN (jsdelivr), so viewing needs internet - "
    "the rendered tiles themselves are local."
)


def render_script(home: Path) -> Path:
    """``src/render.py`` inside a checkout. Beside :func:`_minerva_parts`' ``src/app.py``."""
    return Path(home) / "vendor" / "minerva-author" / "src" / "render.py"


def render_exhibit(ome_path, story_path, out_dir, *, threads: Optional[int] = None,
                   force: bool = True, should_stop=None, timeout: float = 3600.0) -> Path:
    """Render *ome_path* + *story_path* into a viewable Minerva exhibit. Returns its ``index.html``.

    **This is the one path from our export to a Minerva view with no file picking.** Minerva
    Author's editor cannot be pointed at a file: its server reads ``sys.argv`` once and only for
    ``--dev`` (``app.py:2049``), its only image-opening route is ``POST /api/import`` which
    returns the project to the HTTP caller and not to the browser (``app.py:1728``), and ``GET /``
    unconditionally serves ``index.html`` (``app.py:865``). ``src/render.py`` is a different
    program with a real ``argparse`` (``render.py:271-323``) that takes the OME-TIFF, the same
    ``.story.json`` we already write, and an output directory. It reads our ``groups``, so the
    colours and contrast in the story are the colours and contrast in the exhibit.

    It is a VIEWER, not an editor. Waypoints, story text and mask authoring are Author's job, so
    this sits beside :func:`launch_minerva` rather than replacing it.

    Three costs, measured or cited rather than asserted:

    * **Lossy.** The output is a JPEG pyramid. The OME-TIFF is untouched and remains the archival
      copy; the exhibit is a rendering of it.
    * **Slow, and here are the numbers.** Measured on this machine against real exported Squid
      mosaics, both written by :func:`export_selection` itself:

      ==========================================  =========  ==========  ==========
      input                                       threads    wall        output
      ==========================================  =========  ==========  ==========
      4ch 2048x2048 uint16 (33.5 MB OME-TIFF)     4          2.1 s       580 KB
      4ch 11535x9635 uint16 (889 MB OME-TIFF)     8          132 s       29 MB
      ==========================================  =========  ==========  ==========

      The FIRST render of a session costs about 13 s more than the table says. That is
      ``render.py``'s own imports (skimage, tifffile, ome-types) loading cold in Minerva's venv,
      not work on the image: the same 2048x2048 input measured 15.1 s on the first run of the
      session and 2.1 s on a later one into a fresh output directory. It is a fixed cost, so it
      dominates a small crop and is noise on a real region.

      So: **seconds for a crop, minutes for a whole region**, which is why this takes
      *should_stop* and why the GUI runs it off the main thread.
    * **Needs internet to VIEW.** See :data:`NEEDS_INTERNET_NOTE`.

    Raises
    ------
    FileNotFoundError
        No checkout, no venv interpreter, or no ``render.py`` in it. Named, so the status line can
        say which.
    RuntimeError
        ``render.py`` exited non-zero (its stderr tail is the message), or the wait was abandoned,
        or it finished without writing an ``index.html``. It runs as a script under a FOREIGN
        venv, so its failure arrives as an exit code and stderr, never as a Python exception we
        could let through - turning that into one here is what lets the caller report it by name.
    """
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

    # Poll rather than communicate(): a render is minutes long and the GUI thread that closes the
    # window must not be held for it. Same contract as launch_minerva's liveness wait.
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
    """Open a rendered exhibit's ``index.html`` in the browser. Best-effort, never raises.

    A ``file://`` URL, not a server: the exhibit is static tiles plus one HTML file. The viewer
    JavaScript still comes off the CDN (:data:`NEEDS_INTERNET_NOTE`), which is the one thing about
    it that is not local.
    """
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
