"""IMA-254: one TINY synthetic acquisition per Squid output writer.

THE POINT. Two of Squid's writers were unserved by ``squidxplorer.reader`` — one of them silently —
and the reason was not that they are hard. It is that this repo only ever had two acquisitions to
test against, and both of them came from the same writer. Coverage followed whatever happened to
be in ``~/Downloads``. This module makes coverage follow the SPEC instead: every writer in
``control/core/job_processing.py`` gets a fixture here, so adding a writer to Squid without adding
one here is what fails, rather than a customer's acquisition.

The writers, read out of ``control/core/job_processing.py`` and ``control/utils.py`` (verified,
not taken on trust — the ``fov`` field really is zero-padded to ``_def.FILE_ID_PADDING``, the 5-D
zarr axis order really is ``TCZYX`` and not ``TZCYX``, and ``acquisition.zarr`` really is a zarr
ARRAY rather than a group):

    ``SaveImageJob`` default          ``{t}/{region}_{fov}_{z}_{channel}.tiff``
    ``SaveImageJob`` MULTI_PAGE_TIFF  ``{t}/{region}_{fov:0PAD}_stack.tiff``
    ``SaveOMETiffJob``                ``ome_tiff/{region}_{fov:0PAD}.ome.tiff``   (T, Z, C, Y, X)
    ``SaveZarrJob`` HCS               ``plate.ome.zarr/{row}/{col}/{fov}/0``      (T, C, Z, Y, X)
    ``SaveZarrJob`` non-HCS per-FOV   ``zarr/{region}/fov_{n}.ome.zarr/0``        (T, C, Z, Y, X)
    ``SaveZarrJob`` non-HCS 6D        ``zarr/{region}/acquisition.zarr``   (FOV, T, C, Z, Y, X)

DISK. Every fixture is the same tiny acquisition — 2 regions x 2 FOVs x 2 z x 2 channels of 4x4
uint16 — so the whole set is a few tens of kilobytes and is built inside ``tmp_path``. The
identical pixel payload across all six means a test can assert the SAME array through every
reader, which is what makes "exact pixels, verified against a direct read" a one-liner per writer.

FIDELITY. Where Squid's write is subtle, the subtlety is reproduced rather than approximated. The
multi-page builder makes the exact ``TiffWriter.write(metadata=..., description=..., extratags=
[(285, 's', 0, channel, False)])`` call Squid makes, so whatever tifffile does with two competing
description arguments, the fixture has it too. The zarr builders emit Squid's own ``attributes.
ome`` payload, including the ``_squid`` block and the ``datasets[0].path`` of ``"."`` that the 6D
layout uses because its metadata lives in the array's own ``zarr.json``.

PADDING. The TIFF builders default to ``FILE_ID_PADDING=4`` even though the reference Squid config
ships ``0``. A fixture written at width 0 cannot tell a reader that parses the width from one that
hardcodes it, and the width is a per-deployment setting. The padded fixture is the one that fails
a reader which assumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from tests.conftest import (
    _ACQ_YAML,
    _FOV_MM,
    _PARAMS,
    _YAML,
    _pixel_value,
    CHANNELS,
    FOVS,
    NZ,
    REGIONS,
)

# The one shape every writer fixture shares. 8x8 uint16 keeps the whole six-writer set in the tens
# of kilobytes; it is not smaller because tifffile's OME writer re-infers axes on very small
# planes and rejects a 5-D 4x4 stack as "axes do not match stored shape". 8x8 is the smallest
# frame every writer here accepts.
FRAME = (8, 8)
N_T = 1
FILE_ID_PADDING = 4          # deliberately != Squid's default 0; see the module docstring


def plane(region: str, fov: int, z: int, channel: str) -> np.ndarray:
    """The canonical pixels for one plane — identical across every writer fixture.

    Deterministic and unique per (region, fov, z, channel), so an exact-array comparison after a
    round trip through any writer proves the reader resolved the right plane, not merely a plane.
    """
    base = _pixel_value(REGIONS.index(region), fov, z, CHANNELS.index(channel))
    return (np.arange(FRAME[0] * FRAME[1], dtype=np.uint16).reshape(FRAME) + base).astype(np.uint16)


def expected_arrays() -> dict:
    """``{(region, fov, z, channel): array}`` for the whole canonical acquisition."""
    return {
        (r, f, z, c): plane(r, f, z, c)
        for r in REGIONS for f in FOVS for z in range(NZ) for c in CHANNELS
    }


def _sidecars(root: Path, coordinates: bool = True) -> None:
    """The sidecar files Squid drops next to any acquisition, whichever writer produced it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    if coordinates:
        lines = ["region,fov,z_level,x (mm),y (mm),z (um),time"]
        for region in REGIONS:
            for fov in FOVS:
                for z in range(NZ):
                    x, y = _FOV_MM[fov]
                    lines.append(f"{region},{fov},{z},{x},{y},0.0,2025-10-28 13:40:43")
        (root / "coordinates.csv").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------
# SaveImageJob — MULTI_PAGE_TIFF branch
# --------------------------------------------------------------------------------------------

def build_multi_page_tiff(root, padding: int = FILE_ID_PADDING, jitter_mm: float = 1e-4) -> Path:
    """``{t}/{region}_{fov:0{padding}}_stack.tiff`` — one appended page per (z, channel).

    Reproduces ``SaveImageJob.save_image``'s MULTI_PAGE_TIFF branch call for call: the same
    metadata dict, the same ``description=json.dumps(metadata)``, the same
    ``extratags=[(285, 's', 0, channel, False)]``, and the same append-one-page-at-a-time
    ``TiffWriter(path, append=True)`` loop.

    *jitter_mm* perturbs ``x_mm``/``y_mm`` by z-level. Squid re-reads ``stage.get_pos()`` for every
    single capture, so a real stack's pages disagree about position at the micrometre level; a
    reader that treats that as a corrupt file would reject every real z-stack. NO ``coordinates.csv``
    is written, because this writer records positions inline and the fixture must prove the reader
    uses them.
    """
    root = Path(root)
    _sidecars(root, coordinates=False)
    folder = root / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        for fov in FOVS:
            path = folder / f"{region}_{fov:0{padding}}_stack.tiff"
            x_mm, y_mm = _FOV_MM[fov]
            for z in range(NZ):
                for c_i, channel in enumerate(CHANNELS):
                    metadata = {
                        "z_level": z,
                        "channel": channel,
                        "channel_index": c_i,
                        "region_id": region,
                        "fov": fov,
                        "x_mm": x_mm + z * jitter_mm,
                        "y_mm": y_mm + z * jitter_mm,
                        "z_mm": 0.0015 * z,
                        "time": "2025-10-28 13:40:43.939945",
                        "z_piezo (um)": 1.5 * z,
                    }
                    with tifffile.TiffWriter(path, append=True) as writer:
                        writer.write(
                            plane(region, fov, z, channel),
                            metadata=metadata,
                            description=json.dumps(metadata),
                            extratags=[(285, "s", 0, str(channel), False)],
                        )
    return root


# --------------------------------------------------------------------------------------------
# SaveOMETiffJob
# --------------------------------------------------------------------------------------------

def build_ome_tiff(root, padding: int = FILE_ID_PADDING) -> Path:
    """``ome_tiff/{region}_{fov:0{padding}}.ome.tiff`` — one 5-D ``TZCYX`` stack per field.

    ``utils_ome_tiff_writer.ome_base_name`` is ``f"{region_id}_{fov:0{FILE_ID_PADDING}}"`` and
    ``ome_output_folder`` is ``{experiment_path}/ome_tiff``. The axis order is ``T, Z, C, Y, X``
    — note this is NOT the zarr writer's ``T, C, Z, Y, X``; the two Squid writers genuinely
    disagree, and a fixture that used one order for both would hide a real transposition bug.

    The write is Squid's two-step, not a one-shot ``imwrite(data)``: allocate the full 5-D file
    from ``shape=``/``dtype=`` with ``ome=True``, then fill it plane by plane through
    ``tifffile.memmap``. That is exactly what ``SaveOMETiffJob._save_ome_tiff`` does (each job
    holds a file lock and writes ONE plane), and it matters — handing tifffile a populated array
    with a size-1 T axis makes it re-infer the axes and reject the shape outright, so a one-shot
    fixture would have to fake a T it does not have.
    """
    root = Path(root)
    _sidecars(root, coordinates=True)
    out = root / "ome_tiff"
    out.mkdir(parents=True, exist_ok=True)
    shape = (N_T, NZ, len(CHANNELS)) + FRAME
    for region in REGIONS:
        for fov in FOVS:
            path = out / f"{region}_{fov:0{padding}}.ome.tiff"
            tifffile.imwrite(
                path, shape=shape, dtype=np.uint16, ome=True,
                metadata={"axes": "TZCYX", "Channel": {"Name": list(CHANNELS)}},
            )
            stack = tifffile.memmap(path, dtype=np.uint16, mode="r+")
            stack.shape = shape
            try:
                for z in range(NZ):
                    for c_i, channel in enumerate(CHANNELS):
                        stack[0, z, c_i, :, :] = plane(region, fov, z, channel)
                stack.flush()
            finally:
                del stack
    return root


# --------------------------------------------------------------------------------------------
# SaveZarrJob — the three layouts
# --------------------------------------------------------------------------------------------

def _write_zarr_array(path: Path, data: np.ndarray) -> None:
    """One zarr **v3** array at *path*, matching ``ZarrWriter.initialize``'s driver and layout."""
    import tensorstore as ts

    path.parent.mkdir(parents=True, exist_ok=True)
    store = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(data.shape),
                "chunk_grid": {"name": "regular",
                               "configuration": {"chunk_shape": list(data.shape)}},
                "chunk_key_encoding": {"name": "default"},
                "data_type": "uint16",
                "fill_value": 0,
            },
        },
        create=True,
        delete_existing=True,
    ).result()
    store[...].write(np.ascontiguousarray(data)).result()


def _omero_channels() -> list:
    return [{"label": name, "active": True,
             "color": ("00FF00" if i == 0 else "FF0000"),
             "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}}
            for i, name in enumerate(CHANNELS)]


def _ome_attrs(name: str, axes_6d: bool, dataset_path: str,
               pixel_size_um: float = 0.325, z_step_um: float = 1.5) -> dict:
    """Squid's ``attributes`` payload, byte-for-byte in structure with ``_write_zarr_metadata``."""
    space = [{"name": "z", "type": "space", "unit": "micrometer"},
             {"name": "y", "type": "space", "unit": "micrometer"},
             {"name": "x", "type": "space", "unit": "micrometer"}]
    tc = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}]
    if axes_6d:
        axes = [{"name": "fov", "type": "fov"}] + tc + space
        scale = [1.0, 1.0, 1.0, z_step_um, pixel_size_um, pixel_size_um]
    else:
        axes = tc + space
        scale = [1.0, 1.0, z_step_um, pixel_size_um, pixel_size_um]
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [{
                "version": "0.5",
                "name": name,
                "axes": axes,
                "datasets": [{"path": dataset_path,
                              "coordinateTransformations": [{"type": "scale", "scale": scale}]}],
                "coordinateTransformations": [{"type": "identity"}],
            }],
            "omero": {"name": name, "version": "0.5", "channels": _omero_channels()},
        },
        "_squid": {
            "structure": "6D-FTCZYX" if axes_6d else "5D-TCZYX",
            "pixel_size_um": pixel_size_um,
            "z_step_um": z_step_um,
            "acquisition_complete": True,
        },
    }


def _write_group(path: Path, attrs: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}, indent=2)
    )


def _tczyx(region: str, fov: int) -> np.ndarray:
    """One field as Squid's 5-D zarr array: ``(T, C, Z, Y, X)`` — channel BEFORE z."""
    arr = np.zeros((N_T, len(CHANNELS), NZ) + FRAME, dtype=np.uint16)
    for c_i, channel in enumerate(CHANNELS):
        for z in range(NZ):
            arr[0, c_i, z] = plane(region, fov, z, channel)
    return arr


def build_zarr_hcs(root) -> Path:
    """``plate.ome.zarr/{row}/{col}/{fov}/0`` plus Squid's plate and well group metadata.

    Mirrors ``write_plate_metadata`` / ``write_well_metadata``: the plate group lists
    ``rows``/``columns``/``wells`` with ``path`` = ``"{row}/{col}"``, each well group lists its
    ``images``, and the FIELD group carries the multiscales with the array at ``0``. Squid writes
    no metadata at the intermediate row level, and neither does this.
    """
    root = Path(root)
    _sidecars(root, coordinates=True)
    plate = root / "plate.ome.zarr"
    parsed = [(r[0], r[1:]) for r in REGIONS]
    rows = sorted({row for row, _ in parsed})
    cols = sorted({col for _, col in parsed}, key=int)
    _write_group(plate, {"ome": {"version": "0.5", "plate": {
        "version": "0.5", "name": "plate",
        "rows": [{"name": r} for r in rows],
        "columns": [{"name": str(c)} for c in cols],
        "wells": [{"path": f"{row}/{col}", "rowIndex": rows.index(row),
                   "columnIndex": cols.index(col)} for row, col in parsed],
    }}})
    for region, (row, col) in zip(REGIONS, parsed):
        _write_group(plate / row / col, {"ome": {"version": "0.5", "well": {
            "version": "0.5", "images": [{"path": str(f)} for f in FOVS]}}})
        for fov in FOVS:
            field = plate / row / col / str(fov)
            _write_group(field, _ome_attrs(str(fov), axes_6d=False, dataset_path="0"))
            _write_zarr_array(field / "0", _tczyx(region, fov))
    return root


def build_zarr_per_fov(root) -> Path:
    """``zarr/{region}/fov_{n}.ome.zarr/0`` — ``utils.build_per_fov_zarr_path``, 5-D per FOV.

    ``ZarrWriter._is_ome_ngff_array_path`` is True for this output path (it ends in ``/0``), so
    the OME metadata lands on the PARENT group's ``zarr.json`` with ``datasets[0].path == "0"``,
    exactly as in HCS mode. The difference from HCS is purely the directory nesting: there is no
    plate node and no well node, so the region id comes from the folder name and the FOV id from
    the ``fov_{n}`` filename.
    """
    root = Path(root)
    _sidecars(root, coordinates=True)
    for region in REGIONS:
        for fov in FOVS:
            group = root / "zarr" / region / f"fov_{fov}.ome.zarr"
            _write_group(group, _ome_attrs(group.name, axes_6d=False, dataset_path="0"))
            _write_zarr_array(group / "0", _tczyx(region, fov))
    return root


def build_zarr_6d(root) -> Path:
    """``zarr/{region}/acquisition.zarr`` — ``utils.build_6d_zarr_path``, ONE 6-D array per region.

    This is the layout Squid itself calls non-standard, and it is shaped differently from every
    other one: ``_is_ome_ngff_array_path`` is False (the path does not end in ``/0``), so the OME
    metadata is merged into the ARRAY's own ``zarr.json`` — ``node_type`` is ``"array"``, not
    ``"group"`` — with ``datasets[0].path == "."`` pointing at itself. A reader that identifies
    NGFF nodes by group-ness alone cannot see this store at all, which is why it is recognised by
    name.
    """
    root = Path(root)
    _sidecars(root, coordinates=True)
    for region in REGIONS:
        path = root / "zarr" / region / "acquisition.zarr"
        arr = np.zeros((len(FOVS), N_T, len(CHANNELS), NZ) + FRAME, dtype=np.uint16)
        for f_i, fov in enumerate(FOVS):
            arr[f_i] = _tczyx(region, fov)
        _write_zarr_array(path, arr)
        zarr_json = json.loads((path / "zarr.json").read_text())
        zarr_json["attributes"] = _ome_attrs("acquisition.zarr", axes_6d=True, dataset_path=".")
        (path / "zarr.json").write_text(json.dumps(zarr_json, indent=2))
    return root


# --------------------------------------------------------------------------------------------
# The registry the coverage tests and tools/acceptance.py both walk
# --------------------------------------------------------------------------------------------

def build_individual_tiff(root) -> Path:
    """``{t}/{region}_{fov}_{z}_{channel}.tiff`` — ``SaveImageJob``'s default branch."""
    root = Path(root)
    _sidecars(root, coordinates=True)
    folder = root / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        for fov in FOVS:
            for z in range(NZ):
                for channel in CHANNELS:
                    tifffile.imwrite(folder / f"{region}_{fov}_{z}_{channel}.tiff",
                                     plane(region, fov, z, channel))
    return root


# (label, builder, reader class name, does this writer record per-FOV positions itself?)
WRITERS = [
    ("SaveImageJob default (individual TIFF)", build_individual_tiff, "SquidReader", False),
    ("SaveImageJob MULTI_PAGE_TIFF", build_multi_page_tiff, "SquidMultiPageTiffReader", True),
    ("SaveOMETiffJob", build_ome_tiff, "SquidOMEReader", False),
    ("SaveZarrJob HCS plate.ome.zarr", build_zarr_hcs, "SquidZarrReader", False),
    ("SaveZarrJob non-HCS fov_{n}.ome.zarr", build_zarr_per_fov, "SquidZarrReader", False),
    ("SaveZarrJob non-HCS 6D acquisition.zarr", build_zarr_6d, "SquidZarrReader", False),
]
