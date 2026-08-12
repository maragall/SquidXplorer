"""One tiny synthetic acquisition per Squid output writer.

Every writer in control/core/job_processing.py gets a fixture here, so adding a writer to Squid
without adding one here is what fails, rather than a customer's acquisition:

    SaveImageJob default          {t}/{region}_{fov}_{z}_{channel}.tiff
    SaveImageJob MULTI_PAGE_TIFF  {t}/{region}_{fov:0PAD}_stack.tiff
    SaveOMETiffJob                ome_tiff/{region}_{fov:0PAD}.ome.tiff   (T, Z, C, Y, X)
    SaveZarrJob HCS               plate.ome.zarr/{row}/{col}/{fov}/0      (T, C, Z, Y, X)
    SaveZarrJob non-HCS per-FOV   zarr/{region}/fov_{n}.ome.zarr/0        (T, C, Z, Y, X)
    SaveZarrJob non-HCS 6D        zarr/{region}/acquisition.zarr   (FOV, T, C, Z, Y, X)

Every fixture shares the same tiny pixel payload, so a test can assert the SAME array through
every reader. FILE_ID_PADDING defaults to 4, not Squid's own default of 0, so a fixture that
assumes a padding width fails rather than one that happens to match it.
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

# 8x8 is the smallest frame every writer here accepts: tifffile's OME writer re-infers axes on
# smaller planes and rejects a 5-D 4x4 stack as "axes do not match stored shape".
FRAME = (8, 8)
N_T = 1
FILE_ID_PADDING = 4          # != Squid's default 0, so a fixture assuming padding fails a reader
                             # that hardcodes it.


def plane(region: str, fov: int, z: int, channel: str) -> np.ndarray:
    """Deterministic pixels unique per (region, fov, z, channel), identical across every writer
    fixture."""
    base = _pixel_value(REGIONS.index(region), fov, z, CHANNELS.index(channel))
    return (np.arange(FRAME[0] * FRAME[1], dtype=np.uint16).reshape(FRAME) + base).astype(np.uint16)


def expected_arrays() -> dict:
    """{(region, fov, z, channel): array} for the whole canonical acquisition."""
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


def build_multi_page_tiff(root, padding: int = FILE_ID_PADDING, jitter_mm: float = 1e-4) -> Path:
    """{t}/{region}_{fov:0{padding}}_stack.tiff, one appended page per (z, channel), matching
    SaveImageJob's MULTI_PAGE_TIFF branch. jitter_mm simulates real per-capture stage drift; no
    coordinates.csv is written because this writer records positions inline."""
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


def build_ome_tiff(root, padding: int = FILE_ID_PADDING) -> Path:
    """ome_tiff/{region}_{fov:0{padding}}.ome.tiff, one 5-D TZCYX stack per field. Axis order is
    T,Z,C,Y,X, NOT the zarr writer's T,C,Z,Y,X — the two Squid writers genuinely disagree.
    Written two-step (allocate, then fill plane by plane via tifffile.memmap), matching
    SaveOMETiffJob's own per-plane write."""
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


def _write_zarr_array(path: Path, data: np.ndarray) -> None:
    """One zarr v3 array at *path*, matching ZarrWriter.initialize's driver and layout."""
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
    """Squid's attributes payload, byte-for-byte in structure with _write_zarr_metadata."""
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
    """One field as Squid's 5-D zarr array: (T, C, Z, Y, X), channel before z."""
    arr = np.zeros((N_T, len(CHANNELS), NZ) + FRAME, dtype=np.uint16)
    for c_i, channel in enumerate(CHANNELS):
        for z in range(NZ):
            arr[0, c_i, z] = plane(region, fov, z, channel)
    return arr


def build_zarr_hcs(root) -> Path:
    """plate.ome.zarr/{row}/{col}/{fov}/0, mirroring Squid's write_plate_metadata /
    write_well_metadata; no metadata at the intermediate row level, matching Squid."""
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
    """zarr/{region}/fov_{n}.ome.zarr/0, 5-D per FOV. Same OME metadata layout as HCS mode, just
    without the plate/well group nesting."""
    root = Path(root)
    _sidecars(root, coordinates=True)
    for region in REGIONS:
        for fov in FOVS:
            group = root / "zarr" / region / f"fov_{fov}.ome.zarr"
            _write_group(group, _ome_attrs(group.name, axes_6d=False, dataset_path="0"))
            _write_zarr_array(group / "0", _tczyx(region, fov))
    return root


def build_zarr_6d(root) -> Path:
    """zarr/{region}/acquisition.zarr, one 6-D array per region. Non-standard layout: OME
    metadata is merged into the ARRAY's own zarr.json (node_type "array", not "group"), so it
    can't be recognised by group-ness alone."""
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


def build_individual_tiff(root) -> Path:
    """{t}/{region}_{fov}_{z}_{channel}.tiff, SaveImageJob's default branch."""
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
