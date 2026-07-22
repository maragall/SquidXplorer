"""Fused mosaics for pane 2 — the unit displayed is a MOSAIC, never a single FOV.

Qt-free: these exercise the loader and the geometry, which is where the wrongness would be.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squidmip._mosaic_source import (
    fuse_region_mosaic,
    level_paths,
    mosaic_bbox_um,
    open_pyramid,
)


class _Reader:
    """Reads a distinct constant per FOV so placement is checkable by value."""

    def __init__(self, frame=(4, 6), values=None, fail=()):
        self.frame = frame
        self.values = values or {}
        self.fail = set(fail)

    def read(self, region, fov, channel, z, t=0):
        if fov in self.fail:
            raise OSError("simulated unreadable FOV")
        return np.full(self.frame, self.values.get(fov, fov + 1), dtype=np.uint16)


def _meta(positions, fovs, frame=(4, 6), px=2.0):
    return {
        "regions": ["A1"],
        "fovs_per_region": {"A1": fovs},
        "fov_positions_um": positions,
        "pixel_size_um": px,
        "frame_shape": frame,
        "dtype": "uint16",
        "channels": [{"name": "488"}],
    }


# ------------------------------------------------------------------ fusing


def test_two_fovs_are_placed_side_by_side_by_stage_position():
    # 6 px wide frame, 2 µm/px -> 12 µm; put FOV 1 exactly one frame to the right.
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    out = fuse_region_mosaic(_Reader(), meta, "A1", "488")
    assert out is not None
    mosaic, step = out
    assert step == 1.0
    assert mosaic.shape == (4, 12)
    assert np.all(mosaic[:, :6] == 1)     # FOV 0
    assert np.all(mosaic[:, 6:] == 2)     # FOV 1


def test_overlap_is_covered_not_left_as_a_hole():
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (6.0, 0.0)}, [0, 1])
    mosaic, _ = fuse_region_mosaic(_Reader(), meta, "A1", "488")
    assert mosaic.shape == (4, 9)
    assert not (mosaic == 0).any(), "a placed mosaic must have no unwritten pixels"


def test_a_mosaic_is_not_derivable_without_positions_and_says_so_by_returning_none():
    """The same 'not derivable, do not guess' signal _mosaic_boxes returns {} for. A guessed
    layout is a WRONG picture, not a rough one."""
    meta = _meta({}, [0, 1])
    assert fuse_region_mosaic(_Reader(), meta, "A1", "488") is None

    meta_no_px = _meta({("A1", 0): (0.0, 0.0)}, [0])
    meta_no_px["pixel_size_um"] = 0
    assert fuse_region_mosaic(_Reader(), meta_no_px, "A1", "488") is None


def test_an_unreadable_fov_leaves_a_hole_rather_than_shifting_its_neighbours():
    """A silent skip would slide every later FOV over by one frame and produce a mosaic that
    looks fine and is wrong."""
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    mosaic, _ = fuse_region_mosaic(_Reader(fail=[0]), meta, "A1", "488")
    assert mosaic.shape == (4, 12)
    assert np.all(mosaic[:, :6] == 0)     # the hole stays where the unreadable FOV was
    assert np.all(mosaic[:, 6:] == 2)     # FOV 1 did NOT move left


def test_a_large_region_is_decimated_rather_than_truncated():
    """Bounding RAM must not change which region the mosaic covers."""
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (2000.0, 0.0)}, [0, 1],
                 frame=(1000, 1000), px=1.0)
    mosaic, step = fuse_region_mosaic(_Reader(frame=(1000, 1000)), meta, "A1", "488",
                                      max_px=1000)
    assert step > 1
    assert max(mosaic.shape) <= 1000
    # aspect preserved: the full extent is 1000 x 3000, so the mosaic stays 1:3
    assert mosaic.shape[1] == pytest.approx(mosaic.shape[0] * 3, rel=0.02)


# ------------------------------------------------------------------ world placement


def test_bbox_um_is_the_regions_stage_extent():
    meta = _meta({("A1", 0): (100.0, 50.0), ("A1", 1): (112.0, 50.0)}, [0, 1])
    x0, y0, x1, y1 = mosaic_bbox_um(meta, "A1")
    assert (x0, y0) == (100.0, 50.0)
    # extent is 4 px tall, 12 px wide at 2 µm/px -> 8 x 24 µm
    assert (x1 - x0, y1 - y0) == (24.0, 8.0)


def test_bbox_um_is_none_when_placement_is_not_derivable():
    assert mosaic_bbox_um(_meta({}, [0]), "A1") is None


# ------------------------------------------------------------------ pyramid loading


def _write_pyramid(root: Path, shapes):
    """Minimal OME-NGFF v0.5 image group with several levels."""
    import zarr

    root.mkdir(parents=True, exist_ok=True)
    for i, shape in enumerate(shapes):
        z = zarr.create_array(store=str(root / str(i)), shape=shape, dtype="uint16",
                              chunks=tuple(min(8, s) for s in shape), overwrite=True)
        z[:] = i + 1
    doc = {
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"multiscales": [{
            "axes": [{"name": n} for n in "tczyx"],
            "datasets": [{"path": str(i)} for i in range(len(shapes))],
        }]}},
    }
    (root / "zarr.json").write_text(json.dumps(doc))


def test_level_paths_follow_the_datasets_list_not_directory_sort(tmp_path):
    """Directory names sort '10' before '2'; the datasets list is the authority on level order."""
    root = tmp_path / "img"
    _write_pyramid(root, [(1, 1, 1, 32, 32), (1, 1, 1, 16, 16), (1, 1, 1, 8, 8)])
    assert [p.name for p in level_paths(root)] == ["0", "1", "2"]


def test_open_pyramid_returns_lazy_decreasing_levels(tmp_path):
    root = tmp_path / "img"
    _write_pyramid(root, [(1, 1, 1, 32, 32), (1, 1, 1, 16, 16), (1, 1, 1, 8, 8)])
    pyr = open_pyramid(root)

    assert [tuple(d.shape) for d in pyr] == [(32, 32), (16, 16), (8, 8)]
    # lazy: dask, not materialised
    assert all(hasattr(d, "compute") for d in pyr)
    assert int(np.asarray(pyr[0][0, 0])) == 1


def test_open_pyramid_drops_a_level_that_does_not_shrink(tmp_path):
    """napari needs strictly decreasing levels; a duplicate would make it pick nonsense."""
    root = tmp_path / "img"
    _write_pyramid(root, [(1, 1, 1, 32, 32), (1, 1, 1, 32, 32), (1, 1, 1, 8, 8)])
    pyr = open_pyramid(root)
    assert [tuple(d.shape) for d in pyr] == [(32, 32), (8, 8)]


def test_a_group_without_multiscales_is_a_loud_error(tmp_path):
    root = tmp_path / "img"
    root.mkdir()
    (root / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {}}))
    with pytest.raises(ValueError, match="multiscales"):
        level_paths(root)


# ------------------------------------------------------------ lazy z stacks


class _CountingReader(_Reader):
    """Counts reads so laziness is provable rather than asserted."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads = 0

    def read(self, region, fov, channel, z, t=0):
        self.reads += 1
        return np.full(self.frame, z + 1, dtype=np.uint16)


def test_a_zstack_becomes_a_lazy_z_y_x_array():
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    meta["n_z"] = 5
    data, step, nz = fuse_region_stack(_Reader(), meta, "A1", "488")

    assert nz == 5
    assert data.shape == (5, 4, 12)
    assert hasattr(data, "compute"), "the stack must stay lazy"


def test_only_the_requested_plane_is_ever_materialised():
    """The architecture taken from record-zstack-viewer: requests cost only what is visible.
    Eagerly fusing every z would turn a 10-plane 28-FOV region into ~10x the reads on open."""
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    meta["n_z"] = 8
    reader = _CountingReader()
    data, _step, _nz = fuse_region_stack(reader, meta, "A1", "488")

    after_probe = reader.reads          # one plane is probed for shape/dtype
    plane = np.asarray(data[3])         # ask for exactly one z

    assert plane.shape == (4, 12)
    assert np.all(plane == 4)           # z=3 -> value 4, so the right plane was fused
    assert reader.reads - after_probe == 2, "materialising one z must read one z, once per FOV"


def test_a_single_plane_acquisition_gets_no_singleton_z_axis():
    """A one-position slider is clutter; record-zstack-viewer relies on the viewer hiding
    singleton sliders and napari does the same when the axis simply is not there."""
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (12.0, 0.0)}, [0, 1])
    meta["n_z"] = 1
    data, _step, nz = fuse_region_stack(_Reader(), meta, "A1", "488")

    assert nz == 1
    assert data.ndim == 2


def test_an_oversized_plane_is_refused_loudly_rather_than_paging_the_machine():
    from squidmip._mosaic_source import fuse_region_stack

    # A genuinely huge extent, with decimation effectively disabled.
    meta = _meta({("A1", 0): (0.0, 0.0), ("A1", 1): (40000.0, 0.0)}, [0, 1],
                 frame=(40000, 40000), px=1.0)
    meta["n_z"] = 4
    with pytest.raises(MemoryError, match="plane budget"):
        fuse_region_stack(_Reader(), meta, "A1", "488", max_px=10_000_000)


def test_a_stack_with_no_positions_is_still_not_derivable():
    from squidmip._mosaic_source import fuse_region_stack

    meta = _meta({}, [0, 1])
    meta["n_z"] = 4
    assert fuse_region_stack(_Reader(), meta, "A1", "488") is None
