"""Minerva export: OME-TIFF + .story.json, and the best-effort launch."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

# The export reaches tilefusion at run time through the region-operator seam; not a dependency.
pytest.importorskip("tilefusion", reason="tilefusion (maragall/stitcher) not installed: the minerva "
                                         "export path is UNTESTED here, not passing")

from squidxplorer import _engine, _minerva
from squidxplorer._minerva import (
    auto_groups,
    default_out_dir,
    export_selection,
    launch_minerva,
    write_ome_tiff,
    write_story,
)
from squidxplorer.reader import open_reader
from tests.conftest import CH_IN_YAML, CH_NOT_IN_YAML, NZ, _pixel_value
from tests.writer_fixtures import WRITERS as _WRITERS


# --- export_selection ------------------------------------------------------------------------

def test_export_writes_one_fused_mosaic_per_region_never_one_per_fov(squid_dataset, tmp_path):
    """Minerva Author lays out exactly one image, so a region is a mosaic, not a FOV."""
    root, _ = squid_dataset
    out = tmp_path / "out"
    sel = [("B2", 0), ("B2", 1), ("B3", 0), ("B3", 1)]      # 4 FOVs, 2 regions
    pairs = export_selection(open_reader(root), sel, out)

    assert len(pairs) == 2, "one pair per REGION — a region is a mosaic, not a FOV"
    assert len(list(out.glob("*.ome.tiff"))) == 2
    for ome, story in pairs:
        assert ome.exists() and story.exists()
        assert ome.name.endswith(".ome.tiff")
        assert story.name.endswith(".story.json")
        assert "fov" not in ome.name, "a per-FOV filename means the per-FOV model came back"
    # order is the caller's region order, not the region loop's completion order
    assert "B2" in pairs[0][0].name and "B3" in pairs[1][0].name


def test_each_exported_file_holds_exactly_one_series(squid_dataset, tmp_path):
    """Minerva reads ``series[0]`` and nothing else."""
    root, _ = squid_dataset
    (ome, _), = export_selection(open_reader(root), [("B2", 0), ("B2", 1)], tmp_path)
    with tifffile.TiffFile(str(ome)) as tf:
        assert len(tf.series) == 1


def test_a_fov_subset_is_one_cropped_mosaic_not_n_files(squid_dataset, tmp_path):
    """Selecting some of a region's FOVs still emits one mosaic, cropped to those FOVs."""
    root, _ = squid_dataset
    (whole, _), = export_selection(open_reader(root), [("B2", 0), ("B2", 1)], tmp_path / "whole")
    (crop, _), = export_selection(open_reader(root), [("B2", 1)], tmp_path / "crop")

    assert len(list((tmp_path / "crop").glob("*.ome.tiff"))) == 1
    full_px, crop_px = tifffile.imread(str(whole)), tifffile.imread(str(crop))
    assert crop_px.shape[0] == full_px.shape[0]                  # same channels
    assert crop_px.shape[2] < full_px.shape[2], "the subset was not cropped to its own FOVs"


def test_exported_pixels_are_the_fused_mosaic_byte_for_byte(squid_dataset, tmp_path):
    """The OME-TIFF must carry the real fused pixels in native uint16 — no rescale, no cast."""
    root, arrays = squid_dataset
    # blend_px=0 and correct_illumination=False: fixture concessions on 4 px tiles, so the
    # fused pixels equal a plain MIP of the fixture planes.
    (ome, _), = export_selection(open_reader(root), [("B3", 1)], tmp_path, blend_px=0,
                                 correct_illumination=False)

    written = tifffile.imread(str(ome))
    assert written.dtype == np.uint16

    reader = open_reader(root)
    names = [c["name"] for c in reader.metadata["channels"]]
    for c_i, ch in enumerate(names):
        expected = np.maximum.reduce([arrays[("B3", 1, z, ch)] for z in range(NZ)])
        np.testing.assert_array_equal(written[c_i], expected)


def test_export_goes_through_the_region_operator_seam(squid_dataset, tmp_path):
    """An operator registered at runtime is selectable with no edit to _minerva.py."""
    import squidxplorer
    from squidxplorer import _stitch

    calls = []

    def spy(reader, region, fovs, **kwargs):
        calls.append((region, tuple(fovs)))
        return _stitch.stitch_region(reader, region, fovs, register=False, **kwargs)

    name = "minerva_test_op"
    _engine._OPERATORS.pop(name, None)
    squidxplorer.add_region_operator(name, spy)
    try:
        pairs = export_selection(
            open_reader(squid_dataset[0]), [("B2", 0), ("B2", 1)], tmp_path,
            operator=name,
        )
    finally:
        _engine._OPERATORS.pop(name, None)

    assert calls == [("B2", (0, 1))], "the whole region reached the operator in one call"
    assert len(pairs) == 1 and name in pairs[0][0].name
    written, out = pairs[0][0], Path(tmp_path)
    assert written.is_file(), f"{written} was reported but not written"
    assert out in written.parents, f"{written} escaped {out}"
    assert written.stat().st_size > 0, f"{written} is empty"


@pytest.mark.parametrize("label, build", [(w[0], w[1]) for w in _WRITERS])
def test_export_is_one_mosaic_per_region_from_every_squid_writer(label, build, tmp_path):
    """The export must not care which Squid writer produced the acquisition."""
    root = build(tmp_path / "acq")
    reader = open_reader(root)
    region = list(reader.metadata["fovs_per_region"])[0]
    fovs = reader.metadata["fovs_per_region"][region]
    assert len(fovs) > 1, f"{label}: fixture must have >1 FOV or 'mosaic' means nothing"
    pairs = export_selection(reader, [(region, f) for f in fovs], tmp_path / "out", blend_px=0)
    assert len(pairs) == 1, f"{label}: {len(fovs)} FOVs gave {len(pairs)} files, want 1 mosaic"
    with tifffile.TiffFile(str(pairs[0][0])) as tf:
        assert len(tf.series) == 1
        # fused, not passed through: wider than the single 8 px fixture tile
        assert tf.series[0].shape[-1] > 8


def test_export_rejects_an_unknown_region_operator(squid_dataset, tmp_path):
    root, _ = squid_dataset
    with pytest.raises(KeyError, match="unknown operator"):
        export_selection(open_reader(root), [("B2", 0)], tmp_path, operator="nope")


def test_export_refuses_a_channel_with_no_name(squid_dataset, tmp_path, monkeypatch):
    """An unnamed channel would mislabel every later one; refusal beats a silent mislabel."""
    root, _ = squid_dataset
    reader = open_reader(root)
    meta = dict(reader.metadata)
    meta["channels"] = [dict(meta["channels"][0], name=""), *meta["channels"][1:]]
    monkeypatch.setattr(type(reader), "metadata", property(lambda self: meta))
    with pytest.raises(ValueError, match="no name"):
        export_selection(reader, [("B2", 0)], tmp_path)
    assert not list(tmp_path.glob("*.ome.tiff"))


def test_export_rejects_a_timepoint_out_of_range(squid_dataset, tmp_path):
    root, _ = squid_dataset
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="out of range"):
        export_selection(open_reader(root), [("B2", 0)], out, time_point=7)
    assert not out.exists() or not list(out.iterdir())


def test_group_selection_groups_by_region_in_first_seen_order():
    from squidxplorer._minerva import group_selection
    assert group_selection([("B3", 1), ("B2", 0), ("B3", 0), ("B3", 1)]) == {
        "B3": [1, 0], "B2": [0]}


def test_export_honours_the_z_operator_choice(squid_dataset, tmp_path):
    """`reference` picks the sharpest plane rather than reducing — a different image.

    Asserted on PIXELS, not on the filename: the name's z-operator token is interpolated from the
    caller's string, so a build that ignored the choice and always MIP'd would still be named
    "reference". Both reductions are computed here from the fixture planes and the file must
    match ITS OWN one and differ from the other.
    """
    root, arrays = squid_dataset
    # correct_illumination=False: the claim under test is that the PROJECTOR choice reaches the
    # pixels, and both expectations below are computed from the raw fixture planes. Leaving the
    # default flat-field on would divide both by a gain field estimated from 4 px tiles, which
    # tests nothing about z-operator dispatch.
    (mip, _), = export_selection(open_reader(root), [("B2", 0)], tmp_path / "a",
                                 z_operator="mip", blend_px=0, correct_illumination=False)
    (ref, _), = export_selection(
        open_reader(root), [("B2", 0)], tmp_path / "b", z_operator="reference", blend_px=0,
        correct_illumination=False,
    )
    assert "mip" in mip.name and "reference" in ref.name

    names = [c["name"] for c in open_reader(root).metadata["channels"]]
    mip_px, ref_px = tifffile.imread(str(mip)), tifffile.imread(str(ref))
    for c_i, ch in enumerate(names):
        planes = [arrays[("B2", 0, z, ch)] for z in range(NZ)]
        expected_mip = np.maximum.reduce(planes)
        # every fixture plane has the same gradient, so Tenengrad ties and `reference` keeps the
        # lowest z — a single plane, NOT the max over z.
        expected_ref = planes[0]
        np.testing.assert_array_equal(mip_px[c_i], expected_mip)
        np.testing.assert_array_equal(ref_px[c_i], expected_ref)
        assert not np.array_equal(expected_mip, expected_ref)      # the two really differ
    assert not np.array_equal(mip_px, ref_px)


def test_export_rejects_an_empty_selection(squid_dataset, tmp_path):
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="nothing selected"):
        export_selection(open_reader(root), [], tmp_path)


@pytest.mark.parametrize(
    "selection, match",
    [([("ZZ", 0)], "unknown region"), ([("B2", 99)], "unknown fov")],
)
def test_export_rejects_an_unknown_target_before_writing(squid_dataset, tmp_path, selection, match):
    root, _ = squid_dataset
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=match):
        export_selection(open_reader(root), selection, out)
    assert not out.exists() or not list(out.iterdir())     # validated before anything is written


def test_export_creates_a_missing_out_dir(squid_dataset, tmp_path):
    root, _ = squid_dataset
    out = tmp_path / "deep" / "nested" / "out"
    export_selection(open_reader(root), [("B2", 0)], out)
    assert out.is_dir()


def test_default_out_dir_never_writes_into_the_acquisition(squid_dataset, tmp_path, monkeypatch):
    """README's "Good to know" promises the tool never writes into the acquisition folder, and
    acquisition volumes are often read-only network shares. Not a temp dir either: Minerva is a
    separate long-lived process and OS sweeping would delete a story it still has open."""
    root, _ = squid_dataset
    monkeypatch.setattr(_minerva.Path, "home", staticmethod(lambda: tmp_path))
    reader = open_reader(root)

    out = default_out_dir(reader)
    assert out == tmp_path / "minerva_export" / root.name
    assert root not in out.parents and out != root

    (ome, _), = export_selection(reader, [("B2", 0)])
    assert ome.parent == out
    assert not (root / "minerva_export").exists()      # the acquisition is untouched
    assert list(root.iterdir())                        # ...and still intact


def test_export_refuses_an_acquisition_with_no_pixel_size(squid_dataset, tmp_path, monkeypatch):
    """Minerva 500s without PhysicalSizeX. Refusing beats writing a fabricated 1.0 scale,
    which would silently corrupt every measurement made downstream."""
    root, _ = squid_dataset
    reader = open_reader(root)
    meta = dict(reader.metadata)
    meta["pixel_size_um"] = None
    monkeypatch.setattr(type(reader), "metadata", property(lambda self: meta))

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no objective pixel size"):
        export_selection(reader, [("B2", 0)], out)
    assert not out.exists() or not list(out.iterdir())


def test_export_reads_only_the_requested_timepoint(squid_dataset, tmp_path):
    """OV6: project_well used to compute every timepoint and have the caller throw all but
    one away — an n_t-fold wasted read of the whole z-stack."""
    root, _ = squid_dataset
    reader = open_reader(root)
    seen_t = []
    real_read = type(reader).read

    def spy(self, region, fov, channel, z_level, time_point=0):
        seen_t.append(time_point)
        return real_read(self, region, fov, channel, z_level, time_point)

    type(reader).read = spy
    try:
        export_selection(reader, [("B2", 0)], tmp_path, t=0)
    finally:
        type(reader).read = real_read
    assert set(seen_t) == {0}


def test_two_timepoints_export_to_two_files_with_different_pixels(multi_time_point_dataset,
                                                                  tmp_path):
    """A MULTI-TIMEPOINT acquisition must land a DIFFERENT image for a different *t*.

    ``test_export_reads_only_the_requested_timepoint`` above pins that the reader is asked for
    one timepoint, but it asks for t=0 on a single-timepoint fixture, so it cannot tell "the
    timepoint reached the pixels" from "there was only ever one timepoint". This does: the
    fixture's planes carry the timepoint in their hundreds digit
    (``time_series_pixel_value``), so the two exports are comparable constants and a t that was
    dropped anywhere in the chain shows up as equal pixels rather than as a message.

    ``correct_illumination=False``: the fixture is 4x4 constant planes and an estimated gain
    field over them is meaningless. Same fixture concession the byte-for-byte test above makes,
    and for the same reason — the claim here is about the TIME axis, not about flat-fielding.
    """
    from tests.conftest import (
        TIME_SERIES_CHANNELS, TIME_SERIES_FOV, TIME_SERIES_NZ, TIME_SERIES_REGION,
        time_series_pixel_value,
    )

    root, _planes = multi_time_point_dataset
    sel = [(TIME_SERIES_REGION, TIME_SERIES_FOV)]
    (ome1, _), = export_selection(open_reader(root), sel, tmp_path / "t1", t=1,
                                  blend_px=0, correct_illumination=False)
    (ome2, _), = export_selection(open_reader(root), sel, tmp_path / "t2", t=2,
                                  blend_px=0, correct_illumination=False)

    assert "_t1_" in ome1.name and "_t2_" in ome2.name, "the filename does not name the timepoint"
    px1, px2 = tifffile.imread(str(ome1)), tifffile.imread(str(ome2))
    assert not np.array_equal(px1, px2), (
        "both timepoints exported the same pixels — the t never reached the read")
    # ...and they are the RIGHT timepoints, not merely two different ones. MIP over z of a
    # constant plane is the brightest z, which is the last one.
    top_z = TIME_SERIES_NZ - 1
    for t, px in ((1, px1), (2, px2)):
        for c in range(len(TIME_SERIES_CHANNELS)):
            assert px[c].min() == px[c].max() == time_series_pixel_value(t, top_z, c)


def test_export_reports_progress_in_regions_not_fovs(squid_dataset, tmp_path):
    """The export unit is a fused mosaic per region, so the readout counts regions. Four FOVs
    across two regions is 2 steps, not 4 — a FOV count here would promise progress the export
    cannot deliver (a region is indivisible: it fuses or it does not)."""
    root, _ = squid_dataset
    seen = []
    export_selection(
        open_reader(root), [("B2", 0), ("B2", 1), ("B3", 0), ("B3", 1)], tmp_path,
        on_progress=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_re_export_overwrites_in_place(squid_dataset, tmp_path):
    """Same selection twice must not accumulate full-resolution TIFFs on the data volume."""
    root, _ = squid_dataset
    export_selection(open_reader(root), [("B2", 0)], tmp_path)
    export_selection(open_reader(root), [("B2", 0)], tmp_path)
    assert len(list(tmp_path.glob("*.ome.tiff"))) == 1


# --- write_ome_tiff --------------------------------------------------------------------------

def test_ome_xml_carries_names_and_physical_size(squid_dataset, tmp_path):
    root, _ = squid_dataset
    (ome, _), = export_selection(open_reader(root), [("B2", 0)], tmp_path)

    with tifffile.TiffFile(str(ome)) as tf:
        assert tf.is_ome, "Minerva needs real OME-XML, not a bare TIFF"
        xml = tf.ome_metadata
    assert CH_IN_YAML in xml and CH_NOT_IN_YAML in xml
    assert "PhysicalSizeX" in xml
    assert "0.325" in xml, "must be the authoritative acquisition.yaml value, not recomputed"


def test_writer_rejects_a_non_ome_suffix(tmp_path):
    """Minerva takes the last two extension components; anything else is 'Invalid tiff file'."""
    img = np.zeros((2, 4, 4), np.uint16)
    with pytest.raises(ValueError, match="ending in"):
        write_ome_tiff(img, tmp_path / "x.tiff", ["a", "b"], 0.3)
    write_ome_tiff(img, tmp_path / "x.ome.tiff", ["a", "b"], 0.3)      # accepted


def test_writer_rejects_a_channel_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="refusing to mislabel"):
        write_ome_tiff(np.zeros((2, 4, 4), np.uint16), tmp_path / "x.ome.tiff", ["only-one"], 0.3)


def test_writer_preserves_dtype(tmp_path):
    img = (np.arange(32, dtype=np.uint16).reshape(2, 4, 4) + 60000).astype(np.uint16)
    path = write_ome_tiff(img, tmp_path / "x.ome.tiff", ["a", "b"], 0.3)
    np.testing.assert_array_equal(tifffile.imread(str(path)), img)


# --- story.json ------------------------------------------------------------------------------

def test_story_points_at_the_ome_with_an_absolute_path(squid_dataset, tmp_path):
    """Author resolves in_file from its own cwd, not ours."""
    root, _ = squid_dataset
    (ome, story), = export_selection(open_reader(root), [("B2", 0)], tmp_path)
    data = json.loads(story.read_text())
    assert data["in_file"] == str(ome.resolve())
    from pathlib import Path
    assert Path(data["in_file"]).is_absolute() and Path(data["in_file"]).exists()
    for key in ("csv_file", "waypoints", "groups", "sample_info"):
        assert key in data, "api_import hard-indexes these keys"


def test_the_story_says_which_timepoint_and_which_fovs_these_pixels_are(squid_dataset, tmp_path):
    """``sample_info.text`` is what an OME-TIFF carries once the log that described it is gone.

    Both facts became variable in the same change and neither was recorded: the exported
    timepoint used to be hardcoded to 0 and a FOV subset was not expressible from any GUI path,
    so "the mosaic of region B2" was unambiguous. It no longer is — the same acquisition now
    yields a different file per timepoint and per box — and a crop that does not SAY it is a
    crop is a measurement waiting to be made on the wrong extent.
    """
    root, _ = squid_dataset
    reader = open_reader(root)
    all_fovs = reader.metadata["fovs_per_region"]["B2"]
    assert len(all_fovs) > 1, "fixture cannot express a crop"

    (_, whole), = export_selection(reader, [("B2", f) for f in all_fovs], tmp_path / "whole")
    (_, crop), = export_selection(reader, [("B2", all_fovs[0])], tmp_path / "crop")

    whole_text = json.loads(whole.read_text())["sample_info"]["text"]
    crop_text = json.loads(crop.read_text())["sample_info"]["text"]

    assert "region B2" in whole_text and "timepoint t=0" in whole_text
    assert f"all {len(all_fovs)} FOV(s)" in whole_text
    assert "CROPPED" not in whole_text, "a whole region must not claim to be a crop"
    assert f"CROPPED to 1 of {len(all_fovs)} FOV(s)" in crop_text, crop_text
    # The registration timepoint is a DIFFERENT number and is still reported separately, so the
    # story can never read as though the pixels came from the timepoint the geometry was solved on.
    assert "flat-field" in crop_text


def test_story_groups_carry_our_channel_colours(squid_dataset, tmp_path):
    """OV1: Minerva ignores OME-TIFF channel colours entirely and colours by index. The
    story groups are the ONLY path for our colours, so this is the assertion that matters."""
    root, _ = squid_dataset
    (_, story), = export_selection(open_reader(root), [("B2", 0)], tmp_path)
    groups = json.loads(story.read_text())["groups"]

    channels = {c["label"]: c for c in groups[0]["channels"]}
    reader = open_reader(root)
    for ch in reader.metadata["channels"]:
        expected = str(ch["display_color"]).lstrip("#").lower()
        assert channels[ch["name"]]["color"].lower() == expected
    # the YAML-nested colour must survive — this is the squid2minerva bug we do not carry
    assert channels[CH_IN_YAML]["color"].lower() == "ff0000"


def test_story_channel_ids_are_the_image_channel_order(squid_dataset, tmp_path):
    """Minerva maps groups onto planes by index, so id must equal the OME channel index."""
    root, _ = squid_dataset
    (_, story), = export_selection(open_reader(root), [("B2", 0)], tmp_path)
    channels = json.loads(story.read_text())["groups"][0]["channels"]
    names = [c["name"] for c in open_reader(root).metadata["channels"]]
    assert [c["id"] for c in channels] == list(range(len(names)))
    assert [c["label"] for c in channels] == names


def test_auto_groups_contrast_is_normalised_and_ordered():
    img = np.stack([
        np.full((8, 8), 100, np.uint16),
        (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000),
    ])
    (group,) = auto_groups(img, ["flat", "ramp"], [(255, 0, 0), (0, 255, 0)])
    for ch in group["channels"]:
        assert 0.0 <= ch["min"] <= ch["max"] <= 1.0
    assert group["channels"][1]["max"] > group["channels"][0]["max"]


def test_write_story_strips_the_ome_suffix_from_the_dataset_name(tmp_path):
    ome = tmp_path / "plate_B2_fov0.ome.tiff"
    ome.write_bytes(b"")
    story = write_story(tmp_path / "s.story.json", ome, [])
    assert json.loads(story.read_text())["out_name"] == "plate_B2_fov0"


# --- the on-screen LUTs reach the story ------------------------------------------------------
#
# Julio: "channels need to be set to specific colors". The export's own defaults are the
# acquisition's display_color plus a 1/99.9 percentile stretch, neither of which knows the user
# recoloured a layer or dragged a contrast slider. These assert the OUTCOME - what is in the
# .story.json Minerva reads - and never that some function was called.

def _story_channels(story) -> dict:
    return {c["label"]: c for c in json.loads(story.read_text())["groups"][0]["channels"]}


def test_a_supplied_lut_reaches_the_story_groups(squid_dataset, tmp_path):
    """The whole point of the parameter: colour AND contrast in the file Minerva opens."""
    root, _ = squid_dataset
    reader = open_reader(root)
    name = reader.metadata["channels"][0]["name"]
    luts = {name: {"clim": (100.0, 500.0), "rgb": (0x12, 0x34, 0x56)}}

    (_, story), = export_selection(reader, [("B2", 0)], tmp_path, luts=luts)
    ch = _story_channels(story)[name]

    assert ch["color"] == "123456", "the on-screen colour, not the acquisition's display_color"
    # uint16 planes, so clim is normalised against 65535 exactly as the percentiles are.
    assert ch["min"] == pytest.approx(100.0 / 65535.0, abs=1e-6)
    assert ch["max"] == pytest.approx(500.0 / 65535.0, abs=1e-6)


def test_omitting_luts_keeps_the_percentile_and_display_colour_behaviour(squid_dataset, tmp_path):
    """The default must not move. The plate-level export and the CLI have no screen to read, so
    the percentiles are the right answer there and every existing caller passes nothing.

    Asserted as an EQUALITY between two exports rather than against remembered numbers: that is
    what makes it a regression test for "did adding luts change the no-luts path".
    """
    root, _ = squid_dataset
    before = export_selection(open_reader(root), [("B2", 0)], tmp_path / "a")
    after = export_selection(open_reader(root), [("B2", 0)], tmp_path / "b", luts=None)

    a, b = _story_channels(before[0][1]), _story_channels(after[0][1])
    assert a == b
    for ch in open_reader(root).metadata["channels"]:
        assert a[ch["name"]]["color"].lower() == str(ch["display_color"]).lstrip("#").lower()
        assert 0.0 <= a[ch["name"]]["min"] < a[ch["name"]]["max"] <= 1.0


def test_a_channel_missing_from_luts_keeps_its_defaults(squid_dataset, tmp_path):
    """A view window showing three of four channels must not blank the fourth. Per channel, not
    all-or-nothing: the one that IS in luts moves and the ones that are not do not."""
    root, _ = squid_dataset
    reader = open_reader(root)
    names = [c["name"] for c in reader.metadata["channels"]]
    assert len(names) > 1, "this fixture needs more than one channel to say anything"
    luts = {names[0]: {"clim": (7.0, 9.0), "rgb": (1, 2, 3)}}

    plain = _story_channels(export_selection(reader, [("B2", 0)], tmp_path / "a")[0][1])
    mixed = _story_channels(export_selection(reader, [("B2", 0)], tmp_path / "b", luts=luts)[0][1])

    assert mixed[names[0]] != plain[names[0]], "the supplied channel moved"
    for n in names[1:]:
        assert mixed[n] == plain[n], f"{n} is not in luts and must be untouched"


def test_an_unrepresentable_colormap_falls_back_instead_of_guessing(squid_dataset, tmp_path):
    """``rgb: None`` is what colormap_hue_rgb returns for a multi-stop map (viridis, turbo).
    Minerva has ONE colour field per channel and cannot hold a ramp, so the acquisition's colour
    is kept. Emitting the ramp's brightest stop would put a colour into Minerva that is on no
    screen - which is the failure this asserts against. The CONTRAST still comes from the screen:
    a colormap we cannot represent says nothing about where the slider is."""
    root, _ = squid_dataset
    reader = open_reader(root)
    ch0 = reader.metadata["channels"][0]
    luts = {ch0["name"]: {"clim": (10.0, 20.0), "rgb": None}}

    (_, story), = export_selection(reader, [("B2", 0)], tmp_path, luts=luts)
    ch = _story_channels(story)[ch0["name"]]

    assert ch["color"].lower() == str(ch0["display_color"]).lstrip("#").lower()
    assert ch["max"] == pytest.approx(20.0 / 65535.0, abs=1e-6)


def test_auto_groups_takes_clim_in_raw_units_not_normalised_ones():
    """The unit is the one napari's contrast slider works in, which is also the unit the
    percentiles are computed in - so both paths divide by dtype_max in the same place."""
    img = np.stack([np.full((8, 8), 100, np.uint16), np.full((8, 8), 200, np.uint16)])
    (group,) = auto_groups(
        img, ["a", "b"], [(255, 0, 0), (0, 255, 0)],
        luts={"a": {"clim": (0.0, 65535.0), "rgb": None}},
    )
    a = group["channels"][0]
    assert a["min"] == 0.0 and a["max"] == 1.0


# --- render.py: the destination that needs no file picking ------------------------------------

def test_render_exhibit_says_which_piece_is_missing(monkeypatch, tmp_path):
    """render.py runs as a script under a FOREIGN venv, so nothing it does raises into us. Every
    way it can be unavailable must therefore be named here, not discovered as a silent no-op."""
    monkeypatch.setattr(_minerva, "minerva_home", lambda: None)
    with pytest.raises(FileNotFoundError, match="checkout not found"):
        _minerva.render_exhibit("a.ome.tiff", "a.story.json", tmp_path / "out")

    home = tmp_path / "home"
    (home / "vendor" / "minerva-author" / "src").mkdir(parents=True)
    (home / "vendor" / "minerva-author" / "src" / "app.py").write_text("")
    monkeypatch.setattr(_minerva, "minerva_home", lambda: home)
    with pytest.raises(FileNotFoundError, match="no .venv interpreter"):
        _minerva.render_exhibit("a.ome.tiff", "a.story.json", tmp_path / "out")

    py = home / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    with pytest.raises(FileNotFoundError, match="no renderer"):
        _minerva.render_exhibit("a.ome.tiff", "a.story.json", tmp_path / "out")


def _fake_checkout(tmp_path, render_body: str):
    """A checkout whose render.py is a real Python script we control. Runs the actual subprocess
    plumbing - Popen, poll, exit code, stdout - instead of mocking the thing under test."""
    import sys

    home = tmp_path / "home"
    src = home / "vendor" / "minerva-author" / "src"
    src.mkdir(parents=True)
    (src / "app.py").write_text("")
    (src / "render.py").write_text(render_body)
    py = home / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.symlink_to(sys.executable)
    return home


def test_render_exhibit_returns_the_index_html_it_wrote(monkeypatch, tmp_path):
    home = _fake_checkout(tmp_path, (
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[3])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'index.html').write_text(sys.argv[1] + '|' + sys.argv[2])\n"
    ))
    monkeypatch.setattr(_minerva, "minerva_home", lambda: home)
    ome, story = tmp_path / "x.ome.tiff", tmp_path / "x.story.json"
    ome.write_text(""); story.write_text("{}")

    index = _minerva.render_exhibit(ome, story, tmp_path / "out")

    assert index == tmp_path / "out" / "index.html" and index.is_file()
    # The renderer is handed OUR pair, resolved - its own cwd is the checkout, not ours.
    assert index.read_text() == f"{ome.resolve()}|{story.resolve()}"


def test_render_exhibit_reports_the_renderers_own_stderr(monkeypatch, tmp_path):
    """A non-zero exit is the only failure signal a foreign script gives us. Losing its output
    would leave the user with 'it did not work' and nothing else."""
    home = _fake_checkout(tmp_path, (
        "import sys\n"
        "print('Image is missing OME-XML pixel size', file=sys.stderr)\n"
        "sys.exit(3)\n"
    ))
    monkeypatch.setattr(_minerva, "minerva_home", lambda: home)
    with pytest.raises(RuntimeError, match="exited 3"):
        _minerva.render_exhibit(tmp_path / "x.ome.tiff", tmp_path / "x.story.json", tmp_path / "o")
    try:
        _minerva.render_exhibit(tmp_path / "x.ome.tiff", tmp_path / "x.story.json", tmp_path / "o")
    except RuntimeError as exc:
        assert "OME-XML pixel size" in str(exc)


def test_render_exhibit_refuses_a_zero_exit_that_wrote_nothing(monkeypatch, tmp_path):
    """Exit 0 is not the deliverable; an index.html is. Returning a path to a directory with no
    viewer in it would hand the user a broken link and call it a success."""
    home = _fake_checkout(tmp_path, "pass\n")
    monkeypatch.setattr(_minerva, "minerva_home", lambda: home)
    with pytest.raises(RuntimeError, match="no index.html"):
        _minerva.render_exhibit(tmp_path / "x.ome.tiff", tmp_path / "x.story.json", tmp_path / "o")


def test_render_exhibit_abandons_a_long_render_when_told_to_stop(monkeypatch, tmp_path):
    """Measured at 132 s for one real region, and closeEvent joins the worker. A stop flag the
    poll never reads would hold the window for the rest of it."""
    import time

    home = _fake_checkout(tmp_path, "import time\ntime.sleep(60)\n")
    monkeypatch.setattr(_minerva, "minerva_home", lambda: home)
    stop = [False]
    threading.Timer(0.3, lambda: stop.__setitem__(0, True)).start()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped"):
        _minerva.render_exhibit(tmp_path / "x.ome.tiff", tmp_path / "x.story.json",
                                tmp_path / "o", should_stop=lambda: stop[0])
    assert time.monotonic() - t0 < 20.0          # not 60 - the poll honoured the flag


def test_the_internet_requirement_is_stated_somewhere_a_user_meets_it():
    """Both Minerva front ends load their JavaScript from jsdelivr, so both need internet to
    VIEW. That was recorded nowhere in this codebase and a user meets it as a blank page."""
    assert "internet" in _minerva.NEEDS_INTERNET_NOTE.lower()
    assert "CDN" in _minerva.NEEDS_INTERNET_NOTE or "cdn" in _minerva.NEEDS_INTERNET_NOTE


# --- launch ----------------------------------------------------------------------------------

def test_launch_returns_false_when_not_installed(monkeypatch):
    """The export already succeeded by then — a missing sibling checkout must never turn it
    into a failure, and must never raise."""
    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: False)
    monkeypatch.setattr(_minerva, "minerva_home", lambda: None)
    assert launch_minerva("/tmp/x.story.json") is False


def test_launch_returns_false_when_the_venv_is_missing(monkeypatch, tmp_path):
    """minerva-author has no venv of its own; without explorer's we cannot start it."""
    app = tmp_path / "vendor" / "minerva-author" / "src" / "app.py"
    app.parent.mkdir(parents=True)
    app.write_text("")
    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: False)
    monkeypatch.setenv(_minerva.MINERVA_HOME_ENV, str(tmp_path))
    assert _minerva.minerva_home() == tmp_path
    assert launch_minerva() is False


def test_launch_reuses_an_already_running_server(monkeypatch):
    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: True)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    def boom(*a, **k):                       # must not try to spawn a second server
        raise AssertionError("spawned a server when one was already running")

    monkeypatch.setattr("subprocess.Popen", boom)
    assert launch_minerva() is True
    assert opened == [_minerva.MINERVA_URL]


def test_a_cold_start_opens_exactly_one_tab_and_it_is_not_ours(monkeypatch, tmp_path):
    """minerva-author opens the browser ITSELF and cannot be told not to.

    ``src/app.py`` v1.21.0 (commit c555515) defines ``open_browser()`` at :2033 and calls it at
    :2050 AND :2053, once in each arm of ``if "--dev" in sys.argv`` - its only argv handling. So
    a cold start opened two tabs: Author's and ours. Ours is the one we can drop.

    Asserted on the OUTCOME (how many opens happen) rather than on a flag, so it stays true if
    the implementation changes its mind about how it knows.
    """
    app = tmp_path / "vendor" / "minerva-author" / "src" / "app.py"
    app.parent.mkdir(parents=True)
    app.write_text("")
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv(_minerva.MINERVA_HOME_ENV, str(tmp_path))

    live = [False]                                    # comes up on the second poll
    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: live[0])
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    def spawn(*a, **k):
        live[0] = True                                # ...and Author opens its own tab here
        return None

    monkeypatch.setattr("subprocess.Popen", spawn)

    assert launch_minerva() is True
    assert opened == [], "Minerva Author already opened a tab; a second one is the defect"


def test_an_already_running_server_still_gets_a_tab_from_us(monkeypatch):
    """The other half of the same rule, and the reason this is not just 'delete the call'.

    Nothing is spawned when the server is already answering, so nobody else is about to open
    anything. Dropping our call unconditionally would mean the second export of a session
    silently opened no browser at all.
    """
    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: True)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("subprocess.Popen",
                        lambda *a, **k: pytest.fail("spawned a second server"))
    assert launch_minerva() is True
    assert opened == [_minerva.MINERVA_URL]


def test_launch_abandons_the_liveness_wait_when_told_to_stop(monkeypatch, tmp_path):
    """The wait is up to 90 s and the GUI JOINS this thread on close, so a stop flag the poll
    never reads froze the window for the rest of it (measured 84 s). Bounded here at ~1 s."""
    import time

    app = tmp_path / "vendor" / "minerva-author" / "src" / "app.py"
    app.parent.mkdir(parents=True)
    app.write_text("")
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv(_minerva.MINERVA_HOME_ENV, str(tmp_path))
    monkeypatch.setattr(_minerva, "is_running", lambda timeout=1.0: False)   # never comes up
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: None)
    monkeypatch.setattr("webbrowser.open", lambda url: None)

    stop = [False]
    t0 = time.monotonic()
    # flip the flag from another thread while the poll is sleeping
    threading.Timer(0.3, lambda: stop.__setitem__(0, True)).start()
    assert launch_minerva(timeout=90.0, should_stop=lambda: stop[0]) is False
    assert time.monotonic() - t0 < 5.0            # not 90 — the poll honoured the flag


def test_launch_does_not_start_a_server_when_already_stopped(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("spawned a server after the caller gave up")

    monkeypatch.setattr("subprocess.Popen", boom)
    assert launch_minerva(should_stop=lambda: True) is False


def test_minerva_home_prefers_the_env_var(monkeypatch, tmp_path):
    app = tmp_path / "vendor" / "minerva-author" / "src" / "app.py"
    app.parent.mkdir(parents=True)
    app.write_text("")
    monkeypatch.setenv(_minerva.MINERVA_HOME_ENV, str(tmp_path))
    assert _minerva.minerva_home() == tmp_path


def test_minerva_home_is_none_without_the_app(monkeypatch, tmp_path):
    monkeypatch.setenv(_minerva.MINERVA_HOME_ENV, str(tmp_path))
    monkeypatch.setattr(_minerva.Path, "home", staticmethod(lambda: tmp_path))
    assert _minerva.minerva_home() is None
