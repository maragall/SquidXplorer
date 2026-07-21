"""IMA-228 — Minerva export: OME-TIFF + .story.json, and the best-effort launch.

Every requirement asserted here was read out of minerva-author's own ``src/app.py``; none of
it is documented upstream, and the checkout is an untagged ``--depth 1`` clone. These tests
pin OUR side of that contract. They cannot prove Minerva still honours it — that is what the
manual check in ``docs/ima-228-eng-review.md`` (T10) is for.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
import tifffile

from squidmip import _minerva
from squidmip._minerva import (
    auto_groups,
    default_out_dir,
    export_selection,
    launch_minerva,
    write_ome_tiff,
    write_story,
)
from squidmip.reader import open_reader
from tests.conftest import CH_IN_YAML, CH_NOT_IN_YAML, NZ, _pixel_value


# --- export_selection ------------------------------------------------------------------------

def test_export_writes_one_pair_per_fov(squid_dataset, tmp_path):
    root, _ = squid_dataset
    out = tmp_path / "out"
    pairs = export_selection(open_reader(root), [("B2", 0), ("B3", 1)], out)

    assert len(pairs) == 2
    for ome, story in pairs:
        assert ome.exists() and story.exists()
        assert ome.name.endswith(".ome.tiff")
        assert story.name.endswith(".story.json")
    # order is the caller's order, not completion order — IMA-205 relies on this
    assert "B2_fov0" in pairs[0][0].name
    assert "B3_fov1" in pairs[1][0].name


def test_exported_pixels_are_the_mip_byte_for_byte(squid_dataset, tmp_path):
    """The OME-TIFF must carry the real projection in native uint16 — no rescale, no cast."""
    root, arrays = squid_dataset
    (ome, _), = export_selection(open_reader(root), [("B3", 1)], tmp_path)

    written = tifffile.imread(str(ome))
    assert written.dtype == np.uint16

    reader = open_reader(root)
    names = [c["name"] for c in reader.metadata["channels"]]
    for c_i, ch in enumerate(names):
        expected = np.maximum.reduce([arrays[("B3", 1, z, ch)] for z in range(NZ)])
        np.testing.assert_array_equal(written[c_i], expected)


def test_export_honours_the_projector_choice(squid_dataset, tmp_path):
    """`reference` picks the sharpest plane rather than reducing — a different image.

    Asserted on PIXELS, not on the filename: the name's projector token is interpolated from the
    caller's string, so a build that ignored the choice and always MIP'd would still be named
    "reference". Both reductions are computed here from the fixture planes and the file must
    match ITS OWN one and differ from the other.
    """
    root, arrays = squid_dataset
    (mip, _), = export_selection(open_reader(root), [("B2", 0)], tmp_path / "a", projector="mip")
    (ref, _), = export_selection(
        open_reader(root), [("B2", 0)], tmp_path / "b", projector="reference"
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

    def spy(self, region, fov, channel, z, t=0):
        seen_t.append(t)
        return real_read(self, region, fov, channel, z, t)

    type(reader).read = spy
    try:
        export_selection(reader, [("B2", 0)], tmp_path, t=0)
    finally:
        type(reader).read = real_read
    assert set(seen_t) == {0}


def test_export_reports_progress(squid_dataset, tmp_path):
    root, _ = squid_dataset
    seen = []
    export_selection(
        open_reader(root), [("B2", 0), ("B2", 1)], tmp_path, on_progress=lambda d, t: seen.append((d, t))
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
