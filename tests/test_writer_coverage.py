"""IMA-254: the reader covers EVERY Squid output writer, and never skips one in silence.

Two defects, one root cause. ``squidxplorer.reader`` served three of Squid's six on-disk shapes; of the
three it did not serve, MULTI_PAGE_TIFF was the worst, because it was not refused — the discovery
glob hit a bare ``continue`` on every single file and the acquisition came out EMPTY. An operator
pointing the tool at a real multi-page run was told the folder had no images in it. The comment on
that ``continue`` even named the format it was discarding. The code knew and said nothing.

The root cause is not the missing formats, it is that coverage tracked this machine's ``~/Downloads``
rather than Squid's source. Only two acquisitions were ever tested against, and both came from the
same writer. So the deliverable is this file plus ``tests/writer_fixtures.py``: a tiny synthetic
acquisition per writer, walked by a parametrised suite, so a new Squid writer fails HERE rather
than at a customer.

What is asserted, per writer:

* ``open_reader`` dispatches to the right reader class — not "a reader", the right one;
* ``metadata`` resolves regions, fovs_per_region, channels, n_z, n_t IDENTICALLY across all six
  (they encode the same acquisition, so any difference is a reader bug);
* every plane's pixels are EXACT against a direct read of the bytes on disk, per writer, using
  that format's own native library rather than another squidxplorer reader (an oracle that is the
  thing under test proves nothing);
* ``fov_positions_um`` is populated and in MICROMETRES wherever the writer records positions.

Plus, separately: every unsupported or malformed layout fails LOUD and names the format, and no
code path silently ``continue``s past a file matching a known Squid naming pattern.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import tifffile

import squidxplorer
from squidxplorer import open_reader
from squidxplorer.reader import _MM_TO_UM
from tests.conftest import CHANNELS, FOVS, NZ, REGIONS, _FOV_MM
from tests import writer_fixtures
from tests.writer_fixtures import WRITERS, expected_arrays, plane

_IDS = [w[0] for w in WRITERS]


@pytest.fixture(params=WRITERS, ids=_IDS)
def any_writer(request, tmp_path):
    """(label, root, reader_class_name, records_positions) for one writer's tiny acquisition."""
    label, builder, reader_cls, records_positions = request.param
    root = builder(tmp_path / re.sub(r"\W+", "_", label))
    return label, root, reader_cls, records_positions


# --- dispatch ---------------------------------------------------------------------------------

def test_every_writer_dispatches_to_its_reader(any_writer):
    """The gap IMA-254 closes: three of these six used to raise, silently empty, or mis-dispatch."""
    label, root, reader_cls, _ = any_writer
    reader = open_reader(root)
    assert type(reader).__name__ == reader_cls, f"{label} dispatched to {type(reader).__name__}"


def test_every_writer_resolves_the_same_acquisition(any_writer):
    """All six fixtures encode ONE acquisition. Any metadata difference is a reader bug."""
    label, root, _, _ = any_writer
    meta = open_reader(root).metadata
    assert meta["regions"] == REGIONS, label
    assert meta["fovs_per_region"] == {r: list(FOVS) for r in REGIONS}, label
    assert meta["n_z"] == NZ, label
    assert meta["n_t"] == 1, label
    # Channel SET, not order: the TIFF readers derive names from filenames/PageName and sort
    # them, while the Zarr reader takes omero's declared order (which is the C-axis order and
    # must not be resorted). Both are correct, and no consumer indexes channels positionally —
    # every lookup in this package is by name.
    assert {c["name"] for c in meta["channels"]} == set(CHANNELS), label
    assert meta["frame_shape"] == writer_fixtures.FRAME, label
    assert np.dtype(meta["dtype"]) == np.uint16, label


def test_every_writer_reads_exact_pixels(any_writer):
    """Every plane of every writer, byte-exact — not a shape check, not a checksum."""
    label, root, _, _ = any_writer
    reader = open_reader(root)
    want = expected_arrays()
    for (region, fov, z, channel), expected in want.items():
        got = reader.read(region, fov, channel, z)
        assert got.dtype == expected.dtype, f"{label} {region}/{fov}/{z}/{channel}"
        np.testing.assert_array_equal(
            got, expected, err_msg=f"{label} region={region} fov={fov} z={z} ch={channel}"
        )


def test_every_writer_has_the_same_metadata_key_set(any_writer):
    """The shared-interface promise: no consumer may need to know which writer produced a folder."""
    _, root, _, _ = any_writer
    assert set(open_reader(root).metadata) == {
        "regions", "fovs_per_region", "fov_positions_um", "channels", "n_z", "z_levels",
        "dz_um", "pixel_size_um", "wellplate_format", "frame_shape", "dtype", "n_t",
    }


def test_every_writer_populates_positions_in_micrometres(any_writer):
    """``fov_positions_um`` is populated for every writer, in um, at the documented offsets.

    Five of the six get positions from the sibling coordinates.csv; MULTI_PAGE_TIFF gets them
    from its own pages and ships no CSV at all. Either way the key, the units and the values are
    the same — the mm -> um conversion happens once, at whichever producer owns it, and nothing
    downstream compensates.
    """
    label, root, _, _ = any_writer
    positions = open_reader(root).metadata["fov_positions_um"]
    assert set(positions) == {(r, f) for r in REGIONS for f in FOVS}, label
    for (region, fov), (x_um, y_um) in positions.items():
        want_x, want_y = _FOV_MM[fov]
        # 1 um tolerance absorbs the multi-page fixture's deliberate per-z stage jitter.
        assert abs(x_um - want_x * _MM_TO_UM) < 1.0, f"{label} {region}/{fov} x={x_um}"
        assert abs(y_um - want_y * _MM_TO_UM) < 1.0, f"{label} {region}/{fov} y={y_um}"
    # The 1000x tell: these FOVs are 0.5 mm apart, i.e. 500 um. In mm the span would be 0.5.
    xs = [v[0] for v in positions.values()]
    assert 400 < max(xs) - min(xs) < 600, f"{label} x span {max(xs) - min(xs)} is not micrometres"


# --- independent oracles: read the bytes with each format's own library ------------------------

def test_individual_tiff_pixels_against_a_direct_tifffile_read(tmp_path):
    root = writer_fixtures.build_individual_tiff(tmp_path / "acq")
    reader = open_reader(root)
    for region in REGIONS:
        for fov in FOVS:
            for z in range(NZ):
                for channel in CHANNELS:
                    direct = tifffile.imread(root / "0" / f"{region}_{fov}_{z}_{channel}.tiff")
                    np.testing.assert_array_equal(reader.read(region, fov, channel, z), direct)


def test_multipage_pixels_against_a_direct_page_read(tmp_path):
    """Oracle: locate the page by its OWN metadata, independently of the reader's index."""
    root = writer_fixtures.build_multi_page_tiff(tmp_path / "acq")
    reader = open_reader(root)
    for region in REGIONS:
        for fov in FOVS:
            path = root / "0" / f"{region}_{fov:0{writer_fixtures.FILE_ID_PADDING}}_stack.tiff"
            with tifffile.TiffFile(path) as tif:
                for page in tif.pages:
                    payload = json.loads(page.tags.get(270).value)
                    channel = page.tags.get(285).value
                    np.testing.assert_array_equal(
                        reader.read(region, fov, channel, payload["z_level"]), page.asarray()
                    )


def test_ome_tiff_pixels_against_a_direct_series_read(tmp_path):
    root = writer_fixtures.build_ome_tiff(tmp_path / "acq")
    reader = open_reader(root)
    pad = writer_fixtures.FILE_ID_PADDING
    for region in REGIONS:
        for fov in FOVS:
            stack = tifffile.imread(root / "ome_tiff" / f"{region}_{fov:0{pad}}.ome.tiff")
            # imread drops the size-1 T axis; restore the writer's declared TZCYX to index it.
            stack = stack.reshape((writer_fixtures.N_T, NZ, len(CHANNELS)) + writer_fixtures.FRAME)
            for z in range(NZ):
                for c_i, channel in enumerate(CHANNELS):
                    np.testing.assert_array_equal(
                        reader.read(region, fov, channel, z), stack[0, z, c_i]
                    )


@pytest.mark.parametrize("builder,array_of", [
    (writer_fixtures.build_zarr_hcs,
     lambda root, region, fov: root / "plate.ome.zarr" / region[0] / region[1:] / str(fov) / "0"),
    (writer_fixtures.build_zarr_per_fov,
     lambda root, region, fov: root / "zarr" / region / f"fov_{fov}.ome.zarr" / "0"),
], ids=["hcs", "per_fov"])
def test_zarr_5d_pixels_against_a_direct_tensorstore_read(tmp_path, builder, array_of):
    """Oracle: open the array with tensorstore directly and index (T, C, Z, Y, X) by hand."""
    import tensorstore as ts

    root = builder(tmp_path / "acq")
    reader = open_reader(root)
    for region in REGIONS:
        for fov in FOVS:
            arr = ts.open({"driver": "zarr3",
                           "kvstore": {"driver": "file",
                                       "path": str(array_of(root, region, fov))}},
                          open=True).result()
            for z in range(NZ):
                for c_i, channel in enumerate(CHANNELS):
                    direct = np.asarray(arr[0, c_i, z].read().result())
                    np.testing.assert_array_equal(reader.read(region, fov, channel, z), direct)


def test_zarr_6d_pixels_against_a_direct_tensorstore_read(tmp_path):
    """The 6-D layout's whole risk is the leading FOV axis: FOV 1 must not return FOV 0."""
    import tensorstore as ts

    root = writer_fixtures.build_zarr_6d(tmp_path / "acq")
    reader = open_reader(root)
    for region in REGIONS:
        arr = ts.open({"driver": "zarr3",
                       "kvstore": {"driver": "file",
                                   "path": str(root / "zarr" / region / "acquisition.zarr")}},
                      open=True).result()
        for f_i, fov in enumerate(FOVS):
            for z in range(NZ):
                for c_i, channel in enumerate(CHANNELS):
                    direct = np.asarray(arr[f_i, 0, c_i, z].read().result())
                    np.testing.assert_array_equal(reader.read(region, fov, channel, z), direct)


def test_zarr_6d_fovs_are_distinct_planes_not_all_fov_zero(tmp_path):
    """A regression guard aimed straight at the failure mode a 6-D store invites.

    Before this change ``_Multiscale.index`` had no ``fov`` parameter, so an unrecognised leading
    axis fell through ``picks.get(name, 0)`` and every FOV of a region returned FOV 0's pixels —
    a complete, plausible, wrong mosaic with no error anywhere.
    """
    root = writer_fixtures.build_zarr_6d(tmp_path / "acq")
    reader = open_reader(root)
    a = reader.read(REGIONS[0], FOVS[0], CHANNELS[0], 0)
    b = reader.read(REGIONS[0], FOVS[1], CHANNELS[0], 0)
    assert not np.array_equal(a, b), "FOV 1 returned FOV 0's pixels"
    np.testing.assert_array_equal(b, plane(REGIONS[0], FOVS[1], 0, CHANNELS[0]))


# --- the silence, specifically ----------------------------------------------------------------

def test_a_multipage_acquisition_is_never_reported_as_empty(tmp_path):
    """THE bug. Every file took a bare ``continue`` and the folder read as having no images."""
    root = writer_fixtures.build_multi_page_tiff(tmp_path / "acq")
    meta = open_reader(root).metadata
    assert meta["regions"] and meta["fovs_per_region"] and meta["channels"]
    assert meta["n_z"] == NZ


def test_the_individual_tiff_reader_refuses_stacks_by_name_instead_of_skipping(tmp_path):
    """Forcing the WRONG reader onto multi-page output must raise and name both formats."""
    from squidxplorer.reader import SquidReader

    root = writer_fixtures.build_multi_page_tiff(tmp_path / "acq")
    with pytest.raises(ValueError) as exc:
        SquidReader(root).metadata
    message = str(exc.value)
    assert "MULTI_PAGE_TIFF" in message
    assert "_stack.tiff" in message
    assert "SquidMultiPageTiffReader" in message


# The names of the regexes/constants that encode a KNOWN Squid on-disk naming convention. A
# ``continue`` guarded by one of these is, by definition, discarding a file the project can name.
_KNOWN_PATTERNS = ("_STACK_STEM_RE", "_STEM_RE", "_OME_STEM_RE", "_PER_FOV_ZARR_RE",
                   "_SIXD_ZARR_NAME")


def find_silent_skips(lines) -> list:
    """Lines where a ``continue`` is the WHOLE body of a branch that tested a Squid pattern.

    Deliberately narrow, on two axes:

    * only a ``continue`` whose immediately preceding statement ends in ``:`` counts — that is a
      branch whose entire effect is "skip this file". A ``continue`` that merely advances the loop
      after doing the work (``index[key] = ...; continue``) is normal control flow, not a skip.
    * the branch must mention a known Squid naming pattern, and must not ``raise`` — refusing by
      name IS the fix, so a branch that raises is the desired shape, not an offender.

    Returned as ``[(line_number, text)]``. Shared by the guard and by its mutation check, so the
    mutation check exercises the real detector rather than a paraphrase of it.
    """
    offenders = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped != "continue" and not stripped.startswith("continue "):
            continue
        previous = next((lines[j].strip() for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        if not previous.endswith(":"):
            continue                    # the branch did real work first; this only advances it
        window = "\n".join(lines[max(0, i - 6):i])
        if any(name in window for name in _KNOWN_PATTERNS) and "raise" not in window:
            offenders.append((i + 1, stripped))
    return offenders


def test_no_reader_silently_continues_past_a_known_squid_filename():
    """A source-level guard, because this defect is invisible at runtime BY CONSTRUCTION.

    A ``continue`` that discards a file matching a Squid naming pattern produces no exception, no
    warning and no log line — the acquisition simply comes out smaller, or empty. No behavioural
    test catches the general case, because there is nothing to observe. So the general case is
    asserted against the source: in reader.py, no branch whose only body is ``continue`` may be
    guarded by a known Squid naming pattern.

    ``continue`` is a fine statement. What is banned is skipping a file the project KNOWS how to
    name, without saying so.
    """
    lines = Path(squidxplorer.reader.__file__).read_text().splitlines()
    offenders = find_silent_skips(lines)
    assert not offenders, (
        "reader.py skips files matching a known Squid naming pattern without raising:\n"
        + "\n".join(f"  reader.py:{n}: {text}" for n, text in offenders)
    )


def test_the_silent_skip_guard_actually_fires_on_a_reintroduced_skip():
    """Mutation-check: plant the exact line IMA-254 removed and confirm the detector catches it.

    A guard that has never failed is a guard that might not work. This is the original
    ``reader.py:432`` verbatim, comment and all.
    """
    reintroduced = [
        "            m = _STEM_RE.match(f.stem)",
        "            if not m:",
        "                continue  # e.g. {region}_{fov}_stack.tiff (multi-page) - "
        "not this reader's format",
    ]
    assert find_silent_skips(reintroduced), "the detector does not catch the original IMA-254 bug"


def test_the_silent_skip_guard_does_not_fire_on_normal_loop_control():
    """The other half of the mutation check: the detector must not cry wolf on working code.

    A guard that flags every ``continue`` would be turned off within a week, which is the same
    outcome as not having it.
    """
    benign = [
        "            m = _STEM_RE.match(f.stem)",
        "            if m:",
        "                index[key] = f.suffix",
        "                continue",
    ]
    refusing = [
        "            if _STACK_STEM_RE.match(f.stem):",
        "                raise ValueError('multi-page, use SquidMultiPageTiffReader')",
        "            if other:",
        "                continue",
    ]
    assert not find_silent_skips(benign)
    assert not find_silent_skips(refusing)


# --- unsupported and corrupt layouts fail loud, naming the format -------------------------------

def _message_names_formats(message: str) -> None:
    """Every refusal must say what it looked for. An error that only says 'no' is a dead end."""
    assert "{region}" in message or "ome.zarr" in message or "ome_tiff" in message, message


def test_an_unrecognised_folder_names_every_format_it_looked_for(tmp_path):
    root = tmp_path / "acq"
    root.mkdir()
    (root / "0").mkdir()
    tifffile.imwrite(root / "0" / "not_a_squid_name.tiff", np.zeros((4, 4), np.uint16))
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    message = str(exc.value)
    for expected in ("{region}_{fov}_{z}_{channel}.tiff", "{region}_{fov}_stack.tiff",
                     "ome_tiff", "plate.ome.zarr"):
        assert expected in message, f"refusal does not mention {expected}: {message}"
    assert "not_a_squid_name.tiff" in message, "refusal does not say what it DID find"
    _message_names_formats(message)


def test_an_empty_acquisition_folder_refuses_rather_than_reporting_zero_images(tmp_path):
    root = tmp_path / "acq"
    root.mkdir()
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    _message_names_formats(str(exc.value))


def test_an_unreadable_non_hcs_zarr_folder_names_the_zarr_layouts(tmp_path):
    """A ``zarr/`` folder with region dirs but no store must not fall through to the TIFF reader.

    Falling through produced "no {region}_{fov}_{z}_{channel}.tiff found" for a Zarr acquisition —
    a true statement that sends the reader looking for the wrong thing entirely.
    """
    root = tmp_path / "acq"
    (root / "zarr" / "manual0").mkdir(parents=True)
    (root / "zarr" / "manual0" / "readme.txt").write_text("nothing zarr-shaped here")
    with pytest.raises(ValueError) as exc:
        open_reader(root)
    message = str(exc.value)
    assert "fov_{n}.ome.zarr" in message
    assert "acquisition.zarr" in message


def test_a_region_folder_with_no_store_is_named_not_dropped(tmp_path):
    """One unreadable region among readable ones must raise, not silently shrink ``regions``."""
    root = writer_fixtures.build_zarr_per_fov(tmp_path / "acq")
    (root / "zarr" / "orphan").mkdir()
    (root / "zarr" / "orphan" / "stray.txt").write_text("x")
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    assert "orphan" in str(exc.value)
    assert "fov_{n}.ome.zarr" in str(exc.value)


def test_a_stack_page_with_no_metadata_is_refused_by_name(tmp_path):
    """A page missing its ImageDescription cannot be placed; page ORDER must not be used instead."""
    root = tmp_path / "acq"
    (root / "0").mkdir(parents=True)
    with tifffile.TiffWriter(root / "0" / "B2_0000_stack.tiff", append=True) as w:
        w.write(np.zeros((4, 4), np.uint16))
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    message = str(exc.value)
    assert "z_level" in message
    assert "270" in message


def test_a_stack_page_that_disagrees_about_its_channel_is_refused(tmp_path):
    """PageName vs the JSON channel: two answers for one plane is a refusal, not a precedence rule."""
    root = tmp_path / "acq"
    (root / "0").mkdir(parents=True)
    meta = {"z_level": 0, "channel": "Fluorescence_488_nm_-_Penta", "region_id": "B2", "fov": 0,
            "x_mm": 1.0, "y_mm": 2.0}
    with tifffile.TiffWriter(root / "0" / "B2_0000_stack.tiff", append=True) as w:
        w.write(np.zeros((4, 4), np.uint16), metadata=meta, description=json.dumps(meta),
                extratags=[(285, "s", 0, "Fluorescence_638_nm_-_Penta", False)])
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    assert "PageName" in str(exc.value)


def test_two_stack_pages_claiming_the_same_plane_are_refused(tmp_path):
    """A duplicated (z, channel) makes one page unreachable — refuse rather than pick the last."""
    root = tmp_path / "acq"
    (root / "0").mkdir(parents=True)
    meta = {"z_level": 0, "channel": "Fluorescence_488_nm_-_Penta", "region_id": "B2", "fov": 0,
            "x_mm": 1.0, "y_mm": 2.0}
    for _ in range(2):
        with tifffile.TiffWriter(root / "0" / "B2_0000_stack.tiff", append=True) as w:
            w.write(np.zeros((4, 4), np.uint16), metadata=meta, description=json.dumps(meta),
                    extratags=[(285, "s", 0, meta["channel"], False)])
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    assert "two pages" in str(exc.value)


def test_a_6d_store_with_a_mislabelled_leading_axis_is_refused(tmp_path):
    """If the leading axis is not ``fov``, which axis the FOV lives on is unknowable — refuse."""
    root = writer_fixtures.build_zarr_6d(tmp_path / "acq")
    path = root / "zarr" / REGIONS[0] / "acquisition.zarr" / "zarr.json"
    payload = json.loads(path.read_text())
    axes = payload["attributes"]["ome"]["multiscales"][0]["axes"]
    axes[0] = {"name": "q", "type": "other"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as exc:
        open_reader(root).metadata
    assert "6D" in str(exc.value)
    assert "fov" in str(exc.value)


# --- the padding contract ----------------------------------------------------------------------

@pytest.mark.parametrize("padding", [0, 1, 3, 6])
def test_the_fov_padding_width_is_parsed_not_assumed(tmp_path, padding):
    """``FILE_ID_PADDING`` is a per-deployment setting (0 on the reference config, wider on rigs).

    A reader that hardcodes a width is blind on every site configured differently — the exact
    shape of the bug this ticket is about. Four widths, one reader, same answer.
    """
    root = writer_fixtures.build_multi_page_tiff(tmp_path / f"acq{padding}", padding=padding)
    reader = open_reader(root)
    assert reader.metadata["fovs_per_region"] == {r: list(FOVS) for r in REGIONS}
    np.testing.assert_array_equal(
        reader.read(REGIONS[0], FOVS[1], CHANNELS[0], 1),
        plane(REGIONS[0], FOVS[1], 1, CHANNELS[0]),
    )


# --- interface parity --------------------------------------------------------------------------

def test_plane_ref_addresses_a_real_page_for_every_tiff_writer(tmp_path):
    """``plane_ref`` is what the viewer hands to ndviewer; a wrong page index shows wrong pixels."""
    for builder in (writer_fixtures.build_individual_tiff, writer_fixtures.build_multi_page_tiff,
                    writer_fixtures.build_ome_tiff):
        root = builder(tmp_path / builder.__name__)
        reader = open_reader(root)
        for z in range(NZ):
            for channel in CHANNELS:
                path, page = reader.plane_ref(REGIONS[0], FOVS[0], channel, z)
                with tifffile.TiffFile(path) as tif:
                    np.testing.assert_array_equal(
                        tif.pages[page].asarray(), plane(REGIONS[0], FOVS[0], z, channel)
                    )


def test_both_writers_in_one_folder_warns_rather_than_ignoring_half(tmp_path):
    """Individual TIFFs and stacks side by side is two runs; serving one in silence is IMA-254."""
    root = writer_fixtures.build_individual_tiff(tmp_path / "acq")
    writer_fixtures.build_multi_page_tiff(root)
    with pytest.warns(UserWarning, match="BOTH"):
        open_reader(root)
