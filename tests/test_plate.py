"""Tests for the Plate abstraction, the sample_formats builder and the carrier-art registry (IMA-214).

Non-Qt on purpose (same reasoning as tests/test_plate_shape.py): the plate model is the thing the
mosaic/selection/loupe code will share, so its contract is pinned here where it always runs, not
behind ``pytest.importorskip("PyQt5")``.

UNITS: everything micrometres, every key ends ``_um`` (see _placement.py). sample_formats.csv is
millimetres and is converted exactly once, at the loader.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

from squidmip._plate import (
    CarrierArt,
    Plate,
    PlateBuildError,
    PlateGeometry,
    SlideCarrier,
    WellPlate,
    build_plate,
    carrier_art,
    format_from_pitch_um,
    load_sample_formats,
    measure_region_pitch_um,
    squid_images_dir,
)


# --------------------------------------------------------------------------- helpers

def _positions_um(pitch_x_um, pitch_y_um, regions=("A1", "A2", "B1", "B2"), n_fov=4):
    """{(region, fov): (x_um, y_um)} for a plate with the given pitch, 2x2 FOVs per well."""
    out = {}
    for region in regions:
        row = ord(region[0]) - 65
        col = int(region[1:]) - 1
        x0 = 45_000.0 + col * pitch_x_um
        y0 = 25_000.0 + row * pitch_y_um
        for f in range(n_fov):
            out[(region, f)] = (x0 + (f % 2) * 705.0, y0 + (f // 2) * 705.0)
    return out


def _meta(**kw):
    m = {
        "regions": ["A1", "A2", "B1", "B2"],
        "fovs_per_region": {r: [0, 1, 2, 3] for r in ["A1", "A2", "B1", "B2"]},
        "fov_positions_um": _positions_um(9000.0, 9000.0),
        "wellplate_format": None,
        "pixel_size_um": 0.3728,
    }
    m.update(kw)
    return m


# --------------------------------------------------------------------------- sample_formats loader

def test_load_sample_formats_returns_geometry_in_micrometres():
    formats = load_sample_formats()
    g = formats["96 well plate"]
    assert isinstance(g, PlateGeometry)
    assert (g.rows, g.cols) == (8, 12)
    # csv says well_spacing_mm=9.0, well_size_mm=6.21, a1 = (11.31, 10.75) mm
    assert g.pitch_x_um == pytest.approx(9000.0)
    assert g.pitch_y_um == pytest.approx(9000.0)
    assert g.cell_size_um == pytest.approx(6210.0)
    assert g.a1_x_um == pytest.approx(11310.0)
    assert g.a1_y_um == pytest.approx(10750.0)


def test_load_sample_formats_has_every_standard_format_and_no_mm_keys():
    formats = load_sample_formats()
    for name in ("glass slide", "6 well plate", "12 well plate", "24 well plate",
                 "96 well plate", "384 well plate", "1536 well plate"):
        assert name in formats, name
    # the units contract: no attribute may carry a bare mm value
    for g in formats.values():
        for field in ("pitch_x_um", "pitch_y_um", "cell_size_um", "a1_x_um", "a1_y_um"):
            assert hasattr(g, field)
        assert not any(f.endswith("_mm") for f in vars(g))


def test_load_sample_formats_reads_a_real_csv(tmp_path):
    csv = tmp_path / "sample_formats.csv"
    csv.write_text(
        "format,a1_x_mm,a1_y_mm,a1_x_pixel,a1_y_pixel,well_size_mm,well_spacing_mm,"
        "number_of_skip,rows,cols\n"
        "96,11.31,10.75,171,135,6.21,9.0,0,8,12\n"
    )
    g = load_sample_formats(csv)["96 well plate"]
    assert g.pitch_x_um == pytest.approx(9000.0)
    assert g.a1_x_px == 171


def test_load_sample_formats_missing_csv_falls_back_to_vendored(tmp_path):
    # degrade gracefully: a missing upstream checkout must not break plate layout
    formats = load_sample_formats(tmp_path / "nope.csv")
    assert formats["96 well plate"].pitch_x_um == pytest.approx(9000.0)


# --------------------------------------------------------------------------- Plate ABC / WellPlate

def test_plate_is_abstract():
    with pytest.raises(TypeError):
        Plate(PlateGeometry.vendored("96 well plate"))  # type: ignore[abstract]


def test_wellplate_cell_ids_are_row_major():
    p = WellPlate.from_format("6 well plate")
    assert p.rows == 2 and p.cols == 3
    assert p.cell_ids == ["A1", "A2", "A3", "B1", "B2", "B3"]


def test_wellplate_cell_index_roundtrips_including_double_letter_rows():
    p = WellPlate.from_format("1536 well plate")
    assert p.rows == 32 and p.cols == 48
    for cid in ("A1", "B12", "Z48", "AA1", "AF48"):
        r, c = p.cell_index(cid)
        assert p.cell_id(r, c) == cid


def test_wellplate_cell_centre_uses_a1_offset_and_pitch():
    p = WellPlate.from_format("96 well plate")
    assert p.cell_center_um("A1") == pytest.approx((11310.0, 10750.0))
    # B3 is 2 columns and 1 row from A1 at 9 mm pitch
    assert p.cell_center_um("B3") == pytest.approx((11310.0 + 18000.0, 10750.0 + 9000.0))


def test_wellplate_rejects_a_cell_outside_the_grid():
    p = WellPlate.from_format("6 well plate")
    with pytest.raises(KeyError):
        p.cell_center_um("C1")
    with pytest.raises(KeyError):
        p.cell_center_um("A9")


def test_wellplate_extent_um_spans_the_whole_grid():
    p = WellPlate.from_format("96 well plate")
    w, h = p.extent_um
    assert w == pytest.approx(11 * 9000.0 + 6210.0)
    assert h == pytest.approx(7 * 9000.0 + 6210.0)


def test_wellplate_384_and_96_differ_by_exactly_2x_pitch():
    # the whole reason declared-vs-measured matters: these two are a factor of 2 apart
    assert (WellPlate.from_format("96 well plate").pitch_x_um
            == pytest.approx(2 * WellPlate.from_format("384 well plate").pitch_x_um))


# --------------------------------------------------------------------------- SlideCarrier

def test_slide_carrier_is_a_plate_and_shares_the_cell_api():
    c = SlideCarrier.from_format("4 slide carrier")
    assert isinstance(c, Plate)
    assert (c.rows, c.cols) == (1, 4)
    assert len(c.cell_ids) == 4
    # same API as a WellPlate: index/centre/extent all work
    r, col = c.cell_index(c.cell_ids[2])
    assert (r, col) == (0, 2)
    assert c.cell_center_um(c.cell_ids[0])[0] < c.cell_center_um(c.cell_ids[3])[0]


def test_glass_slide_is_a_one_cell_carrier():
    c = SlideCarrier.from_format("glass slide")
    assert (c.rows, c.cols) == (1, 1)
    assert len(c.cell_ids) == 1


def test_slide_carrier_takes_freeform_region_ids_positionally():
    c = SlideCarrier.from_format("4 slide carrier", cell_ids=["manual0", "tissueA", "manual2"])
    assert c.cell_ids[:3] == ["manual0", "tissueA", "manual2"]
    assert c.cell_index("tissueA") == (0, 1)


def test_slide_carrier_refuses_more_regions_than_slots():
    with pytest.raises(PlateBuildError):
        SlideCarrier.from_format("glass slide", cell_ids=["a", "b"])


# --------------------------------------------------------------------------- pitch measurement

def test_measure_region_pitch_um_recovers_9mm():
    px, py = measure_region_pitch_um(_positions_um(9000.0, 9000.0), ["A1", "A2", "B1", "B2"])
    assert px == pytest.approx(9000.0)
    assert py == pytest.approx(9000.0)


def test_measure_region_pitch_um_handles_a_gap_in_column_indices():
    # A1 and A5 only: 4 columns apart, so pitch = dx / 4, not dx
    pos = _positions_um(4500.0, 4500.0, regions=("A1", "A5"))
    px, py = measure_region_pitch_um(pos, ["A1", "A5"])
    assert px == pytest.approx(4500.0)
    assert py is None                       # only one row -> y pitch unmeasurable


def test_measure_region_pitch_um_is_none_for_non_well_regions():
    pos = {("manual0", 0): (1.0, 2.0), ("manual1", 0): (3.0, 4.0)}
    assert measure_region_pitch_um(pos, ["manual0", "manual1"]) == (None, None)


def test_measure_region_pitch_um_is_none_without_coordinates():
    assert measure_region_pitch_um({}, ["A1", "A2"]) == (None, None)


def test_format_from_pitch_um_distinguishes_96_from_384():
    assert format_from_pitch_um(9000.0, 9000.0) == "96 well plate"
    assert format_from_pitch_um(4500.0, 4500.0) == "384 well plate"
    assert format_from_pitch_um(2250.0, 2250.0) == "1536 well plate"


def test_format_from_pitch_um_tolerates_small_stage_error():
    assert format_from_pitch_um(8980.0, 9020.0) == "96 well plate"


def test_format_from_pitch_um_rejects_a_pitch_matching_nothing():
    assert format_from_pitch_um(7000.0, 7000.0) is None


def test_format_from_pitch_um_rejects_disagreeing_axes():
    # x says 96, y says 384 -> not a plate we can name; refuse rather than pick one
    assert format_from_pitch_um(9000.0, 4500.0) is None


# --------------------------------------------------------------------------- the builder

def test_build_plate_uses_the_declared_format_when_geometry_agrees():
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no warning when there is no conflict
        p = build_plate(_meta(wellplate_format="96 well plate"))
    assert isinstance(p, WellPlate)
    assert p.format_name == "96 well plate"


def test_build_plate_measured_geometry_overrides_a_contradicting_declared_format():
    """THE live bug: ~/Downloads/synthetic_2x2_wellplate declares 384 but measures 9.000 mm.

    Trusting the declaration would draw carrier art at exactly 2x wrong scale (IMA-220).
    """
    meta = _meta(wellplate_format="384 well plate")     # positions are 9 mm = 96wp
    with pytest.warns(UserWarning, match="384 well plate"):
        p = build_plate(meta)
    assert p.format_name == "96 well plate"
    assert p.pitch_x_um == pytest.approx(9000.0)
    assert p.format_source == "measured"
    assert p.declared_format == "384 well plate"


def test_build_plate_warning_names_both_formats_and_the_measured_pitch():
    with pytest.warns(UserWarning) as rec:
        build_plate(_meta(wellplate_format="384 well plate"))
    msg = str(rec[0].message)
    assert "384 well plate" in msg and "96 well plate" in msg
    assert "9000" in msg or "9.0" in msg


def test_build_plate_falls_back_to_declared_when_pitch_is_unmeasurable():
    meta = _meta(wellplate_format="384 well plate", fov_positions_um={})
    p = build_plate(meta)
    assert p.format_name == "384 well plate"
    assert p.format_source == "declared"


def test_build_plate_ignores_a_measured_format_too_small_for_the_observed_wells():
    # wells run out to P24 (384-only) but the measured pitch reads 96wp: the measurement cannot
    # be right, because a 96wp has no P24. Keep the declaration, still warn.
    regions = ["A1", "A24", "P24"]
    meta = _meta(
        regions=regions,
        fovs_per_region={r: [0] for r in regions},
        fov_positions_um={("A1", 0): (0.0, 0.0),
                          ("A24", 0): (23 * 9000.0, 0.0),          # row A -> x pitch 9 mm
                          ("P24", 0): (23 * 9000.0, 15 * 9000.0)}, # col 24 -> y pitch 9 mm
        wellplate_format="384 well plate",
    )
    with pytest.warns(UserWarning):
        p = build_plate(meta)
    assert p.format_name == "384 well plate"


def test_build_plate_infers_when_nothing_is_declared():
    meta = _meta(wellplate_format=None)
    p = build_plate(meta)
    # measured 9 mm pitch beats the span rule's under-read (2x2 -> 6wp)
    assert p.format_name == "96 well plate"
    assert p.format_source == "measured"


def test_build_plate_span_inference_when_no_coordinates_and_no_declaration():
    meta = _meta(wellplate_format=None, fov_positions_um={})
    p = build_plate(meta)
    assert p.format_name == "6 well plate"          # _plate_shape's SPAN+SNAP
    assert p.format_source == "inferred"


def test_build_plate_override_wins_over_everything_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = build_plate(_meta(wellplate_format="384 well plate"), override="24 well plate")
    assert p.format_name == "24 well plate"
    assert p.format_source == "override"


def test_build_plate_returns_a_slide_carrier_for_freeform_regions():
    regions = ["manual0", "manual1"]
    meta = _meta(regions=regions, fovs_per_region={r: [0] for r in regions},
                 fov_positions_um={}, wellplate_format=None)
    p = build_plate(meta)
    assert isinstance(p, SlideCarrier)
    assert p.cell_ids[:2] == ["manual0", "manual1"]


def test_build_plate_declared_glass_slide_with_four_regions_is_a_4_slide_carrier():
    regions = ["s0", "s1", "s2", "s3"]
    meta = _meta(regions=regions, fovs_per_region={r: [0] for r in regions},
                 fov_positions_um={}, wellplate_format="glass slide")
    p = build_plate(meta)
    assert isinstance(p, SlideCarrier)
    assert p.cols == 4


def test_build_plate_populates_cells_from_the_acquisition():
    p = build_plate(_meta(wellplate_format="96 well plate"))
    assert p.occupied_cells == ["A1", "A2", "B1", "B2"]
    assert p.fovs("A1") == [0, 1, 2, 3]
    assert p.fovs("H12") == []              # a real but unacquired well
    assert p.is_occupied("A1") and not p.is_occupied("H12")


def test_build_plate_occupied_cells_are_in_plate_row_major_order():
    regions = ["B10", "AA1", "B2", "A1"]
    meta = _meta(regions=regions, fovs_per_region={r: [0] for r in regions},
                 fov_positions_um={}, wellplate_format="1536 well plate")
    p = build_plate(meta)
    assert p.occupied_cells == ["A1", "B2", "B10", "AA1"]


def test_build_plate_rejects_a_region_outside_the_resolved_grid():
    regions = ["A1", "Z99"]
    meta = _meta(regions=regions, fovs_per_region={r: [0] for r in regions},
                 fov_positions_um={}, wellplate_format="96 well plate")
    with pytest.raises(PlateBuildError, match="Z99"):
        build_plate(meta)


# --------------------------------------------------------------------------- carrier art registry

def test_carrier_art_returns_none_when_the_squid_images_dir_is_absent(tmp_path):
    assert carrier_art("96 well plate", images_dir=tmp_path) is None


def test_carrier_art_never_invents_a_filename(tmp_path):
    (tmp_path / "96 well plate_1509x1010.png").write_bytes(b"")
    assert carrier_art("96 well plate", images_dir=tmp_path).path.exists()
    assert carrier_art("384 well plate", images_dir=tmp_path) is None   # not on disk -> None


def test_carrier_art_scale_is_micrometres_per_pixel(tmp_path):
    (tmp_path / "96 well plate_1509x1010.png").write_bytes(b"")
    art = carrier_art("96 well plate", images_dir=tmp_path)
    assert isinstance(art, CarrierArt)
    # Squid's NavigationViewer: mm_per_pixel = 0.084665 for every plate
    assert art.um_per_px == pytest.approx(84.665)


def test_carrier_art_slide_carrier_has_its_own_scale(tmp_path):
    (tmp_path / "slide carrier_828x662.png").write_bytes(b"")
    (tmp_path / "4 slide carrier_1509x1010.png").write_bytes(b"")
    assert carrier_art("glass slide", images_dir=tmp_path).um_per_px == pytest.approx(145.3)
    assert carrier_art("4 slide carrier", images_dir=tmp_path).um_per_px == pytest.approx(84.665)


def test_carrier_art_origin_places_a1_at_its_recorded_pixel(tmp_path):
    (tmp_path / "96 well plate_1509x1010.png").write_bytes(b"")
    art = carrier_art("96 well plate", images_dir=tmp_path)
    # Squid: origin_px = a1_pixel - a1_mm / mm_per_pixel; so a1 must map back to (171, 135)
    x_px, y_px = art.um_to_px(11310.0, 10750.0)
    assert x_px == pytest.approx(171, abs=1)
    assert y_px == pytest.approx(135, abs=1)


def test_carrier_art_is_reachable_from_a_plate(tmp_path):
    (tmp_path / "96 well plate_1509x1010.png").write_bytes(b"")
    p = WellPlate.from_format("96 well plate")
    assert p.art(images_dir=tmp_path) is not None
    assert p.art(images_dir=tmp_path / "gone") is None


def test_squid_images_dir_is_optional_and_never_raises():
    d = squid_images_dir()
    assert d is None or isinstance(d, Path)


# --------------------------------------------------------------------------- real dataset

_SYNTH = Path.home() / "Downloads" / "synthetic_2x2_wellplate"


@pytest.mark.skipif(not _SYNTH.is_dir(), reason="synthetic_2x2_wellplate not present")
def test_real_synthetic_dataset_resolves_to_96_not_the_declared_384():
    from squidmip.reader import open_reader

    md = open_reader(_SYNTH).metadata
    assert md["wellplate_format"] == "384 well plate"        # what the yaml says
    with pytest.warns(UserWarning):
        p = build_plate(md)
    assert p.format_name == "96 well plate"                  # what the stage says
    assert p.pitch_x_um == pytest.approx(9000.0, abs=1.0)
