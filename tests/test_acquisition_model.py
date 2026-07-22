"""The typed acquisition model (Defect 1): validate the ACQUISITION, not just the command line.

``reader.metadata`` was a raw dict touched ~96 times across the package. A missing or wrong-typed
key surfaced as a blank render or an opaque failure several layers away — ``pixel_size_um`` being
the documented one (``_placement._require_pixel_size`` is a hand-written guard for exactly one
field, which is the shape of a problem that wants a schema).

These tests pin three things:

1. the schema is validated ONCE, at the reader boundary, and a malformed acquisition is refused
   THERE with the offending field named;
2. the model is still a Mapping, so the ~96 existing ``meta["..."]`` call sites keep working while
   they are migrated to attributes incrementally;
3. the genuinely-optional fields have LOUD accessors — asking for a missing ``pixel_size_um``
   raises naming the field, and never returns a substituted default.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from squidmip._acquisition import Acquisition, Channel


def _kw(**over):
    base = dict(
        regions=["A1", "A2"],
        fovs_per_region={"A1": [0, 1], "A2": [0]},
        fov_positions_um={("A1", 0): (0.0, 0.0), ("A1", 1): (100.0, 0.0)},
        channels=[{"name": "Fluorescence_488_nm_Ex", "display_name": "488",
                   "display_color": "#1FFF00", "ex": 488.0}],
        n_z=3,
        z_levels=[0, 1, 2],
        dz_um=1.5,
        pixel_size_um=0.325,
        wellplate_format="24 well plate",
        frame_shape=(2084, 3000),
        dtype=np.dtype("uint16"),
        n_t=2,
    )
    base.update(over)
    return base


# --- the schema itself ------------------------------------------------------------------

def test_builds_from_the_reader_dict():
    a = Acquisition(**_kw())
    assert a.pixel_size_um == 0.325
    assert a.frame_shape == (2084, 3000)
    assert a.dtype == np.dtype("uint16")
    assert a.channels[0].name == "Fluorescence_488_nm_Ex"
    assert isinstance(a.channels[0], Channel)


def test_channel_names_are_a_convenience_not_a_reimplementation():
    # `[c["name"] for c in meta["channels"]]` appears at a dozen call sites; one accessor.
    assert Acquisition(**_kw()).channel_names == ["Fluorescence_488_nm_Ex"]


def test_a_missing_required_field_is_refused_naming_it():
    kw = _kw()
    del kw["n_t"]
    with pytest.raises(ValidationError, match="n_t"):
        Acquisition(**kw)


def test_an_unknown_field_is_refused_so_a_typo_is_not_silently_stored():
    # extra="forbid": `meta["pixel_size"]` (no _um) must not quietly become a second, ignored key.
    with pytest.raises(ValidationError, match="pixel_size"):
        Acquisition(**_kw(pixel_size=0.325))


def test_a_wrong_typed_field_is_refused_at_construction():
    with pytest.raises(ValidationError):
        Acquisition(**_kw(n_z="three"))


def test_negative_dimensions_are_refused():
    with pytest.raises(ValidationError):
        Acquisition(**_kw(n_t=0))
    with pytest.raises(ValidationError):
        Acquisition(**_kw(n_z=0))


def test_a_nonpositive_pixel_size_is_refused_at_the_boundary_not_at_use():
    # 0.0 would divide-by-zero or collapse every FOV onto one spot, far from here.
    with pytest.raises(ValidationError):
        Acquisition(**_kw(pixel_size_um=0.0))
    with pytest.raises(ValidationError):
        Acquisition(**_kw(pixel_size_um=-1.0))


def test_fovs_per_region_must_cover_every_region():
    # A region with no FOV entry is a hole that renders as a blank well, not as an error.
    with pytest.raises(ValidationError, match="A2"):
        Acquisition(**_kw(fovs_per_region={"A1": [0, 1]}))


def test_z_levels_must_agree_with_n_z():
    with pytest.raises(ValidationError, match="z_levels"):
        Acquisition(**_kw(z_levels=[0, 1]))


# --- optional fields fail LOUD at the point of use, never by default --------------------

def test_pixel_size_may_be_absent_but_asking_for_it_raises_naming_the_field():
    a = Acquisition(**_kw(pixel_size_um=None))
    assert a.pixel_size_um is None                      # modelled Optional, honestly
    with pytest.raises(ValueError, match="pixel_size_um"):
        a.require_pixel_size_um()                        # but LOUD at the point of use


def test_require_pixel_size_returns_the_value_when_present():
    assert Acquisition(**_kw()).require_pixel_size_um() == 0.325


def test_dz_may_be_absent_but_asking_for_it_raises_naming_the_field():
    a = Acquisition(**_kw(dz_um=None))
    with pytest.raises(ValueError, match="dz_um"):
        a.require_dz_um()
    assert Acquisition(**_kw()).require_dz_um() == 1.5


def test_channel_index_refuses_an_unknown_channel_rather_than_returning_zero():
    # The Defect-2 shape: `names.index(x) if x in names else 0` is a silent wrong answer.
    a = Acquisition(**_kw())
    assert a.channel_index("Fluorescence_488_nm_Ex") == 0
    with pytest.raises(KeyError, match="Fluorescence_638_nm_Ex"):
        a.channel_index("Fluorescence_638_nm_Ex")


# --- still a Mapping, so the ~96 existing call sites migrate incrementally --------------

def test_subscript_access_still_works_for_unmigrated_call_sites():
    a = Acquisition(**_kw())
    assert a["pixel_size_um"] == 0.325
    assert a["frame_shape"] == (2084, 3000)
    assert a["channels"][0]["name"] == "Fluorescence_488_nm_Ex"   # channels stay dict-like too


def test_get_and_in_and_keys_behave_like_the_old_dict():
    a = Acquisition(**_kw())
    assert a.get("dz_um") == 1.5
    assert a.get("nope", "fallback") == "fallback"
    assert "pixel_size_um" in a
    assert "nope" not in a
    assert set(a.keys()) >= {"regions", "channels", "n_z", "n_t", "dtype", "frame_shape"}
    assert dict(a)["n_t"] == 2
    # `for k in meta` must yield KEYS. BaseModel.__iter__ yields (key, value) PAIRS, which
    # `dict()` happens to accept — so a test that only checks `dict(a)` passes either way and
    # proves nothing. Pin the iteration itself.
    assert all(isinstance(k, str) for k in a), f"iterating yielded non-keys: {list(a)[:2]}"
    assert list(a) == list(a.keys())


def test_an_unknown_key_raises_keyerror_not_none():
    # The old dict did this too; keep it, so a typo is never a silent None.
    with pytest.raises(KeyError):
        Acquisition(**_kw())["pixel_size"]
