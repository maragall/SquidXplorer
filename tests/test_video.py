"""The .mp4 export, asserted on pixels and files rather than on state.

The failure shape guarded against is a recording whose frames are all identical,
so every test reads pixels back — composited frames or frames decoded from the .mp4.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import open_reader
from squidxplorer._video import (
    DEFAULT_FPS,
    MissingEncoder,
    axis_length,
    can_record,
    default_axis,
    encoder_problem,
    record_region,
    region_movie_frames,
    write_mp4,
)

_REPO = Path(__file__).resolve().parent.parent


def _make_5d():
    """``tools/make_5d_fixture.py``, loaded by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_make_5d_fixture", _REPO / "tools" / "make_5d_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def five_d(tmp_path_factory):
    """A tiny 5-D acquisition: 1 region x 1 FOV x 3 z x 2 channels x 3 t, 64 px frames."""
    root = tmp_path_factory.mktemp("video") / "acq5d"
    _make_5d().build(root, ["A1"], n_fovs=1, nz=3, nt=3, size=64)
    reader = open_reader(root)
    return reader, reader.metadata, "A1", root


def _moving_frames(n=5, size=32):
    """*n* distinct RGB frames: a white square that moves. Every consecutive pair differs."""
    out = []
    for i in range(n):
        f = np.zeros((size, size, 3), np.uint8)
        f[4:12, 2 + 4 * i: 10 + 4 * i] = 255
        out.append(f)
    return out


def _decode(path):
    """Every frame of an .mp4, as a list of ``(H, W, 3)`` uint8 arrays."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(f) for f in reader], dict(reader.get_meta_data())
    finally:
        reader.close()


def _tree_digest(root: Path) -> str:
    """A hash over every file's path, size, mtime and bytes under *root*."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        st = p.stat()
        h.update(str(p.relative_to(root)).encode())
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
        h.update(p.read_bytes())
    return h.hexdigest()


needs_encoder = pytest.mark.skipif(
    encoder_problem() is not None,
    reason=f"no mp4 encoder on this machine: {encoder_problem()}")


# --- the gate: when is the feature offered, and on which axis ------------------------------------

def test_can_record_is_true_on_a_z_stack_with_one_timepoint():
    assert can_record({"n_t": 1, "z_levels": list(range(10))}) is True
    assert can_record({"n_t": 3, "z_levels": [0]}) is True
    assert can_record({"n_t": 1, "z_levels": [0]}) is False


def test_default_axis_prefers_time_and_falls_back_to_z():
    assert default_axis({"n_t": 3, "z_levels": list(range(10))}) == "t"
    assert default_axis({"n_t": 1, "z_levels": list(range(10))}) == "z"


def test_axis_length_counts_the_frames_each_axis_is_worth():
    meta = {"n_t": 3, "z_levels": [0, 1, 2, 3]}
    assert axis_length(meta, "t") == 3
    assert axis_length(meta, "z") == 4
    with pytest.raises(ValueError, match="axis must be"):
        axis_length(meta, "c")


# --- the frames themselves ----------------------------------------------------------------------

def test_consecutive_t_frames_are_different_pixels(five_d):
    """A stuck t index yields identical frames; the fixture's blob moves with t."""
    reader, meta, region, _root = five_d
    frames = list(region_movie_frames(reader, meta, region, axis="t"))

    assert len(frames) == meta["n_t"] == 3
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i].astype(int) - frames[i + 1].astype(int))
        assert diff.any(), f"t frames {i} and {i + 1} are pixel-identical — the axis is stuck"
        # the blob is a real feature and moves a real distance, not one stray pixel
        assert (diff.sum(axis=2) > 0).sum() > 0.05 * frames[i][:, :, 0].size, (
            f"t frames {i}/{i + 1} differ in only {(diff.sum(axis=2) > 0).sum()} pixels")


def test_consecutive_z_frames_are_different_pixels(five_d):
    """The fixture sweeps focus, so consecutive z frames must differ."""
    reader, meta, region, _root = five_d
    frames = list(region_movie_frames(reader, meta, region, axis="z"))

    assert len(frames) == len(meta["z_levels"]) == 3
    for i in range(len(frames) - 1):
        assert np.any(frames[i] != frames[i + 1]), (
            f"z frames {i} and {i + 1} are pixel-identical — the axis is stuck")


def test_contrast_is_latched_so_a_brightening_sequence_stays_brightening(five_d):
    """Per-frame percentiles would normalise the change away; one latched window keeps it."""
    reader, meta, region, _root = five_d

    class _Brightening:
        """The fixture's reader, with each timepoint scaled up."""

        metadata = meta

        def read(self, region_, fov, channel, z_level, time_point=0):
            plane = reader.read(region_, fov, channel, z_level, 0).astype(np.float32)
            return np.clip(plane * (1.0 + 0.6 * time_point), 0, 65535).astype(np.uint16)

    frames = list(region_movie_frames(_Brightening(), meta, region, axis="t"))
    means = [float(f.mean()) for f in frames]
    assert means == sorted(means) and means[-1] > means[0] * 1.05, (
        f"frame brightness {means} did not rise — contrast is being re-derived per frame, which "
        f"is how a movie of a changing sample comes out looking static")


def test_only_the_named_channels_are_composited(five_d):
    """A hidden channel is out of the view, so it must be out of the movie."""
    reader, meta, region, _root = five_d
    names = [c["name"] for c in meta["channels"]]
    assert len(names) == 2

    both = next(region_movie_frames(reader, meta, region, axis="t", channels=names))
    one = next(region_movie_frames(reader, meta, region, axis="t", channels=names[:1]))
    assert np.any(both != one), "dropping a channel changed nothing — `channels` is ignored"
    with pytest.raises(ValueError, match="every channel is hidden"):
        next(region_movie_frames(reader, meta, region, axis="t", channels=[]))


def test_a_region_without_stage_positions_refuses_instead_of_guessing(five_d):
    reader, meta, region, _root = five_d
    blind = dict(meta, fov_positions_um={})
    with pytest.raises(ValueError, match="no stage positions"):
        next(region_movie_frames(reader, blind, region, axis="t"))


def test_should_stop_is_polled_before_each_frame(five_d):
    reader, meta, region, _root = five_d
    seen = {"n": 0}

    def stop_after_one():
        seen["n"] += 1
        return seen["n"] > 1

    frames = list(region_movie_frames(reader, meta, region, axis="t",
                                      should_stop=stop_after_one))
    assert len(frames) == 1, f"cancel produced {len(frames)} frames, not 1"


# --- the file ------------------------------------------------------------------------------------

@needs_encoder
def test_the_mp4_decodes_back_to_the_frames_that_went_in(tmp_path):
    """Round trip: the decoded frames differ frame to frame the way the input did."""
    out = tmp_path / "moving.mp4"
    frames_in = _moving_frames(n=5, size=32)
    path, n = write_mp4(frames_in, out, fps=DEFAULT_FPS)

    assert Path(path).exists() and Path(path).stat().st_size > 0
    assert n == 5
    decoded, meta = _decode(path)
    assert len(decoded) == 5, f"encoded 5 frames, decoded {len(decoded)}"
    assert meta["fps"] == DEFAULT_FPS
    for i in range(len(decoded) - 1):
        assert np.any(decoded[i] != decoded[i + 1]), (
            f"decoded frames {i} and {i + 1} are identical — the file is one picture repeated")


@needs_encoder
def test_odd_sized_frames_are_padded_not_resized(tmp_path):
    """H.264 cannot take an odd dimension; padding keeps every input pixel in place."""
    out = tmp_path / "odd.mp4"
    frames_in = [f[:31, :29] for f in _moving_frames(n=3, size=32)]
    path, n = write_mp4(frames_in, out, fps=DEFAULT_FPS)

    assert n == 3
    decoded, _meta = _decode(path)
    assert decoded[0].shape[:2] == (32, 30), f"padded to {decoded[0].shape[:2]}, expected (32, 30)"
    assert np.array_equal(decoded[0][:31, :29].shape, frames_in[0].shape)


@needs_encoder
def test_no_frames_leaves_no_file_behind(tmp_path):
    out = tmp_path / "empty.mp4"
    with pytest.raises(ValueError, match="nothing to encode"):
        write_mp4(iter(()), out, fps=DEFAULT_FPS)
    assert not out.exists(), "an empty encode left a file behind"


@needs_encoder
def test_a_producer_that_raises_part_way_leaves_no_truncated_movie(tmp_path):
    """A playable-but-short .mp4 after a mid-export failure must be deleted."""
    out = tmp_path / "partial.mp4"

    def _dies_on_the_third():
        for i, frame in enumerate(_moving_frames(n=5)):
            if i == 3:
                raise OSError("a plane could not be read")
            yield frame

    with pytest.raises(OSError, match="a plane could not be read"):
        write_mp4(_dies_on_the_third(), out, fps=DEFAULT_FPS)
    assert not out.exists(), (
        "a failed export left a truncated movie behind, which plays and looks like the real thing")


def test_a_missing_encoder_refuses_by_name_and_writes_nothing(tmp_path, monkeypatch):
    import squidxplorer._video as V

    monkeypatch.setattr(V, "encoder_problem",
                        lambda: "imageio-ffmpeg is not installed (test). Install squidxplorer[video].")
    out = tmp_path / "refused.mp4"
    with pytest.raises(MissingEncoder, match="imageio-ffmpeg is not installed"):
        V.write_mp4(_moving_frames(n=2), out, fps=DEFAULT_FPS)
    assert not out.exists(), "a refused export still created a file"


@needs_encoder
def test_record_region_writes_a_playable_movie_of_the_t_axis(five_d, tmp_path):
    """End to end: read -> fuse -> composite -> encode -> decode."""
    reader, meta, region, _root = five_d
    out = tmp_path / f"{region}_t.mp4"
    seen = []
    path, n = record_region(reader, meta, region, out, axis="t", fps=DEFAULT_FPS,
                            on_frame=lambda d, total: seen.append((d, total)))

    assert n == 3 and seen == [(1, 3), (2, 3), (3, 3)]
    assert Path(path).stat().st_size > 0
    decoded, dmeta = _decode(path)
    assert len(decoded) == 3
    assert dmeta["duration"] == pytest.approx(3 / DEFAULT_FPS, abs=0.2)
    for i in range(len(decoded) - 1):
        assert np.any(decoded[i] != decoded[i + 1]), (
            f"decoded t frames {i}/{i + 1} are identical")


@needs_encoder
def test_recording_does_not_touch_the_acquisition(five_d, tmp_path):
    """The recorder reads planes and writes one .mp4 where the user pointed it."""
    reader, meta, region, root = five_d
    before = _tree_digest(root)
    record_region(reader, meta, region, tmp_path / "untouched.mp4", axis="z", fps=DEFAULT_FPS)
    assert _tree_digest(root) == before, "the export modified the acquisition folder"
    assert not list(root.rglob("*.mp4")), "the export wrote a movie into the acquisition folder"


# --- the two real acquisitions --------------------------------------------------------------------

@pytest.mark.integration
@needs_encoder
def test_real_time_series_records_moving_frames(tmp_path):
    """``~/Downloads/sim_5d_2x2_t3``: blob moves with t."""
    root = Path.home() / "Downloads" / "sim_5d_2x2_t3"
    if not root.is_dir():
        pytest.skip(f"the T-axis fixture is not present at {root} "
                    f"(make it: python tools/make_5d_fixture.py {root})")
    reader = open_reader(root)
    meta = reader.metadata
    assert default_axis(meta) == "t"
    path, n = record_region(reader, meta, "A1", tmp_path / "t.mp4", axis="t", fps=DEFAULT_FPS)
    decoded, _m = _decode(path)
    assert n == meta["n_t"] == 3 and len(decoded) == 3
    for i in range(len(decoded) - 1):
        assert np.any(decoded[i] != decoded[i + 1])


@pytest.mark.integration
@needs_encoder
def test_real_z_stack_records_a_focus_sweep(real_dataset, tmp_path):
    """The real 10x tissue acquisition: n_t = 1, 10 z planes."""
    reader = open_reader(real_dataset)
    meta = reader.metadata
    assert default_axis(meta) == "z" and can_record(meta)
    path, n = record_region(reader, meta, "manual0", tmp_path / "z.mp4", axis="z", fps=DEFAULT_FPS)
    decoded, _m = _decode(path)
    assert n == 10 and len(decoded) == 10
    for i in range(len(decoded) - 1):
        assert np.any(decoded[i] != decoded[i + 1])
    assert os.path.getsize(path) > 100_000
