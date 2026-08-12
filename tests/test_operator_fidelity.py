"""OUTPUT-FIDELITY tests: SquidXplorer's operators vs Julio's standalone repos.

The claim these tests defend is stronger than "the operator imports the upstream package".
It is: **given identical synthetic input, SquidXplorer produces byte-identical output to
calling the standalone repo directly** — or, where it deliberately differs, the divergence is
pinned exactly and justified in a comment.

Everything here runs on synthetic in-memory numpy only; no dataset is read, nothing is written
to disk, and no squidxplorer source is modified. CPU is forced everywhere a backend choice exists
(``gpu=False``) so the comparison is deterministic.

Per-operator verdicts (see each test for the exact assertion):

  bgsub      VERIFIED IDENTICAL for the estimate and the float LAYER; the integer-write path
             DIVERGES from the CLI by round-vs-truncate, by design (documented below).
  flatfield  VERIFIED IDENTICAL (profile byte-identical, corrected float plane byte-identical).
  decon      VERIFIED IDENTICAL (PSF byte-identical, RL output byte-identical on CPU).
  stitch     VERIFIED IDENTICAL (solved per-tile offsets byte-identical to the raw tilefusion
             registration pipeline).
"""

from __future__ import annotations

import numpy as np
import pytest

# Fidelity is vs the maragall standalone repos, which the clean-room CI install does not have.
# Skip the whole module when any is absent (CI stays green); runs in full on the dev machine.
pytest.importorskip("petakit")
pytest.importorskip("tilefusion")
pytest.importorskip("bgsub")


# ======================================================================================
# 1. bgsub  (maragall/background_subtraction) — HIGHEST PRIORITY
# ======================================================================================
#
# squidxplorer._background.estimate_background(method="sep") calls bgsub.core._run_sep verbatim and
# returns its background. subtract_background then does raw - background and casts back.
#
# THE DIVERGENCE, pinned precisely:
#   * bgsub.core._run_sep(img, R) -> (img_f32 - bg, bg), both float32, UNCLIPPED.
#   * bgsub.core._process_frame_worker (the CLI's on-disk path) writes
#         np.clip(fg, info.min, info.max).astype(dtype)      # clip, then TRUNCATE
#   * squidxplorer.subtract_background casts with cast_like:
#         np.clip(np.rint(fg), info.min, info.max).astype(dtype)   # clip, then ROUND
#
#   So on a FLOAT plane (the LAYER contract — non-destructive, nothing thrown away) squidxplorer is
#   byte-identical to _run_sep's foreground. On an INTEGER plane squidxplorer rounds where the CLI
#   truncates: same clip, different rounding. That is faithful-by-design, not a bug — rounding
#   avoids the half-count systematic dimming that truncation imposes on every pixel, and it is
#   exactly what makes the layer invertible (see _background.py's cast_like docstring).


def _synthetic_plane_f32():
    """A 128x128 float32 plane: smooth gradient background + three bright Gaussian blobs."""
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    plane = 100.0 + 0.5 * yy + 0.3 * xx
    for cy, cx in [(30, 40), (80, 90), (60, 20)]:
        plane = plane + 500.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
    return plane.astype(np.float32)


def test_bgsub_background_estimate_is_byte_identical_to_run_sep():
    from squidxplorer._background import estimate_background, BackgroundParams
    from bgsub.core import _run_sep

    plane = _synthetic_plane_f32()
    R = 16

    bg_squid = estimate_background(plane, BackgroundParams(method="sep", radius_px=R))
    _, bg_standalone = _run_sep(plane, R)

    # No tolerance: the SAME estimator on the SAME pixels must give the SAME background.
    assert bg_squid.dtype == np.float32
    assert np.array_equal(bg_squid, bg_standalone)


def test_bgsub_float_subtraction_is_the_unclipped_standalone_foreground():
    """The LAYER path. On a float plane squidxplorer == bgsub._run_sep foreground, byte-for-byte,
    UNCLIPPED — a faithful non-destructive layer (no photons discarded, fully invertible)."""
    from squidxplorer._background import subtract_background, BackgroundParams
    from bgsub.core import _run_sep

    plane = _synthetic_plane_f32()
    R = 16

    sub_squid = subtract_background(plane, BackgroundParams(method="sep", radius_px=R))
    fg_standalone, _ = _run_sep(plane, R)

    assert sub_squid.dtype == np.float32
    assert np.array_equal(sub_squid, fg_standalone)


def test_bgsub_integer_subtraction_diverges_from_cli_by_round_vs_truncate():
    """The on-disk/integer path. squidxplorer ROUNDS where the CLI worker TRUNCATES — same clip,
    different rounding. This test pins BOTH sides of that exact, justified divergence."""
    from squidxplorer._background import subtract_background, BackgroundParams
    from bgsub.core import _run_sep

    plane = _synthetic_plane_f32()
    imgu = np.clip(plane, 0, 65535).astype(np.uint16)
    R = 16

    sub_squid = subtract_background(imgu, BackgroundParams(method="sep", radius_px=R))
    fg_standalone, _ = _run_sep(imgu, R)

    info = np.iinfo(np.uint16)
    cli_worker = np.clip(fg_standalone, info.min, info.max).astype(np.uint16)   # truncate
    squid_expected = np.clip(np.rint(fg_standalone), info.min, info.max).astype(np.uint16)  # round

    # squidxplorer == round(clip(fg)); it is NOT the CLI's truncate(clip(fg)).
    assert np.array_equal(sub_squid, squid_expected)
    assert not np.array_equal(sub_squid, cli_worker)
    # The divergence is real and non-trivial (many pixels differ by exactly the rounding), but
    # bounded to at most one count — the two never disagree by more than 1.
    assert np.max(np.abs(sub_squid.astype(np.int32) - cli_worker.astype(np.int32))) <= 1
    assert np.any(sub_squid != cli_worker)


# ======================================================================================
# 2. flatfield  (maragall/stitcher — tilefusion.flatfield)
# ======================================================================================
#
# squidxplorer._flatfield.estimate_profile wraps tilefusion.flatfield.estimate_flatfield_channel and
# stores its output in a FlatfieldProfile (no re-normalisation). correct_flatfield applies
# (raw - dark)/gain with the SAME _MIN_GAIN guard as tilefusion.flatfield.apply_flatfield.


def _synthetic_tile_stack():
    rng = np.random.default_rng(0)
    yy = np.mgrid[0:64, 0:64][0].astype(np.float32)
    dome = 200.0 * np.exp(-((yy - 32) ** 2) / 500.0)
    return (rng.random((6, 64, 64), dtype=np.float32) * 1000.0 + dome).astype(np.float32)


def test_flatfield_profile_is_byte_identical_to_estimate_flatfield_channel():
    from squidxplorer._flatfield import estimate_profile
    from tilefusion.flatfield import estimate_flatfield_channel

    stack = _synthetic_tile_stack()

    prof = estimate_profile(stack)
    ff_standalone, df_standalone = estimate_flatfield_channel(
        np.asarray(stack, np.float32), use_darkfield=False
    )

    assert np.array_equal(prof.flatfield, ff_standalone)
    assert prof.darkfield is None and df_standalone is None


def test_flatfield_corrected_plane_matches_apply_flatfield():
    """Corrected float plane is byte-identical to tilefusion.flatfield.apply_flatfield on the
    same profile. Float in / float out involves no integer rounding, so np.array_equal holds;
    the tolerance-free equality is the strongest statement available here."""
    from squidxplorer._flatfield import estimate_profile, correct_flatfield
    from tilefusion.flatfield import estimate_flatfield_channel, apply_flatfield

    stack = _synthetic_tile_stack()
    prof = estimate_profile(stack)
    ff_standalone, _ = estimate_flatfield_channel(np.asarray(stack, np.float32), use_darkfield=False)

    rng = np.random.default_rng(1)
    plane = (rng.random((64, 64), dtype=np.float32) * 1000.0).astype(np.float32)

    corrected_squid = correct_flatfield(plane, prof)
    corrected_standalone = apply_flatfield(plane, ff_standalone, None)

    assert np.array_equal(corrected_squid, corrected_standalone)


# ======================================================================================
# 3. decon  (maragall/deconvolution — petakit)
# ======================================================================================
#
# squidxplorer._decon builds the in-focus plane of petakit's vectorial PSF (make_psf_2d) and calls
# petakit.deconvolve(method="rl") — NOT the "omw" default. Both the PSF and the RL result are
# checked against calling petakit directly. Runs end-to-end on CPU, so no xfail is needed.


def _optics():
    from squidxplorer._decon import OpticsParams

    # Tiny nz keeps generate_psf cheap; real optics values (NA 0.3, 10x, 0.752 um/px).
    return OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=0.752, dz_um=1.5, nz=3)


def test_decon_in_focus_psf_is_byte_identical_to_petakit_generate_psf():
    """make_psf_2d == in-focus plane of petakit.generate_psf, sized by petakit.compute_psf_size,
    renormalised to sum 1 — recomputed INDEPENDENTLY here, not read back from squidxplorer."""
    import petakit
    from squidxplorer._decon import make_psf_2d

    op = _optics()
    ni = op.immersion_index
    nz_psf, nxy_psf = petakit.compute_psf_size(
        op.nz, op.dxy_um, op.dz_um, wavelength=op.wavelength_um, na=op.na, ni=ni
    )
    psf3 = petakit.generate_psf(
        nz=nz_psf, nxy=nxy_psf, dxy=op.dxy_um, dz=op.dz_um,
        wavelength=op.wavelength_um, na=op.na, ni=ni,
    )
    centre = psf3[psf3.shape[0] // 2]
    indep = np.ascontiguousarray((centre / float(centre.sum()))[None, ...], dtype=np.float32)

    assert np.array_equal(indep, make_psf_2d(op))


def test_decon_plane_is_byte_identical_to_petakit_rl():
    """squidxplorer.deconvolve_plane == petakit.deconvolve(method="rl") with the same PSF/params.

    This pins the WIRING (method='rl', not the 'omw' default; same PSF; same iterations) AND the
    numeric result. On a float plane there is no integer cast, so the two are byte-identical."""
    import petakit
    from squidxplorer._decon import deconvolve_plane, make_psf_2d, METHOD

    assert METHOD == "rl"  # never inherit petakit's "omw" default (returns all-zero on this data)

    op = _optics()
    rng = np.random.default_rng(2)
    plane = (rng.random((48, 48), dtype=np.float32) * 500.0 + 100.0).astype(np.float32)

    out_squid = deconvolve_plane(plane, op, iterations=3, gpu=False)
    out_standalone = petakit.deconvolve(
        plane[None, ...].astype(np.float32), make_psf_2d(op),
        method="rl", iterations=3, gpu=False,
    )[0]

    assert np.array_equal(out_squid, out_standalone)


# ======================================================================================
# 4. stitch  (maragall/stitcher — tilefusion.registration + optimization)
# ======================================================================================
#
# squidxplorer._stitch.solve_offsets_px is exactly four tilefusion.registration calls followed by
# the pose-graph solve. We run the identical raw pipeline here and assert the per-tile offsets
# are byte-identical, AND that a KNOWN injected residual is recovered sub-pixel.


def _two_fov_overlap(inject_dx=3):
    """Two 128x128 tiles cropped from one smooth texture. Stage says the second tile is 100 px
    to the right; its CONTENT is actually at 100+inject_dx px, so registration must recover a
    residual of +inject_dx in x."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    tex = gaussian_filter(rng.random((200, 300), dtype=np.float32), 1.5)
    Y, X = 128, 128
    t0 = tex[0:Y, 0:X]
    t1 = tex[0:Y, 100 + inject_dx:100 + inject_dx + X]
    tiles = np.stack([t0, t1])[:, None, :, :].astype(np.float32)  # (n_tiles, C, Y, X)
    positions = [(0.0, 0.0), (0.0, 100.0)]                        # (y_um, x_um)
    return tiles, positions, (1.0, 1.0), (Y, X), inject_dx


def test_stitch_offsets_are_byte_identical_to_raw_tilefusion_pipeline():
    from squidxplorer._stitch import solve_offsets_px
    from tilefusion.registration import (
        find_adjacent_pairs, rotation_aware_max_shift, compute_pair_bounds,
        register_pairs_batched,
    )
    from tilefusion.optimization import _edges_from_pairwise_metrics, two_round_optimization

    tiles, positions, psize, tshape, _ = _two_fov_overlap()

    offsets_squid = solve_offsets_px(
        tiles, positions, psize, tshape, registration_channel=0, max_workers=2
    )

    # The identical raw tilefusion pipeline, in TileFusion.run()'s order.
    pairs = find_adjacent_pairs(positions, psize, tshape, min_overlap=15)
    max_shift = rotation_aware_max_shift(pairs)
    bounds = compute_pair_bounds(pairs, tshape)

    def read_region(i, y_slice, x_slice):
        return tiles[i][0][y_slice, x_slice]

    metrics = register_pairs_batched(bounds, read_region, (1, 1), 15, max_shift, 2)
    edges = _edges_from_pairwise_metrics(metrics)
    offsets_direct = two_round_optimization(edges, 2, [0], 0.5, 2.0, True)

    assert np.array_equal(offsets_squid, offsets_direct)


def test_stitch_recovers_the_injected_residual():
    """Sanity that the solve is doing real work: a +3 px injected content shift is recovered to
    sub-pixel (phase correlation upsamples), which is what 'the offsets match' is worth."""
    from squidxplorer._stitch import solve_offsets_px

    tiles, positions, psize, tshape, inject_dx = _two_fov_overlap(inject_dx=3)
    offsets = solve_offsets_px(
        tiles, positions, psize, tshape, registration_channel=0, max_workers=2
    )
    recovered_dx = offsets[1][1] - offsets[0][1]
    assert abs(recovered_dx - inject_dx) < 0.5
