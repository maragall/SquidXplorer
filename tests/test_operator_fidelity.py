"""Output-fidelity tests: SquidXplorer's operators vs the standalone repos they wrap.

Given identical synthetic input, SquidXplorer must produce byte-identical output to calling the
standalone repo directly — or, where it deliberately differs, the divergence is pinned exactly.
Synthetic in-memory numpy only; nothing is read from or written to disk. CPU forced (`gpu=False`)
wherever a backend choice exists, so comparisons are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

# Vs the maragall standalone repos, absent from the clean-room CI install; skip there.
pytest.importorskip("petakit")
pytest.importorskip("tilefusion")
pytest.importorskip("bgsub")


# 1. bgsub (maragall/background_subtraction)
#
# squidxplorer._background.estimate_background(method="sep") calls bgsub.core._run_sep verbatim.
# THE DIVERGENCE: bgsub's CLI worker writes clip(fg).astype(dtype) — TRUNCATE — while
# squidxplorer's cast_like does clip(rint(fg)).astype(dtype) — ROUND. Same clip, different
# rounding, by design: rounding avoids truncation's systematic half-count dimming and keeps the
# float layer invertible.


def _synthetic_plane_f32():
    """128x128 float32: smooth gradient background + three bright Gaussian blobs."""
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

    assert bg_squid.dtype == np.float32
    assert np.array_equal(bg_squid, bg_standalone)


def test_bgsub_float_subtraction_is_the_unclipped_standalone_foreground():
    """On a float plane, squidxplorer == bgsub._run_sep foreground, unclipped."""
    from squidxplorer._background import subtract_background, BackgroundParams
    from bgsub.core import _run_sep

    plane = _synthetic_plane_f32()
    R = 16

    sub_squid = subtract_background(plane, BackgroundParams(method="sep", radius_px=R))
    fg_standalone, _ = _run_sep(plane, R)

    assert sub_squid.dtype == np.float32
    assert np.array_equal(sub_squid, fg_standalone)


def test_bgsub_integer_subtraction_diverges_from_cli_by_round_vs_truncate():
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

    assert np.array_equal(sub_squid, squid_expected)
    assert not np.array_equal(sub_squid, cli_worker)
    # Bounded: rounding vs truncation never disagrees by more than one count.
    assert np.max(np.abs(sub_squid.astype(np.int32) - cli_worker.astype(np.int32))) <= 1
    assert np.any(sub_squid != cli_worker)


# 2. flatfield (maragall/stitcher — tilefusion.flatfield)


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


# 3. decon (maragall/deconvolution — petakit)
#
# squidxplorer._decon builds the in-focus plane of petakit's vectorial PSF (make_psf_2d) and calls
# petakit.deconvolve(method="rl"), not the "omw" default.


def _optics():
    from squidxplorer._decon import OpticsParams

    # Tiny nz keeps generate_psf cheap; real optics values (NA 0.3, 10x, 0.752 um/px).
    return OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=0.752, dz_um=1.5, nz=3)


def test_decon_in_focus_psf_is_byte_identical_to_petakit_generate_psf():
    """make_psf_2d == in-focus plane of petakit.generate_psf, recomputed independently here."""
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
    """Pins the wiring (method='rl', same PSF, same iterations) and the numeric result."""
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


# 4. stitch (maragall/stitcher — tilefusion.registration + optimization)


def _two_fov_overlap(inject_dx=3):
    """Two 128x128 tiles from one texture; stage says +100px, content is actually +100+inject_dx,
    so registration must recover a residual of inject_dx in x."""
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

    # The identical raw tilefusion pipeline, in TileFusion.run()'s call order.
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
    """A +3px injected content shift is recovered to sub-pixel via phase-correlation upsampling."""
    from squidxplorer._stitch import solve_offsets_px

    tiles, positions, psize, tshape, inject_dx = _two_fov_overlap(inject_dx=3)
    offsets = solve_offsets_px(
        tiles, positions, psize, tshape, registration_channel=0, max_workers=2
    )
    recovered_dx = offsets[1][1] - offsets[0][1]
    assert abs(recovered_dx - inject_dx) < 0.5
