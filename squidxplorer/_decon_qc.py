"""Richardson-Lucy semi-convergence QC: sweep iteration counts on one real FOV and put
a halo/core metric (and, for the CLI, an orthogonal montage) in front of a human.
Used by the GUI panel, tools/decon_qc.py and the tests — one implementation.
"""
from __future__ import annotations

import numpy as np

TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")

# The turn has to be INSIDE the sampled range or the recommendation is worthless.
DEFAULT_MAX_ITERATIONS = 8

# Lateral half-width RL actually runs on: a crop keeps 8 RL runs cheap, with margin so the
# FFT edges stay far from the structure.
DEFAULT_CROP_HALF = 128

# Half-width of what the montage SHOWS: the structure and its halo fill the panel.
DEFAULT_VIEW_HALF = 32


# --------------------------------------------------------------------------------------
# The QC metric
# --------------------------------------------------------------------------------------
def halo_core_ratio(volume, centre, dxy_um, dz_um, core_um, window_um):
    """Mean brightness of the halo shell divided by the mean brightness of the core sphere."""
    volume = np.asarray(volume, dtype=np.float64)
    zc, yc, xc = centre
    zz, yy, xx = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    r_um = np.sqrt(((zz - zc) * dz_um) ** 2
                   + ((yy - yc) * dxy_um) ** 2
                   + ((xx - xc) * dxy_um) ** 2)
    core = r_um <= core_um
    halo = (r_um <= window_um) & ~core
    if not core.any() or not halo.any():
        raise ValueError(
            f"core radius {core_um} um / window radius {window_um} um do not resolve into "
            f"voxels at dxy={dxy_um} um, dz={dz_um} um."
        )

    # A constant camera offset drags the ratio toward 1; remove a per-volume background floor.
    floor = float(np.percentile(volume, 10.0))
    signal = np.clip(volume - floor, 0.0, None)

    core_mean = float(signal[core].mean())
    if core_mean <= 0:
        raise ValueError(
            "the structure's core is at or below the background floor, so a halo/core "
            "ratio is undefined. Pick a different fov/channel."
        )
    return float(signal[halo].mean() / core_mean)


def recommend(ks, curve):
    """Turn a QC curve into ``(best_k, kind, message)``; kind is "turn", "still-falling" or "rising"."""
    index = int(np.argmin(curve))
    best = int(ks[index])
    if 0 < index < len(curve) - 1:
        return best, "turn", (
            f"RECOMMENDATION: {best} iterations - the curve falls and turns back up "
            f"INSIDE 1..{ks[-1]}, so this is a real semi-convergence minimum.")
    if index == len(curve) - 1:
        return best, "still-falling", (
            f"NO TURN in 1..{ks[-1]}: the halo is still shrinking at the last iteration "
            f"sampled. The minimum is {best} only because the sweep stopped there, so it "
            "is NOT a recommendation - re-run with a larger --iterations. Note the "
            "control result in halo_core_ratio(): the visible halo turns LATE, so "
            "'no visible turn yet' does not by itself mean more iterations are better.")
    return best, "rising", (
        f"NO TURN in 1..{ks[-1]}: the curve rises from the very first iteration, i.e. RL "
        "overshoots immediately on this structure. Use fewer iterations, or a fov with a "
        "better-isolated structure.")


def halo_verdict(history):
    """Read an iterative run's ``[(k, ratio), ...]`` history into ``(kind, sentence)``."""
    history = list(history)
    if not history:
        raise ValueError(
            "halo_verdict needs at least one (iterations, halo/core) pair; an empty history "
            "has no verdict to give and returning a neutral one would read as 'fine so far'.")
    k, ratio = history[-1]
    if len(history) == 1:
        return "first", (
            f"{k} iterations - halo/core {ratio:.3f}. Add one more and compare: the number "
            "should FALL while deconvolution is still concentrating light.")
    best_k, best_ratio = min(history, key=lambda kv: kv[1])
    if best_k == k:
        prev_ratio = history[-2][1]
        return "improving", (
            f"{k} iterations - halo/core {ratio:.3f}, down {prev_ratio - ratio:+.3f} from "
            f"{history[-2][0]}. Still tightening, so another iteration may still buy "
            "concentration.")
    return "worse", (
        f"{k} iterations - halo/core {ratio:.3f}, WORSE than {best_ratio:.3f} at {best_k}. "
        f"The disc is growing back: that is amplified noise wearing the shape of the PSF, "
        f"not more resolution. Go back to {best_k}.")


def qc_window_um(core_um, nz, dz_um, preferred=8.0):
    """Window radius: *preferred* core radii, but never deeper than the stack can hold axially."""
    return float(min(preferred * core_um, max((nz // 2) - 1, 1) * dz_um))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_stack(dataset, region, fov, channel):
    """Read ONE (region, fov, channel) z-stack as (Z, Y, X). Read-only, one file per z."""
    from squidxplorer import open_reader

    reader = open_reader(dataset)
    meta = reader.metadata
    if region is None:
        region = meta["regions"][0]
    if channel is None:
        channel = meta["channels"][0]["name"]
    planes = [reader.read(region, fov, channel, z) for z in meta["z_levels"]]
    return np.stack(planes), region, channel, meta


def brightest_structure(stack, dxy_um, dz_um, core_um, z_margin=0, xy_margin=0):
    """(z, y, x) of the brightest STRUCTURE (core-smoothed argmax), not the brightest pixel."""
    from scipy.ndimage import gaussian_filter

    sigma = (max(core_um / dz_um, 0.5), core_um / dxy_um, core_um / dxy_um)
    smoothed = gaussian_filter(stack.astype(np.float32), sigma)
    nz, ny, nx = smoothed.shape
    # exclude candidates too close to the stack faces / frame edge for the QC window to fit
    z0, z1 = min(z_margin, (nz - 1) // 2), max(nz - z_margin, (nz + 2) // 2)
    y1, x1 = max(ny - xy_margin, xy_margin + 1), max(nx - xy_margin, xy_margin + 1)
    allowed = np.zeros(smoothed.shape, dtype=bool)
    allowed[z0:z1, xy_margin:y1, xy_margin:x1] = True
    return np.unravel_index(
        int(np.argmax(np.where(allowed, smoothed, -np.inf))), smoothed.shape)


def crop_around(stack, centre, half):
    """Crop laterally to +-*half* px around *centre*, keeping every z plane."""
    _, ny, nx = stack.shape
    y0 = int(np.clip(centre[1] - half, 0, max(ny - 2 * half, 0)))
    x0 = int(np.clip(centre[2] - half, 0, max(nx - 2 * half, 0)))
    y1, x1 = min(y0 + 2 * half, ny), min(x0 + 2 * half, nx)
    return stack[:, y0:y1, x0:x1], (centre[0], centre[1] - y0, centre[2] - x0)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def orthogonal_slices(volume, centre, half=None):
    """The x-z and y-z sections through *centre*, both returned as (z, lateral)."""
    zc, yc, xc = centre
    xz, yz = volume[:, yc, :], volume[:, :, xc]
    if half:
        x0, x1 = max(xc - half, 0), min(xc + half, xz.shape[1])
        y0, y1 = max(yc - half, 0), min(yc + half, yz.shape[1])
        xz, yz = xz[:, x0:x1], yz[:, y0:y1]
    return xz, yz


def _display(panel, gamma=0.5, reference=None):
    """Background-subtract, normalise (against *reference* when given), apply a display gamma."""
    panel = np.asarray(panel, dtype=np.float64)
    ref = panel if reference is None else np.asarray(reference, dtype=np.float64)
    floor = float(np.percentile(ref, 10.0))
    peak = float(np.clip(ref - floor, 0.0, None).max())
    panel = np.clip(panel - floor, 0.0, None)
    if peak <= 0:
        return np.zeros_like(panel)
    return (panel / peak) ** gamma


# The GUI's turbo composite (qc_composite / composite_centre_at / turbo_rgb) was removed on
# 2026-08-25 (Julio: "The turbo colormap preview makes no sense. remove it"): the sweep's
# preview is a normal data layer in the view, under the channel's own colormap. The montage
# below stays: it is the CLI's (tools/decon_qc.py) offline report, not the GUI preview.


def write_montage(path, per_iteration, centre, dxy_um, dz_um, title, view_half=None):
    """rows = iterations, columns = [x-z, y-z], TURBO, iteration number labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(per_iteration)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 0.85 * n + 0.6), squeeze=False)
    aspect = dz_um / dxy_um          # z steps are 1.5 um, pixels 0.752 um: draw them square
    for row, (label, volume) in enumerate(per_iteration):
        xz, yz = orthogonal_slices(volume, centre, view_half)
        for col, (panel, name) in enumerate(((xz, "x-z"), (yz, "y-z"))):
            ax = axes[row][col]
            ax.imshow(_display(panel), cmap="turbo", vmin=0.0, vmax=1.0,
                      aspect=aspect, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_curve(path, iterations, values, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(iterations, values, marker="o", color="#1f77b4")
    if best is not None:
        ax.axvline(best, color="#d62728", linestyle="--",
                   label=f"argmin = {best} iterations")
        ax.legend(fontsize=8)
    ax.set_xlabel("RL iterations")
    ax.set_ylabel("energy outside the core / energy in window")
    ax.set_title("RL semi-convergence: down is sharpening, up is noise", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _lateral_sigma_px(psf):
    """Second-moment-equivalent lateral sigma of the in-focus PSF plane, in pixels."""
    plane = np.asarray(psf[psf.shape[0] // 2], dtype=np.float64)
    total = plane.sum()
    if total <= 0:
        return float("nan")
    yy, xx = np.ogrid[:plane.shape[0], :plane.shape[1]]
    cy = float((plane * yy).sum() / total)
    cx = float((plane * xx).sum() / total)
    var = float((plane * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total) / 2.0
    return float(np.sqrt(var))
