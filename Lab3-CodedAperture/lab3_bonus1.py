"""
MECH5720 Lab 3 -- Bonus 1: Angled Aperture (Levin mask rotated 45 degrees).

Compares the upright Levin coded aperture (Part 2/4 captures, furthest
distance marking, 1300mm) against a second Levin aperture physically
rotated 45 degrees before capture (data/lab3/bonus/), at the same
furthest distance marking, same focus. Three comparisons, matching the
brief:

1. Design masks: the upright 13x13 Levin code vs. the same code rotated
   45 degrees "over the same domain" (scipy.ndimage.rotate with
   reshape=False keeps the canvas size fixed, so both sit in an
   identical pixel grid and are directly comparable) -- plus each
   design's 2D FFT magnitude, since the rotation's effect on spatial
   frequency content is exactly what the "sensor sampling rate" /
   forward-model discussion question below is about.
2. Captured PSFs: the upright and rotated masks' actual PSF captures at
   1300mm, centred and cropped the same way as every other Part
   (reusing extract_psf from lab3_6.py -- no new PSF-extraction logic).
3. Deconvolution: each mask's natural-scene capture at 1300mm, Wiener-
   deconvolved with its own captured PSF (reusing psf2otf/
   wiener_deconvolve/laplacian_variance/pick_K_by_curvature from
   lab3_8.py -- same K-selection method as Part 8, just run
   non-interactively here since this is a single one-off comparison
   rather than a full per-mask sweep), side by side.

NOTE on data: data/lab3/bonus/ has natural_300.dng AND
natural_back_300.dng -- inspecting both shows they are near-identical
repeat captures of the same scene (a retake), and only
natural_back_300 has a matching .png preview saved, so that is treated
as the final capture and used here; natural_300.dng is left alone.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

sys.path.insert(0, os.path.dirname(__file__))
from lab3_6 import (
    DATA_ROOT, DARK_DIR,
    compute_fixed_pattern_noise, read_raw_bayer, debayer_green, extract_psf,
    build_design_canvas, crop_centred,
)
from lab3_8 import (
    psf2otf, wiener_deconvolve, laplacian_variance, pick_K_by_curvature,
    crop_centre_fraction, load_green,
)

# ======================================================================
# Configuration
# ======================================================================
OUTDIR = "out/lab3_bonus1"

UPRIGHT_PSF_PATH = os.path.join(DATA_ROOT, "levin", "part2", "back_300.dng")
UPRIGHT_SCENE_PATH = os.path.join(DATA_ROOT, "levin", "part4", "back_300.dng")
ROTATED_PSF_PATH = os.path.join(DATA_ROOT, "bonus", "back_300.dng")
ROTATED_SCENE_PATH = os.path.join(DATA_ROOT, "bonus", "natural_back_300.dng")

PSF_CROP_SIZE = 128       # matches lab3_6.py's own default -- comfortably contains the blob
ROTATION_DEG = 45.0
FOURIER_PAD_SIZE = 256
FOURIER_DB_FLOOR = -60.0

N_K = 50
K_VALUES = np.logspace(-6, np.log10(5e-1), N_K)   # brief: sweep 1e-6 to 5e-1 (Part 8 convention)
CROP_FRACTION = 0.6

# The sharpness-proxy sweep is monotonically decreasing here (no L-curve
# corner -- see lab3_8.py's own note that this metric can't localise a
# "good" K on its own), so the auto-pick lands at the sweep's smallest K,
# which is visibly pure noise (see K_contact_sheet_*.png). K=2.36e-3 was
# instead picked by eye from that contact sheet: it is the largest K
# (most noise-suppressed) that still keeps the rigging lines and the
# target's sawtooth edge sharp in BOTH captures -- larger K (e.g. 3.4e-2)
# visibly starts smoothing that detail away. The same K is used for both
# masks deliberately, so the upright/rotated comparison below isolates
# the aperture's effect rather than a difference in regularisation.
FINAL_K = 2.36e-3


# ======================================================================
# Design mask: upright vs. 45 degree rotated, over the same domain
# ======================================================================
def build_rotated_design(mask_name="levin", angle_deg=ROTATION_DEG):
    """Upright design canvas and its copy rotated angle_deg, both on the
    identical (same-size) supersampled pixel grid -- reshape=False is
    what keeps the domain fixed per the brief's hint. Bilinear
    interpolation (order=1) anti-aliases the new diagonal edges instead
    of leaving jagged binary pixels, which is also physically honest:
    a hard-edged mask re-sampled onto a grid it isn't aligned with
    genuinely does produce partial-coverage (grey) edge pixels."""
    upright = build_design_canvas(mask_name)
    rotated = rotate(upright, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    return upright, np.clip(rotated, 0.0, 1.0)


def fourier_magnitude_db(img, pad_size=FOURIER_PAD_SIZE, db_floor=FOURIER_DB_FLOOR):
    n = max(pad_size, img.shape[0], img.shape[1])
    padded = np.zeros((n, n), dtype=img.dtype)
    r0, c0 = (n - img.shape[0]) // 2, (n - img.shape[1]) // 2
    padded[r0:r0 + img.shape[0], c0:c0 + img.shape[1]] = img
    mag = np.abs(np.fft.fftshift(np.fft.fft2(padded)))
    mag /= mag.max()
    floor_lin = 10.0 ** (db_floor / 20.0)
    return 20.0 * np.log10(np.maximum(mag, floor_lin))


# ======================================================================
# K selection for a single scene/PSF pair (non-interactive: L-curve
# corner only -- see lab3_8.py's pick_K_by_curvature docstring for why
# this, not the raw argmax, is used).
# ======================================================================
def choose_K(scene, psf, K_values=K_VALUES):
    metric = np.array([laplacian_variance(wiener_deconvolve(scene, psf, K)) for K in K_values])
    return pick_K_by_curvature(K_values, metric), metric


def save_single(img, path, cmap="gray", vmin=None, vmax=None):
    """One image, no title/axes/margin -- for the report's individual
    dropzone slots (as opposed to the multi-panel diagnostic figures,
    which stay separate for this script's own sanity-checking)."""
    fig, ax = plt.subplots(figsize=(5, 5 * img.shape[0] / img.shape[1]))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


CONTACT_SHEET_K_FRACS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)  # positions along K_VALUES (log-spaced)


def save_K_contact_sheet(name, scene, psf, K_values=K_VALUES, fracs=CONTACT_SHEET_K_FRACS):
    """Grid of reconstructions across a handful of K values spanning the
    full sweep, for visual K selection in place of an interactive slider
    (this script runs non-interactively) -- the brief's own hint to
    "sanity-check the picked K's image by eye" made literal without a
    live display."""
    idxs = [int(round(f * (len(K_values) - 1))) for f in fracs]
    fig, axes = plt.subplots(1, len(idxs), figsize=(3.2 * len(idxs), 3.6))
    for ax, i in zip(axes, idxs):
        recon = crop_centre_fraction(wiener_deconvolve(scene, psf, K_values[i]), CROP_FRACTION)
        ax.imshow(recon, cmap="gray")
        ax.set_title("K=%.2e" % K_values[i], fontsize=9)
        ax.axis("off")
    fig.suptitle("Bonus 1: K contact sheet -- %s" % name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "K_contact_sheet_%s.png" % name), dpi=130)


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    # ------------------------------------------------------------------
    # 1. Design masks: upright vs. rotated, plus their 2D FFT magnitude
    # ------------------------------------------------------------------
    design_upright, design_rotated = build_rotated_design()
    fig_d, axes_d = plt.subplots(2, 2, figsize=(8, 8))
    axes_d[0, 0].imshow(design_upright, cmap="gray", vmin=0, vmax=1)
    axes_d[0, 0].set_title("Levin design -- upright")
    axes_d[0, 1].imshow(design_rotated, cmap="gray", vmin=0, vmax=1)
    axes_d[0, 1].set_title("Levin design -- rotated 45 deg")
    im2 = axes_d[1, 0].imshow(fourier_magnitude_db(design_upright), cmap="viridis",
                               vmin=FOURIER_DB_FLOOR, vmax=0.0)
    axes_d[1, 0].set_title("FFT magnitude (dB) -- upright")
    axes_d[1, 1].imshow(fourier_magnitude_db(design_rotated), cmap="viridis",
                         vmin=FOURIER_DB_FLOOR, vmax=0.0)
    axes_d[1, 1].set_title("FFT magnitude (dB) -- rotated 45 deg")
    for ax in axes_d.ravel():
        ax.axis("off")
    fig_d.colorbar(im2, ax=axes_d[1, :], label="dB relative to DC", shrink=0.8)
    fig_d.suptitle("Bonus 1: Levin design mask, upright vs. 45 deg rotated")
    fig_d.subplots_adjust(hspace=0.35, wspace=0.4)
    fig_d.savefig(os.path.join(OUTDIR, "design_upright_vs_rotated.png"), dpi=150)

    # Report dropzone pair: "Levin -- upright design" / "Levin -- 45deg rotated design"
    save_single(design_upright, os.path.join(OUTDIR, "design_upright.png"), vmin=0, vmax=1)
    save_single(design_rotated, os.path.join(OUTDIR, "design_rotated.png"), vmin=0, vmax=1)

    # ------------------------------------------------------------------
    # 2. Captured PSFs: upright vs. rotated, at 1300mm
    # ------------------------------------------------------------------
    psf_upright, (cy_u, cx_u), (h_u, w_u) = extract_psf(
        UPRIGHT_PSF_PATH, fpn, colors, black, white, crop_size=PSF_CROP_SIZE)
    psf_rotated, (cy_r, cx_r), (h_r, w_r) = extract_psf(
        ROTATED_PSF_PATH, fpn, colors, black, white, crop_size=PSF_CROP_SIZE)

    print("upright PSF  centroid=(row=%.1f, col=%.1f)  blob=%dx%d px" % (cy_u, cx_u, h_u, w_u))
    print("rotated PSF  centroid=(row=%.1f, col=%.1f)  blob=%dx%d px" % (cy_r, cx_r, h_r, w_r))

    vmax = max(psf_upright.max(), psf_rotated.max())
    fig_p, axes_p = plt.subplots(1, 2, figsize=(9, 4.5))
    axes_p[0].imshow(psf_upright, cmap="inferno", vmin=0, vmax=vmax)
    axes_p[0].set_title("Captured PSF -- upright\nblob=%dx%d px" % (h_u, w_u))
    axes_p[1].imshow(psf_rotated, cmap="inferno", vmin=0, vmax=vmax)
    axes_p[1].set_title("Captured PSF -- rotated 45 deg\nblob=%dx%d px" % (h_r, w_r))
    for ax in axes_p:
        ax.axis("off")
    fig_p.suptitle("Bonus 1: captured Levin PSF at 1300mm, upright vs. rotated")
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(OUTDIR, "psf_upright_vs_rotated.png"), dpi=150)

    # ------------------------------------------------------------------
    # 3. Deconvolution: each mask's own natural-scene capture with its
    #    own captured PSF, side by side.
    # ------------------------------------------------------------------
    scene_upright = load_green(UPRIGHT_SCENE_PATH, fpn, colors, black, white)
    scene_rotated = load_green(ROTATED_SCENE_PATH, fpn, colors, black, white)

    K_upright, metric_upright = choose_K(scene_upright, psf_upright)
    K_rotated, metric_rotated = choose_K(scene_rotated, psf_rotated)
    print("upright: L-curve auto-pick K=%.3e" % K_upright)
    print("rotated: L-curve auto-pick K=%.3e" % K_rotated)
    save_K_contact_sheet("upright", scene_upright, psf_upright)
    save_K_contact_sheet("rotated", scene_rotated, psf_rotated)

    fig_k, ax_k = plt.subplots(figsize=(7, 5))
    ax_k.loglog(K_VALUES, metric_upright, marker=".", label="upright")
    ax_k.loglog(K_VALUES, metric_rotated, marker=".", label="rotated 45 deg")
    ax_k.axvline(FINAL_K, color="k", linestyle="--", alpha=0.7, label="K used (picked by eye)")
    ax_k.set_xlabel("K")
    ax_k.set_ylabel("variance of Laplacian (sharpness proxy)")
    ax_k.set_title("Bonus 1: Wiener K sweep, natural scene at 1300mm\n"
                    "(metric is monotonic here -- K picked by eye, see K_contact_sheet_*.png)")
    ax_k.legend()
    ax_k.grid(True, which="both", alpha=0.3)
    fig_k.tight_layout()
    fig_k.savefig(os.path.join(OUTDIR, "K_sweep.png"), dpi=150)

    recon_upright = crop_centre_fraction(
        wiener_deconvolve(scene_upright, psf_upright, FINAL_K), CROP_FRACTION)
    recon_rotated = crop_centre_fraction(
        wiener_deconvolve(scene_rotated, psf_rotated, FINAL_K), CROP_FRACTION)

    fig_r, axes_r = plt.subplots(1, 2, figsize=(11, 5.5))
    axes_r[0].imshow(recon_upright, cmap="gray")
    axes_r[0].set_title("Deconvolved -- upright Levin\nK=%.2e" % FINAL_K)
    axes_r[1].imshow(recon_rotated, cmap="gray")
    axes_r[1].set_title("Deconvolved -- rotated 45 deg Levin\nK=%.2e" % FINAL_K)
    for ax in axes_r:
        ax.axis("off")
    fig_r.suptitle("Bonus 1: natural scene at 1300mm, deconvolved with each mask's own PSF")
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(OUTDIR, "deconv_upright_vs_rotated.png"), dpi=150)

    # Report dropzone pair: "Deconvolved -- upright Levin" / "Deconvolved -- rotated Levin"
    save_single(recon_upright, os.path.join(OUTDIR, "deconv_upright.png"))
    save_single(recon_rotated, os.path.join(OUTDIR, "deconv_rotated.png"))

    plt.show()
