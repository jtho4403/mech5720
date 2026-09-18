"""
MECH5720 Lab 3 -- Part 9: deconvolving a scene with the PSF from each
distance marker.

The scene (data/lab3_2/part9/scene/img_2.dng) was captured at one fixed
distance, but Part 9's own psf/ folder has a fresh PSF capture at all
five distance markings (700 / 850 / 1000 / 1150 / 1300 mm), same
point-source procedure as Part 2/6. Wiener-deconvolving the SAME scene
with each of these five PSFs shows how sensitive the reconstruction is
to using the wrong depth's PSF -- only the marker matching the scene's
true capture distance should come out sharp; the rest are deconvolved
with a mismatched blur kernel.

NOTE on K: each depth gets its OWN K, tuned by eye with its own slider
(lab3_8.pick_K_by_slider, reused directly) on a downsampled preview --
not one shared K across all five. A single shared K was tried first,
but each depth's PSF has a very different energy/shape, so one K can't
be simultaneously well-regularised for all of them.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from lab3_6_new import (
    compute_fixed_pattern_noise,
    read_raw_bayer,
    debayer_green,
    analyse_psf_blob,
    crop_centred,
    SATURATION_LEVEL,
)
from lab3_8 import (
    load_green,
    wiener_deconvolve,
    build_preview_stack,
    pick_K_by_slider,
    K_VALUES,
)

# ======================================================================
# Configuration
# ======================================================================
DARK_DIR = "data/lab3_2/part0"          # dark frames are shared across all parts
PART9_ROOT = "data/lab3_2/part9"
PSF_ROOT = os.path.join(PART9_ROOT, "psf")
SCENE_PATH = os.path.join(PART9_ROOT, "scene", "img_2.dng")

OUTDIR = "out/lab3_part9"

MASK = "levin"
K_INIT = 2.628e-02   # starting point for the slider -- picked K from lab3_8.py

# lab3_6_new's default CROP_SIZE (64) is too small for this mask/setup --
# most of these depths' measured PSF blobs (up to ~141px) don't fit in a
# 64x64 window, silently truncating most of the actual coded-aperture
# pattern. Even 256 still clips 1300mm's outer lobes: its analyse_psf_blob
# bounding box (123x62) only covers the single connected component
# touching the brightest pixel, and at 1300mm's low overall brightness
# the fainter outer lobes fall below the 10%-of-peak threshold and drop
# out of that connected blob entirely, so the reported box understates
# the pattern's true extent. 384 was checked directly against the raw
# capture and comfortably contains the whole visible pattern with margin.
CROP_SIZE = 384

def extract_psf_stable(path, fpn, colors, black, white, crop_size):
    """Like lab3_6_new.extract_psf, but locates the crop centre on a
    lightly Gaussian-blurred copy of the frame, then crops the ORIGINAL
    (unsmoothed) data at that location -- the blur is only used to find
    where to crop, never returned.

    analyse_psf_blob anchors on the single brightest pixel plus its
    above-threshold neighbours. At this dataset's dimmest depths
    (1150/1300mm, peak ~0.05-0.07) that's noise-sensitive: which pixel
    is literally brightest is close to a coin-flip between several
    similarly-bright, noisy lobe centres, so the unsmoothed crop centre
    can land well off the pattern's true centre. Smoothing first for
    peak-finding only stabilises that choice without softening the
    actual calibrated PSF values used for deconvolution.
    """
    bayer, _, _, _ = read_raw_bayer(path)
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)
    green = debayer_green(bayer, colors)
    cy, cx, height, width = analyse_psf_blob(gaussian_filter(green, sigma=3.0))
    psf = crop_centred(green, cy, cx, crop_size)
    return psf, (cy, cx), (height, width)


# The five distance markings, same file-naming convention as Part 2/4/6.
DEPTHS = {
    "700mm":  "fwd_300.dng",
    "850mm":  "fwd_150.dng",
    "1000mm": "centre.dng",
    "1150mm": "back_150.dng",
    "1300mm": "back_300.dng",
}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    scene = load_green(SCENE_PATH, fpn, colors, black, white)

    psfs = {}
    fig_psf, axes_psf = plt.subplots(1, len(DEPTHS), figsize=(3.2 * len(DEPTHS), 4.0))
    for ax_psf, (depth, filename) in zip(axes_psf, DEPTHS.items()):
        psf_path = os.path.join(PSF_ROOT, filename)
        psf, (cy, cx), (height, width) = extract_psf_stable(psf_path, fpn, colors, black, white,
                                                              crop_size=CROP_SIZE)
        psfs[depth] = psf
        np.save(os.path.join(OUTDIR, "psf_%s_%s.npy" % (MASK, depth)), psf)

        clipped_pct = 100.0 * np.mean(psf >= SATURATION_LEVEL)
        # Per-panel autoscale (vmax = this capture's own peak), same as
        # lab3_6_new's PSF grid, so shape stays visible regardless of
        # each depth's absolute brightness.
        ax_psf.imshow(psf, cmap="inferno", vmin=0.0, vmax=psf.max())
        ax_psf.set_title("%s\npeak=%.3f  %.0f%% clipped" % (depth, psf.max(), clipped_pct))
        ax_psf.axis("off")

        print("%-8s PSF centroid=(row=%.1f, col=%.1f)  blob=%dx%d px  clipped=%5.1f%%  file=%s"
              % (depth, cy, cx, height, width, clipped_pct, filename))

    fig_psf.suptitle("%s mask -- Part 9 calibration PSFs, green channel only" % MASK)
    fig_psf.tight_layout()
    fig_psf.savefig(os.path.join(OUTDIR, "psf_grid_part9.png"), dpi=150)

    print("\nFor each depth marker a slider window will open on a downsampled preview --")
    print("drag it to tune that depth's own K, close the window to accept it.\n")
    init_idx = int(np.argmin(np.abs(K_VALUES - K_INIT)))
    best_K = {}
    for depth in DEPTHS:
        stack = build_preview_stack(scene, psfs[depth], K_VALUES)
        best_K[depth] = pick_K_by_slider(depth, stack, K_VALUES, init_idx=init_idx,
                                          note=" (downsampled preview)")
        print("%-8s using K=%.3e" % (depth, best_K[depth]))

    recons = {}
    for depth, psf in psfs.items():
        recon = wiener_deconvolve(scene, psf, best_K[depth])
        # The phone was held in portrait, but the raw sensor buffer (and
        # everything upstream of this) stays in its native landscape
        # layout -- rotate only here, for display, to match the scene's
        # true upright orientation (verified against the camera's own
        # preview JPEG/PNG).
        recons[depth] = np.rot90(recon, k=-1)

    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(3.0 * len(DEPTHS), 6.5))
    for ax, (depth, recon) in zip(axes, recons.items()):
        ax.imshow(recon, cmap="gray")
        ax.set_title("%s\nK=%.2e" % (depth, best_K[depth]))
        ax.axis("off")

    fig.suptitle("%s mask: scene deconvolved with the PSF from each distance marker (own K per depth)"
                 % MASK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "part9_depth_sweep.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
