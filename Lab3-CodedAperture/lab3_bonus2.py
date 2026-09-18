"""
MECH5720 Lab 3 -- Bonus 2: Chromatics.

Extracts and compares the PSF for the red, green, and blue channels of
ONE chosen capture, to visualise chromatic aberration -- a PSF that
varies with wavelength, not just spatially (Part 6's corner comparison).

Usage:
    python lab3_bonus2.py [mask] [capture]
e.g. python lab3_bonus2.py nayar centre
     python lab3_bonus2.py levin back_150
     python lab3_bonus2.py circle back_300
Defaults to nayar/centre if no arguments are given. <mask> is a folder
under DATA_ROOT (levin/nayar/circle); <capture> is any Part 2 filename
stem that exists there (centre, back_150, back_300, back_150_c, ...).

Reuses lab3_6_new.py's FPN/black-level correction, centroid-finding,
and crop helpers (same camera, same dataset) -- the only new piece here
is demosaicing R/B as well as G.

NOTE on crop alignment: each channel's centroid is measured
INDEPENDENTLY (see the printed "shift relative to green"), since a
genuine lateral chromatic shift should show up as a real difference
between them. But the saved/plotted crops all share ONE origin -- green's
centroid -- rather than each being re-centred on itself. Re-centring
each channel independently would hide any such shift inside the crop;
sharing one origin keeps it visible, including in the RGB overlay panel.

Also deconvolves ONE natural-scene capture (data/lab3_2/<mask>/part4/
<capture>.dng -- same mask + depth name as the PSF capture above, e.g.
"nayar centre" uses both part2/centre.dng for the PSF and
part4/centre.dng as the scene) using the green-channel PSF applied to
all three of its R/G/B channels, per the brief's first chromatics step
("deconvolve your image using the green PSF on each of the red, green
and blue channels"). Reuses lab3_8.py's Wiener core (wiener_deconvolve)
and its interactive K-picking slider (pick_K_by_slider on a
build_preview_stack), the same pattern lab3_9.py uses.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import lab3_6_new as L
from lab3_8 import (
    wiener_deconvolve,
    build_preview_stack,
    pick_K_by_slider,
    crop_centre_fraction,
    K_VALUES,
)

OUTDIR = "out/lab3_bonus2"
CROP_SIZE = L.CROP_SIZE
CROP_FRACTION = 0.6   # for the before/after scene comparison figure, matches lab3_8's convention

_KG = L._KG
_KRB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 4.0


def debayer_channel(bayer, colors, channel):
    """Bilinear-demosaic one Bayer channel ('R', 'G', or 'B') -> (H, W)."""
    h, w = bayer.shape
    mask2x2 = (colors == channel)
    reps = ((h + 1) // 2 + 1, (w + 1) // 2 + 1)
    full_mask = np.tile(mask2x2, reps)[:h, :w]
    samp = np.where(full_mask, bayer, 0.0)
    kernel = _KG if channel == "G" else _KRB
    return convolve(samp, kernel, mode="mirror")


def extract_rgb_psfs(path, fpn, colors, black, white, crop_size=CROP_SIZE):
    """Correct + demosaic one capture into R/G/B, locate each channel's own
    centroid, then crop all three from a shared (green-centred) origin.

    Returns (psfs, centroids): psfs = {"R"/"G"/"B": crop_size square array},
    centroids = {"R"/"G"/"B": (cy, cx, height, width)} in full-frame coords.
    """
    bayer, _, _, _ = L.read_raw_bayer(path)
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)

    channels, centroids = {}, {}
    for ch in "RGB":
        img = debayer_channel(bayer, colors, ch)
        cy, cx, height, width = L.analyse_psf_blob(img)
        channels[ch] = img
        centroids[ch] = (cy, cx, height, width)

    gy, gx, _, _ = centroids["G"]
    psfs = {ch: L.crop_centred(channels[ch], gy, gx, crop_size) for ch in "RGB"}
    return psfs, centroids


def load_rgb_scene(path, fpn, colors, black, white):
    """Correct + demosaic one capture's full frame into R/G/B (no crop) --
    the scene image to deconvolve, as opposed to extract_rgb_psfs's small
    PSF-calibration crop."""
    bayer, _, _, _ = L.read_raw_bayer(path)
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)
    return {ch: debayer_channel(bayer, colors, ch) for ch in "RGB"}


def to_display_rgb(channels, gamma=1.0 / 2.2, lo_pct=1, hi_pct=99):
    """Stack an {"R","G","B": array} dict into a viewable RGB image: clip
    negatives, percentile-normalise, gamma-stretch. Display only -- the
    saved .npy channels are never touched by this."""
    stacked = np.clip(np.stack([channels["R"], channels["G"], channels["B"]], axis=-1), 0, None)
    lo, hi = np.percentile(stacked, [lo_pct, hi_pct])
    stacked = np.clip((stacked - lo) / max(hi - lo, 1e-9), 0, 1)
    return stacked ** gamma


def main():
    mask = sys.argv[1] if len(sys.argv) > 1 else "nayar"
    capture = sys.argv[2] if len(sys.argv) > 2 else "centre"

    os.makedirs(OUTDIR, exist_ok=True)
    fpn, colors, black, white = L.compute_fixed_pattern_noise(L.DARK_DIR)

    path = os.path.join(L.DATA_ROOT, mask, "part2", capture + ".dng")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No capture at %s -- check the mask/capture names (e.g. 'nayar centre', 'levin back_150')" % path)

    psfs, centroids = extract_rgb_psfs(path, fpn, colors, black, white)

    for ch in "RGB":
        np.save(os.path.join(OUTDIR, "psf_%s_%s_%s.npy" % (ch, mask, capture)), psfs[ch])
        cy, cx, height, width = centroids[ch]
        print("%s  centroid=(row=%.1f, col=%.1f)  blob=%dx%d px  peak=%.3f"
              % (ch, cy, cx, height, width, psfs[ch].max()))

    gy, gx, _, _ = centroids["G"]
    ry, rx, _, _ = centroids["R"]
    by, bx, _, _ = centroids["B"]
    print("Shift relative to green: R=(row%+.1f, col%+.1f)  B=(row%+.1f, col%+.1f)"
          % (ry - gy, rx - gx, by - gy, bx - gx))

    cmap_for = {"R": "Reds", "G": "Greens", "B": "Blues"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, ch in zip(axes[:3], "RGB"):
        ax.imshow(psfs[ch], cmap=cmap_for[ch], vmin=0.0, vmax=max(psfs[ch].max(), 1e-9))
        ax.set_title("%s channel\npeak=%.3f" % (ch, psfs[ch].max()))
        ax.axis("off")

    rgb_composite = np.clip(np.stack([psfs["R"], psfs["G"], psfs["B"]], axis=-1), 0, None)
    rgb_composite = rgb_composite / max(rgb_composite.max(), 1e-9)  # display only; saved .npy stays unclamped
    axes[3].imshow(rgb_composite)
    axes[3].set_title("RGB overlay\n(colour fringing = chromatic shift)")
    axes[3].axis("off")

    fig.suptitle("Chromatic PSF comparison -- %s mask, %s" % (mask, capture))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "chromatic_%s_%s.png" % (mask, capture)), dpi=150)

    # ------------------------------------------------------------------
    # Deconvolve the matching natural-scene capture (part4/<capture>.dng,
    # same mask + depth name) using the GREEN PSF applied to all three of
    # its channels -- the brief's first chromatics step.
    # ------------------------------------------------------------------
    scene_path = os.path.join(L.DATA_ROOT, mask, "part4", capture + ".dng")
    if not os.path.exists(scene_path):
        raise FileNotFoundError(
            "No scene capture at %s -- part4 doesn't have a file matching this capture name" % scene_path)

    scene = load_rgb_scene(scene_path, fpn, colors, black, white)
    green_psf = psfs["G"]

    print("\nPick K for deconvolving with the green PSF (applied to all three channels) --")
    print("slider runs on the green channel's own preview, same K is then used for R/G/B.\n")
    stack = build_preview_stack(scene["G"], green_psf, K_VALUES)
    init_idx = len(K_VALUES) // 2
    K_green_psf = pick_K_by_slider("%s %s (green PSF on all channels)" % (mask, capture),
                                    stack, K_VALUES, init_idx=init_idx, note=" (downsampled preview)")
    print("using K=%.3e" % K_green_psf)

    recon_green_psf = {ch: wiener_deconvolve(scene[ch], green_psf, K_green_psf) for ch in "RGB"}
    for ch in "RGB":
        np.save(os.path.join(OUTDIR, "recon_greenpsf_%s_%s_%s.npy" % (ch, mask, capture)), recon_green_psf[ch])

    scene_cropped = {ch: crop_centre_fraction(scene[ch], CROP_FRACTION) for ch in "RGB"}
    recon_cropped = {ch: crop_centre_fraction(recon_green_psf[ch], CROP_FRACTION) for ch in "RGB"}
    before = to_display_rgb(scene_cropped)
    after = to_display_rgb(recon_cropped)

    fig_dec, axes_dec = plt.subplots(1, 2, figsize=(11, 5.5))
    axes_dec[0].imshow(before)
    axes_dec[0].set_title("Scene (blurred)")
    axes_dec[0].axis("off")
    axes_dec[1].imshow(after)
    axes_dec[1].set_title("Deconvolved with green PSF\n(applied to R, G and B), K=%.2e" % K_green_psf)
    axes_dec[1].axis("off")

    fig_dec.suptitle("Green-PSF deconvolution on all channels -- %s mask, %s" % (mask, capture))
    fig_dec.tight_layout()
    fig_dec.savefig(os.path.join(OUTDIR, "deconv_greenpsf_%s_%s.png" % (mask, capture)), dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
