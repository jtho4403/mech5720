"""
MECH5720 Lab 3 -- Part 1: PSF characterisation across depth.

Loads the pinhole point-source captures at the five distance markings
(700 / 850 / 1000 / 1150 / 1300 mm), corrects each for the camera's fixed
pattern noise and black level (Part 0), debayers to the green channel
only, locates the PSF by its intensity centroid, and crops a centred,
fixed-size window around it -- reusing the exact pipeline already built
for Part 6 (see lab3_6.py: read_raw_bayer, debayer_green,
compute_fixed_pattern_noise, analyse_psf_blob, crop_centred).

For each depth this also computes the magnitude of the 2D FFT of the
cropped PSF (a Hann window is applied first, since the crop boundary is
not periodic and would otherwise leak spurious high-frequency energy
into the spectrum).

NOTE on saturation: every capture clips a small fraction of pixels
(~0.1-1%), heaviest at 700mm (~1%) where the PSF is most defocused and
spread over the widest area. This is a minor non-ideality of the
capture exposure, not a processing artefact -- each crop's clipped
fraction is printed and shown in its subplot title.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from lab3_6 import (
    read_raw_bayer,
    debayer_green,
    compute_fixed_pattern_noise,
    analyse_psf_blob,
    crop_centred,
)

# ======================================================================
# Configuration
# ======================================================================
DATA_DIR = "data/lab3/part1"
DARK_DIR = "data/lab3/part0"
OUTDIR = "out/lab3_part1"

# All five distance markings, relative to data/lab3/part1/. Same
# labelling convention as lab3_6.MASKS.
DEPTHS = {
    "700mm":  "light_fwd300.dng",
    "850mm":  "light_fwd150.dng",
    "1000mm": "light.dng",
    "1150mm": "light_bck150.dng",
    "1300mm": "light_bck300.dng",
}

CROP_SIZE = 256          # comfortably contains every measured PSF blob (max ~220x211 px)
SATURATION_LEVEL = 0.98  # normalised value counted as "clipped" for diagnostics


# ======================================================================
# Per-depth pipeline: raw -> black-level correction -> FPN removal ->
# green-only debayer -> centred crop -> windowed FFT magnitude
# ======================================================================
def process_depth(path, fpn, colors, black, white, crop_size=CROP_SIZE):
    bayer, _, _, _ = read_raw_bayer(path)
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)  # nominal [0, 1], no clipping (normalisation)
    green = debayer_green(bayer, colors)

    cy, cx, height, width = analyse_psf_blob(green)
    psf = crop_centred(green, cy, cx, crop_size)

    window = np.outer(np.hanning(crop_size), np.hanning(crop_size))
    spectrum = np.fft.fftshift(np.fft.fft2(psf * window))
    log_magnitude = np.log1p(np.abs(spectrum))

    clipped_pct = 100.0 * np.mean(psf >= SATURATION_LEVEL)
    diagnostics = {
        "centroid": (cy, cx),
        "blob": (height, width),
        "clipped_pct": clipped_pct,
    }
    return psf, log_magnitude, diagnostics


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    psfs = {}
    spectra = {}
    fig, axes = plt.subplots(2, len(DEPTHS), figsize=(3.2 * len(DEPTHS), 7.4))
    for col, (depth_label, filename) in enumerate(DEPTHS.items()):
        path = os.path.join(DATA_DIR, filename)
        psf, log_mag, diag = process_depth(path, fpn, colors, black, white)
        psfs[depth_label] = psf
        spectra[depth_label] = log_mag
        np.save(os.path.join(OUTDIR, "psf_%s.npy" % depth_label), psf)

        cy, cx = diag["centroid"]
        height, width = diag["blob"]
        print("%-8s centroid=(row=%.1f, col=%.1f)  blob=%dx%d px  clipped=%5.1f%%  file=%s"
              % (depth_label, cy, cx, height, width, diag["clipped_pct"], filename))

        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.imshow(psf, cmap="inferno", vmin=0.0, vmax=1.0)
        ax_top.set_title("%s\n%.0f%% clipped" % (depth_label, diag["clipped_pct"]))
        ax_top.axis("off")

        ax_bot.imshow(log_mag, cmap="viridis")
        ax_bot.set_title("log|FFT|")
        ax_bot.axis("off")

    fig.suptitle("Captured PSF (top) and log-magnitude spectrum (bottom) vs. depth")
    fig.tight_layout(h_pad=4.0)
    fig.subplots_adjust(top=0.88)
    fig.savefig(os.path.join(OUTDIR, "psf_fft_grid.png"), dpi=150)

    plt.show()
