"""
MECH5720 Lab 3 -- Part 0: Setup and fixed-pattern-noise characterisation.

Averages the >=10 dark frames captured with the lens capped (same
exposure/gain used throughout the rest of the lab) to estimate the
camera's fixed pattern noise (FPN): a per-pixel, spatially-fixed offset
from the nominal black level caused by sensor-level nonidealities (e.g.
per-column readout/amplifier offsets, per-pixel dark current variation)
rather than by the scene. Each frame is black-level-subtracted before
averaging; the average is left signed (not clipped) since clipping is
only meaningful once this map is later subtracted from a live capture.

This FPN map is the one every other Part's script subtracts from its raw
captures -- see compute_fixed_pattern_noise, defined once in lab3_6.py
and reused (not reimplemented) here and everywhere else in this lab.

Visualises the full-frame FPN and a 64x64 centre crop for the report,
and prints summary statistics used to characterise the noise's physical
origin (row- vs column-wise structure).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from lab3_6 import compute_fixed_pattern_noise

DARK_DIR = "data/lab3/part0"
OUTDIR = "out/lab3_part0"
CROP_SIZE = 64


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    # Column- vs row-wise structure: averaging out the other axis isolates
    # each direction's contribution, pointing at the noise's physical origin
    # (e.g. per-column readout offsets vs per-row/per-pixel effects).
    col_banding_std = fpn.mean(axis=0).std()   # variation across columns
    row_banding_std = fpn.mean(axis=1).std()   # variation across rows
    print("FPN stats (raw ADU, black-level-subtracted): mean=%.3f  std=%.3f  min=%.3f  max=%.3f"
          % (fpn.mean(), fpn.std(), fpn.min(), fpn.max()))
    print("Column-wise banding std=%.3f  row-wise banding std=%.3f"
          % (col_banding_std, row_banding_std))
    print("black_level=%.1f  white_level=%.1f  CFA=%s" % (black, white, colors))

    h, w = fpn.shape
    r0, c0 = (h - CROP_SIZE) // 2, (w - CROP_SIZE) // 2  # frame-centred crop
    crop = fpn[r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE]

    # Signed, diverging colour scale shared across both panels: FPN is a
    # zero-centred offset (not an intensity), so autoscaling per-panel
    # would make the crop's smaller dynamic range misleadingly look as
    # extreme as the full frame's.
    vmax = np.max(np.abs(fpn))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(fpn, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Full-frame FPN")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="raw ADU (black-level-subtracted)")

    im1 = axes[1].imshow(crop, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("%dx%d crop @ row=%d, col=%d" % (CROP_SIZE, CROP_SIZE, r0, c0))
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="raw ADU (black-level-subtracted)")

    fig.suptitle("Part 0: Fixed Pattern Noise")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fpn.png"), dpi=150)

    plt.show()
