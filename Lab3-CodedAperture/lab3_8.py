"""
MECH5720 Lab 3 -- Part 8: Deconvolution

Wiener-deconvolves the Siemens Star and natural-scene captures using the
PSFs calibrated in Part 6 (out/lab3_part6_128x128/psf_<mask>_<depth>.npy).

Three pieces, matching the brief:

1. Siemens Star: for each mask, sweep the Wiener constant K over
   [1e-6, 5e-1] and score each reconstruction by the variance of its
   Laplacian (a sharpness proxy -- there is no clean ground truth for a
   real capture, so unlike the lecture's synthetic RMSE-vs-truth sweep,
   we can't score against a known answer). Report the sweep curve, plus
   an example reconstruction at each mask's picked K.

2. Natural scene: deconvolve each mask's three depth captures with that
   depth's own calibrated PSF, and crop a shared region across all
   masks/depths for a fair side-by-side.

3. Gold Standard: average the 10(+) burst frames captured at one mask's
   1150mm marking into a low-noise "Gold Standard" image, and compare
   standard (naive inverse) vs Wiener deconvolution on it and on a
   single frame from the same scene -- this is where the naive
   inverse's noise-amplification failure mode should be obvious.

NOTE on picking K: the variance-of-Laplacian metric is a proxy, not a
ground truth. At very small K (~naive inverse) amplified sensor noise
is itself high-frequency and can make the metric peak there rather than
at a genuinely sharp reconstruction -- so instead of the raw global
argmax, `pick_K_by_curvature` uses the standard "L-curve" corner-finding
heuristic (max curvature of the metric-vs-K curve in log-log space),
the usual way to choose a Tikhonov/Wiener regularisation constant
without a ground truth. Always sanity-check the picked K's image by eye
before reporting it -- that's exactly the brief's own hint (think about
dividing by the OTF near its nulls).
"""
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from lab3_6 import (
    DATA_ROOT,
    compute_fixed_pattern_noise,
    read_raw_bayer,
    debayer_green,
)

# ======================================================================
# Configuration
# ======================================================================
DARK_DIR = os.path.join(DATA_ROOT, "part0")
OUTDIR = "out/lab3_part8"
PSF_DIR = "out/lab3_2_part6"   # calibrated PSFs saved by lab3_6.py

MASKS = ("levin", "nayar", "circle")

# Part 3: Siemens Star, shot at the furthest distance marking only.
SIEMENS_STAR_DEPTH = "1300mm"
SIEMENS_STAR_FILE = "mtf.dng"

# Part 4/5 natural-scene captures: third/fourth/last distance markings.
# File naming isn't consistent between masks (nayar uses "back_", the
# others "bck_") -- spelled out explicitly rather than guessed.
NATURAL_SCENE = {
    "levin":  {"1000mm": "centre.dng", "1150mm": "bck_150.dng",  "1300mm": "bck_300.dng"},
    "nayar":  {"1000mm": "centre.dng", "1150mm": "back_150.dng", "1300mm": "back_300.dng"},
    "circle": {"1000mm": "centre.dng", "1150mm": "bck_150.dng",  "1300mm": "bck_300.dng"},
}

# The 10(+)-frame burst was captured at the "1150mm" natural-scene shot
# above -- same scene, same distance, just repeated for averaging.
BURST_GLOB = {
    "levin":  "bck_150_*.dng",
    "nayar":  "back_150_*.dng",
    "circle": "bck_150_*.dng",
}
GOLD_STANDARD_MASK = "levin"     # "a mask of your own choosing from Nayar or Levin"
GOLD_STANDARD_DEPTH = "1150mm"

N_K = 50
K_VALUES = np.logspace(-6, np.log10(5e-1), N_K)   # brief: sweep 1e-6 to 5e-1

# Centre-crop fraction for the natural-scene/Gold Standard comparison
# figures -- zooms in enough to see each K's actual blur/sharpness
# without cutting down to a tiny, oddly-composed window.
CROP_FRACTION = 0.6

# ======================================================================
# Preprocessing: raw -> black-level -> FPN -> normalise -> green-only
# debayer. Same steps as lab3_6.extract_psf, minus the PSF-centred crop
# -- Part 8 needs the full frame, not a small window around a point
# source.
# ======================================================================
def load_green(path, fpn, colors, black, white):
    bayer, _, _, _ = read_raw_bayer(path) 
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)
    return debayer_green(bayer, colors)


def load_burst_mean_green(paths, fpn, colors, black, white):
    """Average a burst into one low-noise green plane.

    Black-level and FPN subtraction and the final normalisation are all
    per-pixel linear operations, so correcting each raw frame before
    averaging vs. averaging the raw frames first and correcting once is
    mathematically identical (no clipping happens anywhere in this
    pipeline to break that equivalence). We do it the second way -- sum
    the raw Bayer frames, correct once, debayer once -- since it's the
    same result for a fraction of the work.
    """
    bayers = []
    colors = black = white = None
    for p in paths:
        bayer, colors, black, white = read_raw_bayer(p)
        bayers.append(bayer)
    mean_bayer = np.mean(bayers, axis=0)
    mean_bayer = mean_bayer - black
    mean_bayer = mean_bayer - fpn
    mean_bayer = mean_bayer / (white - black)
    return debayer_green(mean_bayer, colors), len(paths)


def load_psf(mask, depth):
    return np.load(os.path.join(PSF_DIR, "psf_%s_%s.npy" % (mask, depth)))


# ======================================================================
# Wiener deconvolution core
# ======================================================================
def psf2otf(psf, shape):
    """Zero-pad `psf` to `shape` and circularly shift it so its own
    centre pixel lands at index [0, 0] before the DFT (MATLAB's
    `psf2otf` idiom -- the `otf()` local function in the reference
    Matlab code). Skipping the shift still zero-pads correctly but
    leaves a linear phase ramp in the OTF: the deconvolved image comes
    back shifted/wrapped rather than visibly "wrong shaped", which is
    what makes the mistake easy to miss.
    """
    psf = psf / psf.sum()
    padded = np.zeros(shape, dtype=np.float64)
    padded[:psf.shape[0], :psf.shape[1]] = psf
    padded = np.roll(padded, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(padded)


def wiener_deconvolve(image, psf, K):
    """Wiener-deconvolve `image` with `psf`. K=None gives the naive/
    standard inverse (1/H) instead, for the Gold Standard comparison.

    Symmetric-pads `image` by the PSF size on each side before the FFT
    round trip (matching the Matlab reference's `wienerCycle`): FFT-based
    deconvolution assumes the image tiles periodically, and without the
    pad, content wraps from one edge of the frame into the opposite one.
    """
    pad = psf.shape[0]
    padded = np.pad(image, pad, mode="symmetric")
    H = psf2otf(psf, padded.shape)
    if K is None:
        Wf = 1.0 / H
    else:
        Wf = np.conj(H) / (np.abs(H) ** 2 + K)
    restored = np.real(np.fft.ifft2(Wf * np.fft.fft2(padded)))
    return restored[pad:pad + image.shape[0], pad:pad + image.shape[1]]


def laplacian_variance(image):
    """Variance of the Laplacian -- the brief's proxy for reconstruction
    sharpness, standing in for RMSE-vs-ground-truth (which needs a known
    clean image we don't have for a real capture)."""
    return cv2.Laplacian(image.astype(np.float64), cv2.CV_64F).var()


def pick_K_by_curvature(Ks, metric):
    """L-curve corner: the K of maximum curvature of the metric-vs-K
    curve in log-log space. Used instead of the raw argmax because the
    metric tends to be highest at the smallest K, where amplified sensor
    noise -- not real detail -- inflates the Laplacian variance."""
    x, y = np.log10(Ks), np.log10(np.clip(metric, 1e-12, None))
    d1 = np.gradient(y, x)
    d2 = np.gradient(d1, x)
    curvature = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    return Ks[np.argmax(curvature)]


def crop_centre_fraction(img, fraction):
    """Centre-crop to `fraction` of each dimension, preserving aspect
    ratio -- unlike a fixed small square crop (e.g. 400x400 out of a
    1920x1080 frame), this keeps the scene's proportions and most of its
    content, just trimming the outer margin enough to zoom in on the
    picked K's actual sharpness/blur without losing context."""
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return img[r0:r0 + ch, c0:c0 + cw]


def build_preview_stack(image, psf, Ks, preview_scale=0.25):
    """Downsample `image`/`psf` and precompute a Wiener reconstruction at
    every K in `Ks` -- cheap enough to hold in memory and scrub with
    `pick_K_by_slider` at no per-frame lag (see that function's
    docstring for why full-resolution isn't used here)."""
    small_img = cv2.resize(image, None, fx=preview_scale, fy=preview_scale,
                            interpolation=cv2.INTER_AREA)
    small_psf = cv2.resize(psf, None, fx=preview_scale, fy=preview_scale,
                            interpolation=cv2.INTER_AREA)
    return np.stack([wiener_deconvolve(small_img, small_psf, k) for k in Ks])


def pick_K_by_slider(mask, stack, Ks, init_idx=0, note=""):
    """Scrub a precomputed stack of Wiener reconstructions across K with
    a slider -- adapted from `show_stack` in
    mech5720/Lab2_Multiplexing/imaging.py (scrub-through-frames becomes
    scrub-through-K here). Close the window to accept whichever K the
    slider is on when it closes.

    `stack` must already be precomputed (one reconstruction per K, same
    order as `Ks`) so scrubbing is instant -- recomputing a Wiener
    deconvolution live on every slider move would lag badly at full
    resolution, which is why the caller builds this stack on a
    downsampled preview copy of the image instead.
    """
    fig = plt.figure(figsize=(6.5, 7.2))
    ax = fig.add_axes([0.05, 0.15, 0.90, 0.78])
    vmin, vmax = np.percentile(stack[init_idx], [1, 99])
    im = ax.imshow(stack[init_idx], cmap="gray", vmin=vmin, vmax=vmax)
    title = ax.set_title("%s%s\nK=%.3e" % (mask, note, Ks[init_idx]))
    ax.set_xticks([])
    ax.set_yticks([])

    ax_slider = fig.add_axes([0.15, 0.05, 0.70, 0.04])
    slider = Slider(ax_slider, "K index", 1, len(Ks), valinit=init_idx + 1, valstep=1)

    def update(val):
        i = int(val) - 1
        vmin, vmax = np.percentile(stack[i], [1, 99])
        im.set_data(stack[i])
        im.set_clim(vmin, vmax)
        title.set_text("%s%s\nK=%.3e" % (mask, note, Ks[i]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider  # keep alive

    print("%-8s: drag the slider (K=%.1e..%.1e), close the window to accept its K"
          % (mask, Ks[0], Ks[-1]))
    plt.show()  # blocks until the window is closed

    return Ks[int(round(slider.val)) - 1]


def show_and_hold(): 
    plt.show(block=False)
    plt.pause(0.2)          # let every figure draw before we block
    try:
        input("press enter to close the figures ")
    except EOFError:        # no interactive stdin: fall back to blocking
        plt.show(block=True)
    plt.close("all")

# ======================================================================
# Main
# ======================================================================
def main(): 
    os.makedirs(OUTDIR, exist_ok=True)
    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    # ------------------------------------------------------------------
    # 1. Siemens Star: one simple diagnostic curve (all 3 masks' K sweeps
    #    overlaid, like the metric plot alone -- it's monotonic and can't
    #    localise a "good" K, see NOTE above), then the user picks K for
    #    each mask by eye with a slider, closing the window to accept it
    #    -- the brief's own hint ("sanity-check the picked K's image by
    #    eye") made literal instead of trusting the L-curve corner
    #    blindly.
    # ------------------------------------------------------------------
    stars, psfs_star, K_auto, mask_color = {}, {}, {}, {}
    fig_sweep, ax_sweep = plt.subplots(figsize=(9, 6))

    for mask in MASKS:
        star_path = os.path.join(DATA_ROOT, mask, "part3", SIEMENS_STAR_FILE)
        star = load_green(star_path, fpn, colors, black, white)
        psf = load_psf(mask, SIEMENS_STAR_DEPTH)
        stars[mask], psfs_star[mask] = star, psf

        metric = np.array([
            laplacian_variance(wiener_deconvolve(star, psf, K)) for K in K_VALUES
        ])
        K_opt = pick_K_by_curvature(K_VALUES, metric)
        K_auto[mask] = K_opt
        print("%-8s Siemens Star: L-curve auto-pick K=%.3e (sweep max at K=%.3e)"
              % (mask, K_opt, K_VALUES[np.argmax(metric)]))

        line, = ax_sweep.loglog(K_VALUES, metric, marker=".", label=mask)
        mask_color[mask] = line.get_color()

    ax_sweep.set_xlabel("K")
    ax_sweep.set_ylabel("variance of Laplacian (sharpness proxy)")
    ax_sweep.set_title("Siemens Star Wiener deconvolution: sharpness proxy vs. K\n"
                        "metric is monotonic -- pick K by eye in the slider windows, not from this curve")
    ax_sweep.legend(loc="upper right")
    ax_sweep.grid(True, which="both", alpha=0.3)
    fig_sweep.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)   # let it actually draw before the slider windows open

    print("\nFor each mask a slider window will open on a downsampled preview --")
    print("drag it to scrub through K, close the window to accept whatever K it's on.\n")
    best_K = {}
    for mask in MASKS:
        stack = build_preview_stack(stars[mask], psfs_star[mask], K_VALUES)
        init_idx = int(np.argmin(np.abs(K_VALUES - K_auto[mask])))

        best_K[mask] = pick_K_by_slider(mask, stack, K_VALUES, init_idx=init_idx,
                                         note=" (downsampled preview)")
        print("%-8s using K=%.3e" % (mask, best_K[mask]))

    # Mark the manually-picked K (not the L-curve auto-pick) on the sweep
    # curve, now that the slider windows have actually set best_K.
    for mask in MASKS:
        ax_sweep.axvline(best_K[mask], color=mask_color[mask], linestyle="--", alpha=0.6,
                          label="%s picked K = %.2e" % (mask, best_K[mask]))
    ax_sweep.legend(loc="upper right", fontsize=8)
    fig_sweep.savefig(os.path.join(OUTDIR, "siemens_star_K_sweep_all_masks.png"), dpi=150)

    recons_at_K = {mask: wiener_deconvolve(stars[mask], psfs_star[mask], best_K[mask])
                   for mask in MASKS}

    fig_ex, axes_ex = plt.subplots(1, len(MASKS), figsize=(4.2 * len(MASKS), 4.5))
    for i, mask in enumerate(MASKS):
        axes_ex[i].imshow(recons_at_K[mask], cmap="gray")
        axes_ex[i].set_title("%s\nK=%.2e" % (mask, best_K[mask]))
        axes_ex[i].axis("off")

    fig_ex.suptitle("Siemens Star deconvolved at each mask's chosen K")
    fig_ex.tight_layout()
    fig_ex.savefig(os.path.join(OUTDIR, "siemens_star_best_K.png"), dpi=150)

    # Levin on its own, at its manually-picked K. figsize matches the
    # image's own (wide, non-square) aspect ratio -- a square figure
    # around a wide image just letterboxes it with vertical whitespace.
    levin_h, levin_w = recons_at_K["levin"].shape
    fig_levin, ax_levin = plt.subplots(figsize=(6, 6 * levin_h / levin_w + 0.5))
    ax_levin.imshow(recons_at_K["levin"], cmap="gray")
    ax_levin.set_title("levin\nK=%.2e" % best_K["levin"])
    ax_levin.axis("off")
    fig_levin.savefig(os.path.join(OUTDIR, "siemens_star_best_K_levin.png"), dpi=150,
                       bbox_inches="tight", pad_inches=0.05)

    # ------------------------------------------------------------------
    # 2. Natural scene: per-mask, per-depth deconvolution + shared crop
    # ------------------------------------------------------------------
    depths = list(NATURAL_SCENE[MASKS[0]].keys())
    fig_scene, axes_scene = plt.subplots(len(MASKS), len(depths),
                                          figsize=(3.6 * len(depths), 3.6 * len(MASKS)))
    for row, mask in enumerate(MASKS):
        for col, depth in enumerate(depths):
            path = os.path.join(DATA_ROOT, mask, "part4", NATURAL_SCENE[mask][depth])
            scene = load_green(path, fpn, colors, black, white)
            psf = load_psf(mask, depth)
            recon = wiener_deconvolve(scene, psf, best_K[mask])
            crop = crop_centre_fraction(recon, CROP_FRACTION)

            ax = axes_scene[row, col]
            ax.imshow(crop, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(depth)
            if col == 0:
                ax.text(-0.08, 0.5, mask, transform=ax.transAxes,
                        ha="right", va="center", fontsize=11)

    fig_scene.suptitle("Natural-scene deconvolution, each mask at its picked K")
    fig_scene.tight_layout()
    fig_scene.savefig(os.path.join(OUTDIR, "natural_scene_deconv.png"), dpi=150)

    # ------------------------------------------------------------------
    # 3. Gold Standard: standard vs. Wiener, burst-average vs. single frame
    # ------------------------------------------------------------------
    burst_paths = sorted(glob.glob(os.path.join(DATA_ROOT, GOLD_STANDARD_MASK, "part4",
                                                 BURST_GLOB[GOLD_STANDARD_MASK])))
    gold, n_frames = load_burst_mean_green(burst_paths, fpn, colors, black, white)
    single_path = os.path.join(DATA_ROOT, GOLD_STANDARD_MASK, "part4",
                                NATURAL_SCENE[GOLD_STANDARD_MASK][GOLD_STANDARD_DEPTH])
    single = load_green(single_path, fpn, colors, black, white)
    psf = load_psf(GOLD_STANDARD_MASK, GOLD_STANDARD_DEPTH)

    # Pick K by eye here too, same slider-on-a-downsampled-preview approach
    # as the Siemens Star -- reusing best_K[GOLD_STANDARD_MASK] would carry
    # over a K picked on a different capture (different depth, and a much
    # lower noise level than either of these two), which is exactly what
    # made the single-frame panel look badly under-regularised.
    print("\nPick K for the Gold Standard comparison -- single frame and burst average")
    print("have very different noise levels, so each gets its own slider.\n")
    init_idx = int(np.argmin(np.abs(K_VALUES - best_K[GOLD_STANDARD_MASK])))

    K_single = pick_K_by_slider("%s single frame" % GOLD_STANDARD_MASK,
                                 build_preview_stack(single, psf, K_VALUES),
                                 K_VALUES, init_idx=init_idx, note=" (downsampled preview)")
    print("%-20s using K=%.3e" % ("single frame", K_single))

    K_gold = pick_K_by_slider("%s Gold Standard" % GOLD_STANDARD_MASK,
                               build_preview_stack(gold, psf, K_VALUES),
                               K_VALUES, init_idx=init_idx, note=" (downsampled preview)")
    print("%-20s using K=%.3e" % ("gold standard", K_gold))

    print("Gold Standard (%s, %d frames averaged): single-frame K=%.3e, gold K=%.3e"
          % (GOLD_STANDARD_MASK, n_frames, K_single, K_gold))

    fig_gold, axes_gold = plt.subplots(2, 2, figsize=(9, 9))
    panels = [
        (single, None, "single frame, standard"),
        (single, K_single, "single frame, Wiener"),
        (gold, None, "Gold Standard, standard"),
        (gold, K_gold, "Gold Standard, Wiener"),
    ]
    for ax, (img, K, title) in zip(axes_gold.ravel(), panels):
        recon = wiener_deconvolve(img, psf, K)
        ax.imshow(crop_centre_fraction(recon, CROP_FRACTION), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig_gold.suptitle("%s mask: standard vs. Wiener deconvolution, single frame vs. Gold Standard"
                       % GOLD_STANDARD_MASK)
    fig_gold.tight_layout()
    fig_gold.savefig(os.path.join(OUTDIR, "gold_standard_comparison.png"), dpi=150)
    
    # Cleanup / close
    show_and_hold()

if __name__ == "__main__":
    main() 