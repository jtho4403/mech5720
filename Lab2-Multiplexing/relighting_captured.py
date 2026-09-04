"""
MECH5720 Lab 2 -- Part 4: Relight YOUR captured scene.

Adapted from relighting.py to work on the raw DNG frames captured in
Part 2. Two differences from the provided PNG example drive every change
below:

  * the frames are Bayer-coded raw (DNG), not debayered colour PNGs, and
  * they carry a non-zero black-level pedestal, and (because the box
    blocks room light) almost no ambient illumination.

One switch selects between the two approaches the brief asks for:

    DEBAYER_MODE = "first"   # Part 4.1  (required)   debayer each frame at load
    DEBAYER_MODE = "last"    # Part 4.2  (bonus)      keep Bayer data throughout,
                             #                        demosaic only for display

The maths in the middle (ambient removal, the H^-1 solve, the PSNR
advantage) is IDENTICAL for both modes: every operation is per-pixel, so
it does not care whether a pixel is one of three colour channels or one
Bayer-masked sample. That is the whole reason debayer-last is possible.

Three conveniences over the first version:
  * OUTDIR       -- figures are written to their own Part 4 folder, so
                    they do not overwrite the Part 3 (relighting.py) figures.
  * DISPLAY_GAIN -- a display-ONLY brightness multiplier. The box + short
                    exposure make the scene genuinely dim; this brightens
                    the *figures* without ever touching the data that the
                    MSE/PSNR are computed from.
  * CROP_SWEEP / RUN_SWEEP -- run the pipeline over several crop windows
                    and print a small PSNR table, to answer Q9 (how the
                    crop region affects PSNR and the SNR advantage) with
                    numbers.

Requires:  pip install rawpy scipy matplotlib
"""
import os
import re
import glob

import numpy as np
from scipy.linalg import hadamard
from scipy.ndimage import convolve

import imaging as im

try:
    import rawpy
except ImportError as err:                       # pragma: no cover
    raise SystemExit(
        "rawpy is needed to read the .dng frames. Install it with:\n"
        "    pip install rawpy\n"
        "(original error: %s)" % err)

# ======================================================================
# Configuration
# ======================================================================
DATA_PATH = "data/lab2/captured"     # folder of .dng frames from Part 2
OUTDIR = "out_part4"                 # figures land here (separate from Part 3)

DEBAYER_MODE = "first"               # "first" (required) or "last" (bonus)

# Crop. The brief tells you to crop before processing. Set this over the
# INTERESTING part of your scene (the transparent + mirror objects), not
# the geometric centre. For debayer-LAST the origin and size MUST be even
# or the crop shifts the Bayer phase and the demosaic assumes the wrong
# pattern. None => auto-centred even crop of CROP_SIZE (a fallback only).
CROP = (250, 400, 900, 1800)                          # or (row0, col0, height, width), all even
CROP_SIZE = 512

# Display-only brightness. 1.0 = untouched. Raise it (e.g. 4-8) to make
# the dim captured scene legible in your report figures. This is applied
# ONLY to images handed to a display call; the arrays that feed MSE/PSNR
# are never scaled.
DISPLAY_GAIN = 8.0

# Black level. Set False to run the Q8 experiment: leaving the pedestal
# in shows what happens if you "forget" to remove it (the advantages
# barely move, because the ambient frame carries the same pedestal).
SUBTRACT_BLACK = True

# Q9 crop sweep. Each entry is (name, crop) with crop = (r0, c0, h, w),
# all even, or None for the auto centre crop. Set RUN_SWEEP = True to
# print a table of the PSNR advantage for each region (no figures).
RUN_SWEEP = True
CROP_SWEEP = [
    ("centre",     None),
    ("objects",    (250, 400, 900, 1800)),   # <- all even, same as crop window
    ("background", (200, 200, 256, 256)),    # zoomed/cropped section of background, no objects present in this window
]

N_SITES   = 15                       # 5 x 3 illumination grid == 15 frames
N_AMBIENT = 16
N_ALLON   = 16
N_FIRSTON = 16

# Site layout on the monitor, from capture_relighting.py: GRID = (5, 3),
# numbered top-left, left-to-right along each row, then down. A
# checkerboard is the sites with (row + col) even; for a 5-wide grid that
# is exactly every second index. Change GRID_COLS if your grid differs.
GRID_COLS, GRID_ROWS = 5, 3

GAMMA_DISPLAY = 2.0
GAMMA_ERROR   = 3.3

ERROR_VIS_SCALE = 60.0   # display-only lift for the tiny squared error; tune 30–200

im.plt.ion()

# ======================================================================
# Raw DNG loading + bilinear demosaic  (replaces imaging.load_stack,
# which reads PNG/TIFF only)
# ======================================================================
_KG  = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], float) / 4.0
_KRB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 4.0


def _cfa_colors(raw):
    """The 2x2 colour-filter-array layout of this sensor, e.g. BGGR.

    Read from the DNG metadata rather than hard-coded, so the code stays
    correct even if your sensor's Bayer phase differs from ours.
    """
    desc = raw.color_desc.decode()          # e.g. 'RGBG'
    pat = raw.raw_pattern                    # 2x2 indices into desc
    return np.array([[desc[pat[i, j]] for j in range(2)] for i in range(2)])


def demosaic_bilinear(bayer, colors):
    """Bilinear demosaic of a single-channel Bayer frame -> (H, W, 3).

    Linear interpolation only: no gamma, no white balance, no clipping.
    The demultiplexing maths needs a linear light->value relationship, so
    the demosaic must stay linear too.

    NOTE (Q14): this interpolation is a spatial low-pass. It averages
    neighbouring pixels, which correlates and slightly attenuates the
    per-pixel noise. Debayer-FIRST runs the whole pipeline on already
    smoothed data (optimistic PSNR); debayer-LAST keeps the true
    per-pixel noise and only smooths at the end, for display.
    """
    h, w = bayer.shape
    out = np.zeros((h, w, 3), float)
    for ci, ch in enumerate("RGB"):
        m = (colors == ch)                                   # 2x2 mask
        reps = ((h + 1) // 2 + 1, (w + 1) // 2 + 1)
        full = np.tile(m, reps)[:h, :w]                      # tile to full frame
        samp = np.where(full, bayer, 0.0)
        out[:, :, ci] = convolve(samp, _KG if ch == "G" else _KRB, mode="mirror")
    return out


def _numeric_key(path):
    digits = re.findall(r"\d+", os.path.basename(path))
    return int(digits[-1]) if digits else -1


def read_dng_bayer(path, subtract_black):
    """Read one DNG as a normalised, single-channel Bayer frame.

    Returns (bayer, colors, black, white), pixels scaled to nominal
    [0, 1] by (raw - black) / (white - black). Nothing is clamped: a
    sample below black is a legitimate negative noise excursion, and
    clamping it here would bias every downstream measurement.
    """
    raw = rawpy.imread(path)
    bayer = raw.raw_image_visible.astype(np.float64)
    colors = _cfa_colors(raw)
    black = float(np.mean(raw.black_level_per_channel))
    white = float(raw.white_level)
    raw.close()

    b = black if subtract_black else 0.0                     # Q8 toggle
    bayer = (bayer - b) / (white - black)
    return bayer, colors, black, white


def _list_frames(path, prefix, n):
    files = sorted(
        glob.glob(os.path.join(path, prefix + "*.dng"))
        + glob.glob(os.path.join(path, prefix + "*.DNG")),
        key=_numeric_key)
    if len(files) < n:
        raise FileNotFoundError(
            "Expected %d '%s*' .dng frames in %s, found %d"
            % (n, prefix, path, len(files)))
    return files[:n]


def load_stack_dng(path, prefix, n, crop, mode, subtract_black):
    """Load a numbered set of DNG frames as a stack.

    debayer-first -> (N, H, W, 3) colour.
    debayer-last  -> (N, H, W, 1) Bayer-masked single channel.

    The crop is applied to the Bayer frame BEFORE any demosaic, so in
    debayer-last it must sit on even boundaries (enforced by the caller).
    """
    files = _list_frames(path, prefix, n)
    r0, c0, hh, ww = crop
    frames, colors = [], None
    black = white = None
    for f in files:
        bayer, colors, black, white = read_dng_bayer(f, subtract_black)
        sub = bayer[r0:r0 + hh, c0:c0 + ww]
        if mode == "first":
            frames.append(demosaic_bilinear(sub, colors))    # (h, w, 3)
        else:
            frames.append(sub[:, :, None])                   # (h, w, 1)
    return np.stack(frames), colors, black, white


def auto_even_crop(shape, size):
    """Centred crop of side `size`, forced onto even row/col boundaries."""
    h0, w0 = shape[:2]
    s = min(size, h0 - (h0 % 2), w0 - (w0 % 2))
    r0 = (h0 - s) // 2; r0 -= r0 % 2
    c0 = (w0 - s) // 2; c0 -= c0 % 2
    return (r0, c0, s, s)


def resolve_crop(crop):
    """Turn None into an auto crop, and check even boundaries."""
    if crop is None:
        probe, _, _, _ = read_dng_bayer(
            _list_frames(DATA_PATH, "Multiplexed", N_SITES)[0], SUBTRACT_BLACK)
        crop = auto_even_crop(probe.shape, CROP_SIZE)
    assert crop[0] % 2 == 0 and crop[1] % 2 == 0 and crop[2] % 2 == 0 \
        and crop[3] % 2 == 0, "crop must be even (Bayer phase): %s" % (crop,)
    return crop


def to_disp(x, colors, mode):
    """Demosaic for display in debayer-last mode; else pass through."""
    if mode != "last" or x.shape[-1] != 1:
        return x
    if x.ndim == 3:
        return demosaic_bilinear(x[..., 0], colors)
    if x.ndim == 4:
        return np.stack([demosaic_bilinear(x[i, ..., 0], colors)
                         for i in range(x.shape[0])])
    return x


# ======================================================================
# The whole pipeline, as a function so a crop sweep is a one-line loop
# ======================================================================
def relight(crop=None, mode=DEBAYER_MODE, outdir=OUTDIR,
            gain=DISPLAY_GAIN, show=True, save=True, verbose=True):
    """Run demux + relight + evaluation for one crop and mode.

    show=False / save=False skips every figure (used by the crop sweep).
    Returns a dict of the PSNR numbers.
    """
    crop = resolve_crop(crop)

    # -- 2. multiplexed frames (also fixes `colors` for display) -------
    mux, colors, black, white = load_stack_dng(
        DATA_PATH, "Multiplexed", N_SITES, crop, mode, SUBTRACT_BLACK)

    # scene image -> display: demosaic if needed, then brighten (display only)
    def S(x):
        return to_disp(x, colors, mode) * gain

    # -- 1. multiplexing matrix (must match capture time) --------------
    H = hadamard(N_SITES + 1)[1:, 1:]
    H = (1 - H) / 2
    if show:
        fig = im.show(H[:, :, None], "Multiplexing matrix H", gamma=1.0, fignum=1)
        if save:
            im.save_figure(fig, "01_multiplexing_matrix", outdir)
        fig = im.show_stack(S(mux), "Multiplexed input frames",
                            GAMMA_DISPLAY, fignum=2)
        if save:
            im.save_figure(fig, "02_multiplexed_input", outdir)

    # -- 3. ambient estimate -------------------------------------------
    ambient_stack, _, _, _ = load_stack_dng(
        DATA_PATH, "Ambient", N_AMBIENT, crop, mode, SUBTRACT_BLACK)
    ambient = ambient_stack.mean(axis=0)
    if show:
        fig = im.show(S(ambient), "Ambient, averaged over %d frames" % N_AMBIENT,
                      GAMMA_DISPLAY, fignum=4)
        if save:
            im.save_figure(fig, "03_ambient_averaged", outdir)

    # -- 4. remove ambient (do NOT clamp) ------------------------------
    mux = mux - ambient
    if show:
        fig = im.show_stack(S(mux), "Multiplexed input, ambient removed",
                            GAMMA_DISPLAY, fignum=5)
        if save:
            im.save_figure(fig, "04_multiplexed_ambient_removed", outdir)

    # -- 5. demultiplex:  single_site = H^-1 @ mux ---------------------
    original_shape = mux.shape
    demux = np.linalg.solve(H, mux.reshape(N_SITES, -1)).reshape(original_shape)
    if show:
        fig = im.show_stack(S(im.side_by_side(mux, demux)),
                            "Multiplexed (left) and demultiplexed (right)",
                            GAMMA_DISPLAY, fignum=6)
        if save:
            im.save_figure(fig, "05_mux_vs_demux", outdir)

    # -- 6. relight ----------------------------------------------------
    if show:
        fig = im.show_stack(S(demux + ambient),
                            "Single site illuminated, with ambient",
                            GAMMA_DISPLAY, fignum=7)

    allon_est_demux = demux.sum(axis=0) + ambient
    if show:
        fig = im.show(S(allon_est_demux), "Synthesised: all sites on",
                      GAMMA_DISPLAY, fignum=8)
        if save:
            im.save_figure(fig, "06_synth_allon", outdir)

    site_ids = np.arange(N_SITES)
    rows_, cols_ = site_ids // GRID_COLS, site_ids % GRID_COLS
    checker_indices = site_ids[(rows_ + cols_) % 2 == 0]
    relit = demux[checker_indices].sum(axis=0) + ambient
    if show:
        fig = im.show(S(relit), "Synthesised: checkerboard illumination",
                      GAMMA_DISPLAY, fignum=9)
        if save:
            im.save_figure(fig, "07_synth_checkerboard", outdir)

    # creative colour relight, built from a colour copy so it works in
    # both modes (debayer-last demux is mono, so demosaic it here)
    if show:
        demux_col = demux if demux.shape[-1] == 3 else to_disp(demux, colors, mode)
        amb_col = ambient if ambient.shape[-1] == 3 else to_disp(ambient, colors, mode)
        creative = np.zeros_like(demux_col[0])
        creative[:, :, 0] = 1.2 * demux_col[0:5:2, :, :, 0].sum(axis=0)   # warm key
        creative[:, :, 1] = 0.9 * demux_col[6:11, :, :, 1].sum(axis=0)    # green fill
        creative[:, :, 2] = 1.6 * demux_col[10::2, :, :, 2].sum(axis=0)   # cool rim
        creative = creative + amb_col / 4.0
        fig = im.show(creative * gain, "Creative relighting", GAMMA_DISPLAY, fignum=10)
        if save:
            im.save_figure(fig, "08_creative", outdir)

    # -- 7. impulse comparison -----------------------------------------
    impulse, _, _, _ = load_stack_dng(
        DATA_PATH, "Impulse", N_SITES, crop, mode, SUBTRACT_BLACK)
    if show:
        fig = im.show_stack(S(im.side_by_side(impulse, demux + ambient)),
                            "Impulse (left) vs demultiplexed (right)",
                            GAMMA_DISPLAY, fignum=100)
        if save:
            im.save_figure(fig, "09_impulse_vs_demux", outdir)

    allon_stack, _, _, _ = load_stack_dng(
        DATA_PATH, "AllOn", N_ALLON, crop, mode, SUBTRACT_BLACK)
    allon_gold_standard = allon_stack.mean(axis=0)

    # each impulse frame holds one ambient copy; summing N sums N copies,
    # the all-on scene needs only one -> remove the extra (N-1)
    allon_est_impulse = (impulse - ambient).sum(axis=0) + ambient
    if show:
        fig = im.show_stack(
            S(im.side_by_side(allon_gold_standard, allon_est_impulse,
                              allon_est_demux)[None]),
            "All sites on:  gold standard  /  impulse estimate  /  demultiplexed estimate",
            GAMMA_DISPLAY, fignum=101)
        if save:
            im.save_figure(fig, "10_allon_three_way", outdir)

    # -- 8. quantitative evaluation (all-on) ---------------------------
    error_impulse = allon_est_impulse - allon_gold_standard
    error_demux   = allon_est_demux   - allon_gold_standard
    if show:
        # error shown mono in debayer-last (honest per-pixel noise), no gain
        fig = im.show_stack(
            im.side_by_side(error_impulse ** 2, error_demux ** 2)[None] * ERROR_VIS_SCALE,
            "Squared error vs all-on gold standard:  impulse (left)  /  demultiplexed (right)",
            GAMMA_ERROR, fignum=102)
        if save:
            im.save_figure(fig, "11_error_images", outdir)

    mse_impulse = np.mean(error_impulse ** 2)
    mse_demux   = np.mean(error_demux ** 2)
    psnr_impulse_db   = 10.0 * np.log10(1.0 / mse_impulse)      # peak == 1.0
    psnr_demux_db     = 10.0 * np.log10(1.0 / mse_demux)
    psnr_advantage_db = psnr_demux_db - psnr_impulse_db

    # -- 9. single illuminant (hardest case) ---------------------------
    firston_stack, _, _, _ = load_stack_dng(
        DATA_PATH, "FirstOn", N_FIRSTON, crop, mode, SUBTRACT_BLACK)
    first_gold_standard = firston_stack.mean(axis=0)

    first_est_impulse = impulse[0]
    first_est_demux   = demux[0] + ambient
    if show:
        fig = im.show_stack(
            S(im.side_by_side(first_gold_standard, first_est_impulse,
                              first_est_demux)[None]),
            "Site 1 only:  gold standard  /  impulse estimate  /  demultiplexed estimate",
            GAMMA_DISPLAY, fignum=103)
        if save:
            im.save_figure(fig, "12_single_site_three_way", outdir)

    mse_first_impulse = np.mean((first_est_impulse - first_gold_standard) ** 2)
    mse_first_demux   = np.mean((first_est_demux   - first_gold_standard) ** 2)
    psnr_first_impulse_db   = 10.0 * np.log10(1.0 / mse_first_impulse)
    psnr_first_demux_db     = 10.0 * np.log10(1.0 / mse_first_demux)
    psnr_first_advantage_db = psnr_first_demux_db - psnr_first_impulse_db

    # -- 10. report ----------------------------------------------------
    if verbose:
        def prow(label, value, unit=""):
            print(("  %-32s %s %s" % (label, value, unit)).rstrip())
        print("")
        print("=" * 58)
        prow("mode", mode)
        prow("crop (r0, c0, h, w)", str(crop))
        prow("black level (counts)", "%.1f" % black)
        prow("white level (counts)", "%.1f" % white)
        prow("black subtracted", str(SUBTRACT_BLACK))
        prow("display gain", "%.1f" % gain)
        prow("ambient mean level", "%.5f" % ambient.mean())
        prow("ambient std dev", "%.5f" % ambient_stack.std(axis=0).mean())
        print("-" * 58)
        prow("ALL-ON IMAGE", "")
        prow("  MSE, impulse estimate", "%.3e" % mse_impulse)
        prow("  MSE, demultiplexed estimate", "%.3e" % mse_demux)
        prow("  PSNR, impulse estimate", "%.3f" % psnr_impulse_db, "dB")
        prow("  PSNR, demultiplexed estimate", "%.3f" % psnr_demux_db, "dB")
        prow("  PSNR ADVANTAGE", "%.3f" % psnr_advantage_db, "dB")
        print("-" * 58)
        prow("SINGLE-SITE IMAGE (site 1)", "")
        prow("  PSNR, impulse estimate", "%.3f" % psnr_first_impulse_db, "dB")
        prow("  PSNR, demultiplexed estimate", "%.3f" % psnr_first_demux_db, "dB")
        prow("  PSNR ADVANTAGE", "%.3f" % psnr_first_advantage_db, "dB")
        print("=" * 58)
        print("")

    return {
        "crop": crop, "mode": mode,
        "psnr_impulse": psnr_impulse_db, "psnr_demux": psnr_demux_db,
        "advantage": psnr_advantage_db,
        "psnr_first_impulse": psnr_first_impulse_db,
        "psnr_first_demux": psnr_first_demux_db,
        "advantage_first": psnr_first_advantage_db,
    }


# ======================================================================
# Q9 crop sweep: how does the crop region change PSNR and the advantage?
# ======================================================================
def crop_sweep(regions, mode=DEBAYER_MODE):
    print("\nCROP SWEEP (mode = %s)" % mode)
    print("=" * 78)
    print("  %-12s %9s %9s %7s  |  %9s %9s %7s"
          % ("region", "imp dB", "demux dB", "adv dB",
             "1-imp dB", "1-dmx dB", "adv dB"))
    print("-" * 78)
    for name, region in regions:
        r = relight(region, mode=mode, show=False, save=False, verbose=False)
        print("  %-12s %9.2f %9.2f %7.2f  |  %9.2f %9.2f %7.2f"
              % (name, r["psnr_impulse"], r["psnr_demux"], r["advantage"],
                 r["psnr_first_impulse"], r["psnr_first_demux"], r["advantage_first"]))
    print("=" * 78)
    print("all-on PSNRs on the left, single-site (site 1) on the right.\n")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    relight(CROP, mode=DEBAYER_MODE, outdir=OUTDIR, gain=DISPLAY_GAIN)

    if RUN_SWEEP:
        crop_sweep(CROP_SWEEP, mode=DEBAYER_MODE)

    im.plt.show(block=False)
    im.plt.pause(0.2)
    try:
        input("press enter to close the figures ")
    except EOFError:
        im.plt.show(block=True)
    im.plt.close("all")
