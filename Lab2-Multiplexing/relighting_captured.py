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

The maths in the middle of the file (ambient removal, the H^-1 solve, the
PSNR advantage) is IDENTICAL for both modes: every operation is per-pixel,
so it does not care whether a pixel is one of three colour channels or one
Bayer-masked sample. That is the whole reason debayer-last is possible.

Run it by stepping through in a debugger, watching each figure appear.

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

DEBAYER_MODE = "first"               # "first" (required) or "last" (bonus)

# Crop. The brief tells you to crop before processing -- a 256x256 window
# is plenty. For debayer-LAST the crop origin and size MUST be even, or
# the crop shifts the Bayer phase and the demosaic assumes the wrong
# pattern. None => auto-centred, even-aligned crop of CROP_SIZE.
CROP = None           # or (row0, col0, height, width), all even
CROP_SIZE = 512

# Black level. Set False to run the Q8 experiment (see the report notes):
# leaving the pedestal in shows what happens if you "forget" to remove it.
SUBTRACT_BLACK = True

N_SITES   = 15                       # 5 x 3 illumination grid == 15 frames
N_AMBIENT = 16
N_ALLON   = 16
N_FIRSTON = 16

# Illumination-site layout on the monitor, from capture_relighting.py:
# GRID = (cols, rows) = (5, 3), sites numbered top-left, left-to-right
# along each row, then down. A checkerboard is the sites with (row+col)
# even. For a 5-wide grid that is exactly every second index, [::2].
GRID_COLS, GRID_ROWS = 5, 3

GAMMA_DISPLAY = 2.0
GAMMA_ERROR   = 3.3

im.plt.ion()

# ======================================================================
# Raw DNG loading + bilinear demosaic  (replaces imaging.load_stack,
# which reads PNG/TIFF only)
# ======================================================================
_KG  = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], float) / 4.0
_KRB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 4.0


def _cfa_colors(raw):
    """The 2x2 colour-filter-array layout of this sensor, e.g. BGGR.

    Read from the DNG metadata rather than hard-coded, so the code is
    correct even if your sensor's Bayer phase differs from ours.
    """
    desc = raw.color_desc.decode()          # e.g. 'RGBG'
    pat = raw.raw_pattern                    # 2x2 indices into desc
    return np.array([[desc[pat[i, j]] for j in range(2)] for i in range(2)])


def demosaic_bilinear(bayer, colors):
    """Bilinear demosaic of a single-channel Bayer frame -> (H, W, 3).

    Linear interpolation only: no gamma, no white balance, no clipping.
    The demultiplexing maths needs a linear relationship between light
    and pixel value, so the demosaic must stay linear too.

    NOTE (relevant to Q14): this interpolation is a spatial low-pass. It
    averages neighbouring pixels, which correlates and slightly
    attenuates the per-pixel noise. Debayer-FIRST therefore runs the
    whole pipeline on already-smoothed data; debayer-LAST keeps the true
    per-pixel noise and only smooths at the very end, for display.
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

    Returns (bayer, colors, black, white). The pixels are scaled to a
    nominal [0, 1] by (raw - black) / (white - black). Nothing is
    clamped: a sample below black is a legitimate negative noise
    excursion (exactly as in Lab 1), and clamping it here would bias
    every measurement downstream.
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
    debayer-last it must sit on even boundaries (enforced elsewhere).
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


# In debayer-last, everything is processed as one Bayer channel; colour
# is reconstructed only when an image is handed to a display call.
def to_disp(x, colors):
    """Demosaic for display if we are in debayer-last mode; else pass through."""
    if DEBAYER_MODE != "last" or x.shape[-1] != 1:
        return x
    if x.ndim == 3:
        return demosaic_bilinear(x[..., 0], colors)
    if x.ndim == 4:
        return np.stack([demosaic_bilinear(x[i, ..., 0], colors)
                         for i in range(x.shape[0])])
    return x


# ======================================================================
# 1. The multiplexing matrix  (unchanged -- must match capture time)
# ======================================================================
H = hadamard(N_SITES + 1)
H = H[1:, 1:]
H = (1 - H) / 2

fig = im.show(H[:, :, None], "Multiplexing matrix H", gamma=1.0, fignum=1)
im.save_figure(fig, "01_multiplexing_matrix")
print("Each multiplexed frame has %d of %d sites on." % (H[0].sum(), N_SITES))

# Work out the crop from the first frame's true size, then report the setup.
_probe, _, _bk, _wt = read_dng_bayer(
    _list_frames(DATA_PATH, "Multiplexed", N_SITES)[0], SUBTRACT_BLACK)
crop = CROP if CROP is not None else auto_even_crop(_probe.shape, CROP_SIZE)
assert crop[0] % 2 == 0 and crop[1] % 2 == 0, "crop origin must be even (Bayer phase)"
print("mode = %s   crop = %s   black = %.1f   white = %.1f   subtract_black = %s"
      % (DEBAYER_MODE, crop, _bk, _wt, SUBTRACT_BLACK))

# ======================================================================
# 2. Load and inspect the multiplexed frames
# ======================================================================
mux, colors, BLACK, WHITE = load_stack_dng(
    DATA_PATH, "Multiplexed", N_SITES, crop, DEBAYER_MODE, SUBTRACT_BLACK)
print("Multiplexed stack:", mux.shape)
fig = im.show_stack(to_disp(mux, colors), "Multiplexed input frames",
                    GAMMA_DISPLAY, fignum=2)
im.save_figure(fig, "02_multiplexed_input")

# ======================================================================
# 3. Estimate the ambient illumination (here: the box's residual light
#    plus the black-level pedestal and dark current)
# ======================================================================
ambient_stack, _, _, _ = load_stack_dng(
    DATA_PATH, "Ambient", N_AMBIENT, crop, DEBAYER_MODE, SUBTRACT_BLACK)
ambient = ambient_stack.mean(axis=0)             # (H, W, C) low-noise estimate
fig = im.show(to_disp(ambient, colors),
              "Ambient, averaged over %d frames" % N_AMBIENT,
              GAMMA_DISPLAY, fignum=4)
im.save_figure(fig, "03_ambient_averaged")

# ======================================================================
# 4. Remove the ambient contribution (do NOT clamp -- see Lab 1 / brief)
# ======================================================================
mux = mux - ambient
fig = im.show_stack(to_disp(mux, colors), "Multiplexed input, ambient removed",
                    GAMMA_DISPLAY, fignum=5)
im.save_figure(fig, "04_multiplexed_ambient_removed")

# ======================================================================
# 5. Demultiplex:  mux = H @ single_site  ->  single_site = H^-1 @ mux
# ======================================================================
original_shape = mux.shape
mux_flat = mux.reshape(N_SITES, -1)
demux_flat = np.linalg.solve(H, mux_flat)
demux = demux_flat.reshape(original_shape)
fig = im.show_stack(to_disp(im.side_by_side(mux, demux), colors),
                    "Multiplexed (left) and demultiplexed (right)",
                    GAMMA_DISPLAY, fignum=6)
im.save_figure(fig, "05_mux_vs_demux")

# ======================================================================
# 6. Relight the scene
# ======================================================================
fig = im.show_stack(to_disp(demux + ambient, colors),
                    "Single site illuminated, with ambient",
                    GAMMA_DISPLAY, fignum=7)

# All sites on = sum of every single-site image, plus one ambient.
allon_est_demux = demux.sum(axis=0) + ambient
fig = im.show(to_disp(allon_est_demux, colors), "Synthesised: all sites on",
              GAMMA_DISPLAY, fignum=8)
im.save_figure(fig, "06_synth_allon")

# Checkerboard = sites with (row + col) even. For our 5-wide grid that is
# exactly the even site indices; change this if your grid differs.
site_ids = np.arange(N_SITES)
rows, cols_ = site_ids // GRID_COLS, site_ids % GRID_COLS
checker_indices = site_ids[(rows + cols_) % 2 == 0]
print("checkerboard sites:", checker_indices.tolist())
relit = demux[checker_indices].sum(axis=0) + ambient
fig = im.show(to_disp(relit, colors), "Synthesised: checkerboard illumination",
              GAMMA_DISPLAY, fignum=9)
im.save_figure(fig, "07_synth_checkerboard")

# --- Creative relight (colour required by the brief). Build it from a
#     colour copy of the single-site images so it works in both modes.
demux_col = demux if demux.shape[-1] == 3 else to_disp(demux, colors)
amb_col = ambient if ambient.shape[-1] == 3 else to_disp(ambient, colors)
creative = np.zeros_like(demux_col[0])
creative[:, :, 0] = 1.2 * demux_col[0:5:2, :, :, 0].sum(axis=0)      # warm key, left sites
creative[:, :, 1] = 0.9 * demux_col[6:11, :, :, 1].sum(axis=0)       # green fill, centre
creative[:, :, 2] = 1.6 * demux_col[10::2, :, :, 2].sum(axis=0)      # cool rim, right sites
creative = creative + amb_col / 4.0
fig = im.show(creative, "Creative relighting", GAMMA_DISPLAY, fignum=10)
im.save_figure(fig, "08_creative")

# ======================================================================
# 7. Qualitative comparison against single-site (impulse) capture
# ======================================================================
impulse, _, _, _ = load_stack_dng(
    DATA_PATH, "Impulse", N_SITES, crop, DEBAYER_MODE, SUBTRACT_BLACK)
fig = im.show_stack(to_disp(im.side_by_side(impulse, demux + ambient), colors),
                    "Impulse (left) vs demultiplexed (right)",
                    GAMMA_DISPLAY, fignum=100)
im.save_figure(fig, "09_impulse_vs_demux")

allon_stack, _, _, _ = load_stack_dng(
    DATA_PATH, "AllOn", N_ALLON, crop, DEBAYER_MODE, SUBTRACT_BLACK)
allon_gold_standard = allon_stack.mean(axis=0)

# Impulse all-on estimate. Each impulse frame already contains one copy
# of the ambient, so summing N of them sums N copies; the all-on scene
# should hold only one, so subtract the (N-1) extra copies.
allon_est_impulse = (impulse - ambient).sum(axis=0) + ambient
fig = im.show_stack(
    to_disp(im.side_by_side(allon_gold_standard, allon_est_impulse,
                            allon_est_demux)[None], colors),
    "All sites on:  gold standard  /  impulse estimate  /  demultiplexed estimate",
    GAMMA_DISPLAY, fignum=101)
im.save_figure(fig, "10_allon_three_way")

# ======================================================================
# 8. Quantitative evaluation -- all-on image
# ======================================================================
error_impulse = allon_est_impulse - allon_gold_standard
error_demux   = allon_est_demux   - allon_gold_standard

# Show the error mono in debayer-last (honest per-pixel noise), colour
# in debayer-first.
err_disp_i = error_impulse if DEBAYER_MODE == "last" else to_disp(error_impulse, colors)
err_disp_d = error_demux   if DEBAYER_MODE == "last" else to_disp(error_demux, colors)
fig = im.show_stack(
    im.side_by_side(err_disp_i ** 2, err_disp_d ** 2)[None],
    "Squared error vs all-on gold standard:  impulse (left)  /  demultiplexed (right)",
    GAMMA_ERROR, fignum=102)
im.save_figure(fig, "11_error_images")

mse_impulse = np.mean(error_impulse ** 2)
mse_demux   = np.mean(error_demux ** 2)
psnr_impulse_db   = 10.0 * np.log10(1.0 / mse_impulse)    # peak value == 1.0
psnr_demux_db     = 10.0 * np.log10(1.0 / mse_demux)
psnr_advantage_db = psnr_demux_db - psnr_impulse_db

# ======================================================================
# 9. Same comparison for a single illuminant (the hardest case)
# ======================================================================
firston_stack, _, _, _ = load_stack_dng(
    DATA_PATH, "FirstOn", N_FIRSTON, crop, DEBAYER_MODE, SUBTRACT_BLACK)
first_gold_standard = firston_stack.mean(axis=0)

first_est_impulse = impulse[0]                 # measured directly, incl. ambient
first_est_demux   = demux[0] + ambient         # recovered site 1, ambient added back
fig = im.show_stack(
    to_disp(im.side_by_side(first_gold_standard, first_est_impulse,
                            first_est_demux)[None], colors),
    "Site 1 only:  gold standard  /  impulse estimate  /  demultiplexed estimate",
    GAMMA_DISPLAY, fignum=103)
im.save_figure(fig, "12_single_site_three_way")

mse_first_impulse = np.mean((first_est_impulse - first_gold_standard) ** 2)
mse_first_demux   = np.mean((first_est_demux   - first_gold_standard) ** 2)
psnr_first_impulse_db   = 10.0 * np.log10(1.0 / mse_first_impulse)
psnr_first_demux_db     = 10.0 * np.log10(1.0 / mse_first_demux)
psnr_first_advantage_db = psnr_first_demux_db - psnr_first_impulse_db

# ======================================================================
# 10. Report these numbers
# ======================================================================
def row(label, value, unit=""):
    print(("  %-32s %s %s" % (label, value, unit)).rstrip())

print("")
print("=" * 58)
row("mode", DEBAYER_MODE)
row("crop (r0, c0, h, w)", str(crop))
row("black level (counts)", "%.1f" % BLACK)
row("white level (counts)", "%.1f" % WHITE)
row("black subtracted", str(SUBTRACT_BLACK))
row("ambient mean level", "%.5f" % ambient.mean())
row("ambient std dev", "%.5f" % ambient_stack.std(axis=0).mean())
print("-" * 58)
row("ALL-ON IMAGE", "")
row("  MSE, impulse estimate", "%.3e" % mse_impulse)
row("  MSE, demultiplexed estimate", "%.3e" % mse_demux)
row("  PSNR, impulse estimate", "%.3f" % psnr_impulse_db, "dB")
row("  PSNR, demultiplexed estimate", "%.3f" % psnr_demux_db, "dB")
row("  PSNR ADVANTAGE", "%.3f" % psnr_advantage_db, "dB")
print("-" * 58)
row("SINGLE-SITE IMAGE (site 1)", "")
row("  PSNR, impulse estimate", "%.3f" % psnr_first_impulse_db, "dB")
row("  PSNR, demultiplexed estimate", "%.3f" % psnr_first_demux_db, "dB")
row("  PSNR ADVANTAGE", "%.3f" % psnr_first_advantage_db, "dB")
print("=" * 58)
print("")

im.plt.show(block=False)
im.plt.pause(0.2)
try:
    input("press enter to close the figures ")
except EOFError:
    im.plt.show(block=True)
im.plt.close("all")
