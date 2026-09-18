"""
MECH5720 Lab 3 -- Part 6: PSF extraction, centring, and design comparison.

For each mask (Levin, Nayar, circle) this loads the Part 2 PSF captures, corrects for fixed pattern noise and black level, debayers to green only, centres the PSF, and crops it -- no rescaling of the capture itself.
It also builds a matching, appropriately-scaled copy of that mask's design (from the course-repo .txt code files, plus a synthesised equal-area disc for "circle") for a direct visual comparison against each captured PSF.
Centred PSF arrays are saved as .npy for reuse in Parts 7-9, and a per-mask figure grid (captured PSF over its scaled design, at every depth) is saved for the report.
At each mask's maximum distance only, the captured PSF and its scaled design are also zero-padded to at least FOURIER_PAD_SIZE, FFT'd, magnitude-normalised to their own DC, and plotted side by side in dB (fourier_<mask>_<depth_label>.png).

NOTE: Levin's 700mm shot has no plain "fwd_300.dng" -- it only exists as "fwd_1500_300.dng" (recaptured at 1500us to avoid saturation).
NOTE: most captures are heavily to fully saturated (see the printed/plotted clipped%), a genuine non-ideality worth flagging in the write-up rather than something this script can fix.
NOTE: since no focal length or mask-to-sensor distance is known, the design's scale is estimated from the capture's own measured blob size -- partly circular by construction, but still a meaningful shape/size check, and a saturated blob's inflation from blooming is itself informative.
"""
import glob
import os

import numpy as np
import rawpy
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import convolve, label

# ======================================================================
# Configuration
# ======================================================================
DATA_ROOT = "data/lab3_2"
DARK_DIR = os.path.join(DATA_ROOT, "part0")
OUTDIR = "out/lab3_2_part6"

# Distance markings from Part 2; filenames relative to data/lab3/<mask>/part2/.
MASKS = {
    "levin": {
        # "700mm":  "fwd_300.dng",
        # "850mm":  "fwd_150.dng",
        "1000mm": "centre.dng",
        "1150mm": "back_150.dng",
        "1300mm": "back_300.dng",
    },
    "nayar": {
        # "700mm":  "fwd_300.dng",
        # "850mm":  "fwd_150.dng",
        "1000mm": "centre.dng",
        "1150mm": "back_150.dng",
        "1300mm": "back_300.dng",
    },
    "circle": {
        # "700mm":  "fwd_300.dng",
        # "850mm":  "fwd_150.dng",
        "1000mm": "centre.dng",
        "1150mm": "back_150.dng",
        "1300mm": "back_300.dng",
    },
}

CROP_SIZE = 64 # 128                 # centred PSF window (report asks for e.g. 64x64)
CENTROID_THRESHOLD_FRAC = 0.1 # 0.1  # fraction of (peak - background) kept for the centroid
SATURATION_LEVEL = 0.98        # normalised value counted as "clipped" for diagnostics

# Padded FFT size and dB floor for the Fourier-domain comparison (Part 6).
FOURIER_PAD_SIZE = 256
FOURIER_DB_FLOOR = -80.0 # 80.0, 60.0

# Threshold (of peak-above-background) used to measure blob SIZE for design scaling -- higher than CENTROID_THRESHOLD_FRAC so a defocus halo doesn't inflate it.
SIZE_THRESHOLD_FRAC = 0.5 # 0.5 # 0.3 # 0.5

# PSF search is restricted to a centred window this large (px), since the point source is always placed near frame centre.
SEARCH_WINDOW_SIZE = 800

# 13x13 code files pulled from RoboticImaging/mech5720 (Lab3-CodedAperture/); "circle" has no file and is synthesised as an equal-open-area disc.
DESIGN_DIR = os.path.join(DATA_ROOT, "designs")
DESIGN_FILES = {"levin": "levin_13x13.txt", "nayar": "nayar.txt"}
DESIGN_GRID_SIZE = 13
DESIGN_SUPERSAMPLE = 2 # 5 # 10 # 1 # 20   # sub-pixels per grid cell in the design canvas

# ======================================================================
# Raw loading + green-only debayer, matching lab2_part4.py
# ======================================================================
_KG = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], float) / 4.0


def _cfa_colors(raw):
    """The 2x2 colour-filter-array layout of this sensor, e.g. BGGR."""
    desc = raw.color_desc.decode()
    pat = raw.raw_pattern
    return np.array([[desc[pat[i, j]] for j in range(2)] for i in range(2)])


def debayer_green(bayer, colors):
    """Bilinear-interpolate just the green plane of a Bayer frame -> (H, W)."""
    h, w = bayer.shape
    mask2x2 = (colors == "G")
    reps = ((h + 1) // 2 + 1, (w + 1) // 2 + 1)
    full_mask = np.tile(mask2x2, reps)[:h, :w]
    samp = np.where(full_mask, bayer, 0.0)
    return convolve(samp, _KG, mode="mirror")


def read_raw_bayer(path):
    """Read one DNG's raw Bayer frame plus its black/white levels and CFA."""
    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.astype(np.float64)
        colors = _cfa_colors(raw)
        black = float(np.mean(raw.black_level_per_channel))
        white = float(raw.white_level)
    return bayer, colors, black, white


# ======================================================================
# Part 0: fixed pattern noise
# ======================================================================
def compute_fixed_pattern_noise(dark_dir):
    """Average >=10 dark frames, each black-level-subtracted first."""
    files = sorted(glob.glob(os.path.join(dark_dir, "dark*.dng")))
    if not files:
        raise FileNotFoundError("No dark*.dng frames found in %s" % dark_dir)

    frames = []
    colors = black = white = None
    for f in files:
        bayer, colors, black, white = read_raw_bayer(f)
        frames.append(bayer - black)
    fpn = np.mean(frames, axis=0)
    print("Fixed pattern noise: averaged %d dark frames from %s" % (len(files), dark_dir))
    return fpn, colors, black, white


# ======================================================================
# PSF centring
# ======================================================================
def analyse_psf_blob(green, threshold_frac=CENTROID_THRESHOLD_FRAC,
                      size_threshold_frac=SIZE_THRESHOLD_FRAC,
                      search_size=SEARCH_WINDOW_SIZE):
    """Within a centred search window, locate the connected blob containing the brightest pixel and return its weighted centroid plus a (separately, more tightly thresholded) bounding-box size.

    Returns (cy, cx, height, width) in FULL-FRAME coordinates.
    """
    h, w = green.shape
    if search_size is not None:
        r0 = max(0, h // 2 - search_size // 2)
        c0 = max(0, w // 2 - search_size // 2)
        r1 = min(h, r0 + search_size)
        c1 = min(w, c0 + search_size)
        region = green[r0:r1, c0:c1]
    else:
        region = green
        r0 = c0 = 0

    bg = np.median(region)
    sig = np.clip(region - bg, 0, None)
    thresh = threshold_frac * sig.max()
    mask = sig >= thresh

    labels, n = label(mask)
    if n == 0:
        raise RuntimeError("No PSF blob found above threshold")
    peak_label = labels[np.unravel_index(np.argmax(sig), sig.shape)]
    blob = labels == peak_label

    ys, xs = np.nonzero(blob)
    weights = sig[ys, xs]
    cy = float(np.sum(ys * weights) / np.sum(weights)) + r0
    cx = float(np.sum(xs * weights) / np.sum(weights)) + c0

    size_mask = blob & (sig >= size_threshold_frac * sig.max())
    if not size_mask.any():
        size_mask = blob  # degenerate capture: fall back to the full blob
    ys2, xs2 = np.nonzero(size_mask)
    height = int(ys2.max() - ys2.min() + 1)
    width = int(xs2.max() - xs2.min() + 1)
    return cy, cx, height, width


def crop_centred(img, cy, cx, size):
    """Crop/zero-pad a `size` x `size` window centred on (cy, cx), snapped to the nearest pixel with no resampling."""
    r0 = int(round(cy)) - size // 2
    c0 = int(round(cx)) - size // 2
    out = np.zeros((size, size), dtype=img.dtype)

    src_r0, src_r1 = max(r0, 0), min(r0 + size, img.shape[0])
    src_c0, src_c1 = max(c0, 0), min(c0 + size, img.shape[1])
    dst_r0, dst_c0 = src_r0 - r0, src_c0 - c0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    out[dst_r0:dst_r1, dst_c0:dst_c1] = img[src_r0:src_r1, src_c0:src_c1]
    return out


# ======================================================================
# Per-capture pipeline: raw -> black-level -> FPN -> green debayer -> crop
# ======================================================================
def extract_psf(path, fpn, colors, black, white, crop_size=CROP_SIZE):
    bayer, _, _, _ = read_raw_bayer(path)
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)  # nominal [0, 1]
    green = debayer_green(bayer, colors)
    cy, cx, height, width = analyse_psf_blob(green)
    psf = crop_centred(green, cy, cx, crop_size)
    return psf, (cy, cx), (height, width)


# ======================================================================
# Design masks: load codes, synthesise the equal-area circle, and scale
# ======================================================================
def load_design_grid(path):
    """Read a comma-separated 13x13 code file -> float array, 1=open."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([int(v) for v in line.rstrip(",").split(",")])
    return np.array(rows, dtype=float)


def build_design_canvas(mask_name, grid_size=DESIGN_GRID_SIZE, supersample=DESIGN_SUPERSAMPLE):
    """Design pattern (1=open), nearest-neighbour-upsampled to a crisp (grid_size*supersample) square canvas."""
    if mask_name in DESIGN_FILES:
        grid = load_design_grid(os.path.join(DESIGN_DIR, DESIGN_FILES[mask_name]))
        return np.kron(grid, np.ones((supersample, supersample)))

    if mask_name == "circle":
        # Equal-open-area disc matching Nayar's open-cell count, per the brief.
        nayar_grid = load_design_grid(os.path.join(DESIGN_DIR, DESIGN_FILES["nayar"]))
        radius_cells = np.sqrt(nayar_grid.sum() / np.pi)
        n = grid_size * supersample
        yy, xx = np.mgrid[0:n, 0:n]
        centre = (n - 1) / 2.0
        r_px = radius_cells * supersample
        return ((yy - centre) ** 2 + (xx - centre) ** 2 <= r_px ** 2).astype(float)

    raise ValueError("no design for mask %r" % mask_name)


def sensor_px_per_cell(blob_height, blob_width, grid_size=DESIGN_GRID_SIZE):
    """Sensor pixels per design grid-cell, estimated from a captured PSF blob's measured extent."""
    return float(np.mean([blob_height, blob_width])) / grid_size


def scaled_design_crop(mask_name, cell_px, crop_size=CROP_SIZE,
                        grid_size=DESIGN_GRID_SIZE, supersample=DESIGN_SUPERSAMPLE):
    """Build the design canvas, zoom it so one grid cell is `cell_px` sensor pixels (anti-aliased when shrinking), then crop/pad to `crop_size` -- the same window as the captured-PSF crop, so the two are directly comparable."""
    mask = build_design_canvas(mask_name, grid_size, supersample)
    zoom_factor = cell_px / supersample

    # if zoom_factor < 1.0:
    #     # Anti-alias pre-filter before shrinking, else ndimage.zoom aliases into jagged noise.
    #     sigma = (1.0 / zoom_factor - 1.0) / 2.0
    #     mask = ndimage.gaussian_filter(mask, sigma=sigma)

    scaled_mask = ndimage.zoom(mask, zoom_factor) # , order=1)
    zc_y, zc_x = scaled_mask.shape[0] / 2.0, scaled_mask.shape[1] / 2.0
    return crop_centred(scaled_mask, zc_y, zc_x, crop_size)


# ======================================================================
# Fourier-domain comparison (max distance only)
# ======================================================================
def pad_to_size(img, size):
    """Zero-pad a square `img` to at least `size` x `size`, centred, without cropping or resampling."""
    n = max(size, img.shape[0], img.shape[1])
    out = np.zeros((n, n), dtype=img.dtype)
    r0 = (n - img.shape[0]) // 2
    c0 = (n - img.shape[1]) // 2
    out[r0:r0 + img.shape[0], c0:c0 + img.shape[1]] = img
    return out


def fourier_magnitude_normalized(img, pad_size=FOURIER_PAD_SIZE):
    """Pad, FFT, shift DC to centre, and normalise the magnitude to its own peak (linear, not dB)."""
    # Pad to 256x256
    padded = pad_to_size(img, pad_size)
    # Fourier Transform
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    # Normalise 
    mag = np.abs(spectrum)
    return mag / mag.max()


def fourier_magnitude_db(img, pad_size=FOURIER_PAD_SIZE, db_floor=FOURIER_DB_FLOOR):
    """Pad, FFT, shift DC to centre, normalise magnitude to its own peak (DC -> 0 dB), and convert to dB."""
    mag_norm = fourier_magnitude_normalized(img, pad_size)
    floor_lin = 10.0 ** (db_floor / 20.0)
    return 20.0 * np.log10(np.maximum(mag_norm, floor_lin))


def plot_fourier_comparison(mask, depth_label, filename, psf, design, outdir):
    """Side-by-side dB magnitude spectra of the captured PSF and its scaled design, saved as fourier_<mask>_<depth_label>.png."""
    psf_db = fourier_magnitude_db(psf)
    design_db = fourier_magnitude_db(design)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    im0 = axes[0].imshow(psf_db, cmap="viridis", vmin=FOURIER_DB_FLOOR, vmax=0.0)
    axes[0].set_title("Captured PSF") # \n%s" % filename)
    axes[1].imshow(design_db, cmap="viridis", vmin=FOURIER_DB_FLOOR, vmax=0.0)
    axes[1].set_title("Scaled Design")
    for ax in axes:
        # ax.set_xlabel("u (cycles / %d px)" % psf_db.shape[1])
        # ax.set_ylabel("v (cycles / %d px)" % psf_db.shape[0])
        ax.set_xlabel("Cycles / Pixel")
        ax.set_ylabel("Cycles / Pixel")

    fig.suptitle("Normalised Fourier Magnitude Response (2D) -- %s mask" % mask)
    fig.colorbar(im0, ax=axes, label="dB relative to DC value", shrink=0.85)
    fig.savefig(os.path.join(outdir, "fourier_%s_%s.png" % (mask, depth_label)), dpi=150)


def radial_average(mag_linear):
    """Azimuthal average of a centred 2D array -> 1D profile indexed by integer pixel radius from centre."""
    n = mag_linear.shape[0]
    cy = cx = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r_int = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    profile = np.bincount(r_int.ravel(), weights=mag_linear.ravel())
    counts = np.bincount(r_int.ravel())
    return profile / np.maximum(counts, 1)


def radial_magnitude_db(img, pad_size=FOURIER_PAD_SIZE, db_floor=FOURIER_DB_FLOOR):
    """Radially-averaged magnitude (dB) and matching spatial-frequency axis (cycles/pixel) for one PSF/design array."""
    profile = radial_average(fourier_magnitude_normalized(img, pad_size))
    freq = np.arange(len(profile)) / float(pad_size)  # Nyquist = 0.5 cycles/pixel
    floor_lin = 10.0 ** (db_floor / 20.0)
    db = 20.0 * np.log10(np.maximum(profile, floor_lin))
    return freq, db


MASK_COLORS = {"levin": "tab:blue", "nayar": "tab:orange", "circle": "tab:green"}


def plot_radial_magnitude(radial_data, outdir):
    """One combined plot of radial magnitude (dB) vs spatial frequency for every mask -- solid = design, dashed = captured."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for mask, (freq, psf_db, design_db) in radial_data.items():
        color = MASK_COLORS.get(mask)
        ax.plot(freq, design_db, linestyle="-", color=color, label="%s (design)" % mask)
        ax.plot(freq, psf_db, linestyle="--", color=color, label="%s (captured)" % mask)
    ax.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax.set_ylabel("Magnitude (dB relative to own DC)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(FOURIER_DB_FLOOR, 5)
    ax.set_title("1D Radial Magnitude Response at Max Distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "radial_magnitude_all_masks.png"), dpi=150)


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    psfs = {}
    radial_data = {}  # mask -> (freq, captured_db, design_db), for the combined 1D plot
    for mask, depths in MASKS.items():
        psfs[mask] = {}
        # fig, axes = plt.subplots(2, len(depths), figsize=(3.2 * len(depths), 7.4), squeeze=False)
        fig, axes = plt.subplots(2, len(depths), figsize=(3.2 * len(depths), 7.4), squeeze=False)

        max_dist_data = None  # (distance_mm, depth_label, filename, psf, design) for the furthest depth only
        for col, (depth_label, filename) in enumerate(depths.items()):
            path = os.path.join(DATA_ROOT, mask, "part2", filename)
            psf, (cy, cx), (height, width) = extract_psf(path, fpn, colors, black, white)
            psfs[mask][depth_label] = psf
            np.save(os.path.join(OUTDIR, "psf_%s_%s.npy" % (mask, depth_label)), psf)

            cell_px = sensor_px_per_cell(height, width)
            design = scaled_design_crop(mask, cell_px)

            # Brief only requires the Fourier comparison at the max distance -- parsed from the label so it's correct regardless of dict order.
            distance_mm = int(depth_label.rstrip("mm"))
            if max_dist_data is None or distance_mm > max_dist_data[0]:
                max_dist_data = (distance_mm, depth_label, filename, psf, design)

            clipped_pct = 100.0 * np.mean(psf >= SATURATION_LEVEL)

            # Per-panel autoscale (own peak) keeps the PSF visible across very different brightness settings.
            ax_top, ax_bot = axes[0, col], axes[1, col]
            ax_top.imshow(psf, cmap="gray", vmin=0.0, vmax=psf.max()) # inferno
            ax_top.set_title("%s\nCaptured PSF" % depth_label, fontsize=12)
            ax_top.axis("off")

            ax_bot.imshow(design, cmap="gray", vmin=0.0, vmax=1.0)
            ax_bot.set_title("Scaled Design", fontsize=12)
            ax_bot.axis("off")

            print("%-8s %-8s centroid=(row=%.1f, col=%.1f)  blob=%dx%d px  clipped=%5.1f%%  file=%s"
                  % (mask, depth_label, cy, cx, height, width, clipped_pct, filename))

        
        fig.suptitle("Captured PSF (top) vs Scaled Design (bottom) - %s mask\n" % mask, fontsize=16)
        fig.tight_layout(h_pad=4.0)
        fig.savefig(os.path.join(OUTDIR, "psf_grid_%s.png" % mask), dpi=150)

        _, max_depth_label, max_filename, max_psf, max_design = max_dist_data
        plot_fourier_comparison(mask, max_depth_label, max_filename, max_psf, max_design, OUTDIR)

        freq, psf_radial_db = radial_magnitude_db(max_psf)
        _, design_radial_db = radial_magnitude_db(max_design)
        radial_data[mask] = (freq, psf_radial_db, design_radial_db)

    plot_radial_magnitude(radial_data, OUTDIR)
    plt.show()
