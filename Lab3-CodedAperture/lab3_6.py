"""
MECH5720 Lab 3 -- Part 6: PSF extraction, centring, and design comparison.

For each aperture design (Levin, Nayar, and the throughput-matched
circular aperture -- folder "circle", the "Nayar Equivalent" of the
brief) this loads the PSF captures from Part 2 at all five distance
markings (700 / 850 / 1000 / 1150 / 1300 mm), corrects each for the
camera's fixed pattern noise and black level (Part 0), debayers to the
green channel only, locates the PSF by its intensity centroid, and
crops a centred, fixed-size window around it -- no rescaling, per the
brief ("do not scale your captured PSFs in any way").

It then builds a matching, appropriately-scaled copy of that mask's
DESIGN (from the .txt code files pulled from the course repo, plus a
synthesised equal-open-area disc for "circle") for a direct visual
comparison against each captured PSF. See "NOTE on design scaling"
below for how the scale is chosen and why it doubles as a saturation
check.

Centred PSF arrays are saved as .npy (for reuse in Parts 7-9, which
need the same calibrated PSFs) and a per-mask figure grid (captured
PSF over its scaled design, at every depth) is saved for the report.

NOTE on the Levin file set: unlike Nayar/Circle, Levin's Part 2 folder
has no plain "fwd_300.dng". The 700 mm shot for that mask only exists
as "fwd_1500_300.dng" (recaptured at 1500us exposure to avoid
saturation) -- that is what MASKS["levin"]["700mm"] points to below.
Every other file used is the plain-named capture.

NOTE on saturation: most of these captures are heavily to fully
saturated -- e.g. every "700mm" shot is 100% clipped within a 64x64
window, and Levin's is a solid ~230x238 px saturated blob. Centroiding
still lands correctly on the blob (verified against connected-component
analysis), but the true PSF shape is not recoverable from the clipped
region. This is a genuine non-ideality of the captured data (exposure
was not actually low enough to avoid saturation, despite the brief's
instruction) and is worth flagging directly in the Part 6 write-up
(the "what nonidealities are present" question) rather than something
this script can fix after the fact. Each crop's saturated-pixel
fraction is printed and shown in its subplot title so this is visible
at a glance.

NOTE on design scaling: nothing in the brief or the capture metadata
gives an independently-known focal length or mask-to-sensor distance,
so there is no way to compute the "true" projected size of the mask on
the sensor from optics alone. Instead, the scale used to resize each
design is estimated directly from the SAME capture it is compared
against: the pixel bounding box of the captured PSF's main connected
blob (see analyse_psf_blob). This makes the comparison partly circular
by construction (the design is sized to roughly match the capture) --
but the *shape* match is still a meaningful, independent check, and
critically, a saturated capture's blob is inflated by blooming well
beyond the sharp-edged design, so the design will visibly undershoot a
badly saturated capture. That size mismatch is exactly the saturation
signal this comparison is for.
"""
import glob
import os

import numpy as np
import rawpy
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, label, zoom

# ======================================================================
# Configuration
# ======================================================================
DATA_ROOT = "data/lab3/"
DARK_DIR = os.path.join(DATA_ROOT, "part0")
OUTDIR = "out/lab3_part6"

# All five distance markings from Part 2. Filenames are relative to
# data/lab3/<mask>/part2/.
MASKS = {
    "levin": {
        "700mm":  "fwd_1500_300.dng", # TODO Retake this one
        "850mm":  "fwd_150.dng",
        "1000mm": "centre.dng",
        "1150mm": "bck_150.dng",
        "1300mm": "bck_300.dng",
    },
    "nayar": {
        "700mm":  "fwd_300.dng",
        "850mm":  "fwd_150.dng",
        "1000mm": "centre.dng",
        "1150mm": "bck_150.dng",
        "1300mm": "bck_300.dng",
    },
    "circle": {
        "700mm":  "fwd_300.dng",
        "850mm":  "fwd_150.dng",
        "1000mm": "centre.dng",
        "1150mm": "bck_150.dng",
        "1300mm": "bck_300.dng",
    },
}

CROP_SIZE = 128                 # centred PSF window (report asks for e.g. 64x64)
CENTROID_THRESHOLD_FRAC = 0.1  # fraction of (peak - background) kept for the centroid
SATURATION_LEVEL = 0.98        # normalised value counted as "clipped" for diagnostics

# Design files pulled from RoboticImaging/mech5720 (Lab3-CodedAperture/),
# copied into data/lab3/designs/. Each is a 13x13 grid, comma-separated,
# 1 = open, 0 = opaque. "circle" has no file -- it is synthesised below
# as an equal-open-area disc on the same 13x13 grid, per the brief
# ("a circular aperture of equivalent area to the Nayar mask").
DESIGN_DIR = os.path.join(DATA_ROOT, "designs")
DESIGN_FILES = {"levin": "levin_13x13.txt", "nayar": "nayar.txt"}
DESIGN_GRID_SIZE = 13
DESIGN_SUPERSAMPLE = 20   # sub-pixels per grid cell in the design canvas

# ======================================================================
# Raw loading + green-only debayer (bilinear), matching the approach
# already used in lab2_part4.py
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
    """Average >=10 dark frames, each black-level-subtracted first.

    Clipping only happens (if at all) once this map is later subtracted
    from a live capture, not here -- the FPN map itself is left signed.
    """
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
def analyse_psf_blob(green, threshold_frac=CENTROID_THRESHOLD_FRAC):
    """Locate the PSF: intensity-weighted centroid + pixel bounding box
    of its main connected blob.

    The coded-aperture PSFs (Levin/Nayar) are multi-lobed and asymmetric,
    so a simple argmax is not a reliable "centre". Instead: subtract a
    robust background (median), threshold at a fraction of the peak
    above that background, keep only the LARGEST connected component
    (so a stray hot pixel or unrelated bright region elsewhere in frame
    can't drag the centroid or bounding box off target), and take the
    weighted centroid + bounding box of just that component.

    Returns (cy, cx, height, width) -- height/width are the blob's
    pixel bounding-box extent, used later to scale the design mask.
    """
    bg = np.median(green)
    sig = np.clip(green - bg, 0, None)
    thresh = threshold_frac * sig.max()
    mask = sig >= thresh

    labels, n = label(mask)
    if n == 0:
        raise RuntimeError("No PSF blob found above threshold")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background label never wins
    blob = labels == np.argmax(sizes)

    ys, xs = np.nonzero(blob)
    weights = sig[ys, xs]
    cy = float(np.sum(ys * weights) / np.sum(weights))
    cx = float(np.sum(xs * weights) / np.sum(weights))
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return cy, cx, height, width


def crop_centred(img, cy, cx, size):
    """Crop a `size` x `size` window centred on (cy, cx), zero-padding
    at the edges if the window runs off the frame. No resampling/scaling
    -- the window is snapped to the nearest integer pixel.
    """
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
# Per-capture pipeline: raw -> black-level correction -> FPN removal ->
# green-only debayer -> centred crop
# ======================================================================
def extract_psf(path, fpn, colors, black, white, crop_size=CROP_SIZE):
    bayer, _, _, _ = read_raw_bayer(path)
    bayer = bayer - black
    bayer = bayer - fpn
    bayer = bayer / (white - black)          # nominal [0, 1], no clipping (normalisation) (matching lab2_part4)
    green = debayer_green(bayer, colors)
    cy, cx, height, width = analyse_psf_blob(green)
    psf = crop_centred(green, cy, cx, crop_size)
    return psf, (cy, cx), (height, width)


# ======================================================================
# Design masks: load Levin/Nayar codes, synthesise the equal-area
# circle, and scale a design to match a captured blob's pixel extent
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
    """1 = open, 0 = opaque design pattern, nearest-neighbour-upsampled
    to a (grid_size*supersample) square canvas so it can later be
    resized (zoomed) to any target pixel scale without softening the
    mask's hard edges.
    """
    if mask_name in DESIGN_FILES:
        grid = load_design_grid(os.path.join(DESIGN_DIR, DESIGN_FILES[mask_name]))
        return np.kron(grid, np.ones((supersample, supersample)))

    if mask_name == "circle":
        # Equivalent-area circular aperture (brief: "a circular aperture
        # of equivalent area to the Nayar mask") -- same open-cell count
        # as Nayar, drawn as a filled disc on the same grid.
        nayar_grid = load_design_grid(os.path.join(DESIGN_DIR, DESIGN_FILES["nayar"]))
        radius_cells = np.sqrt(nayar_grid.sum() / np.pi)
        n = grid_size * supersample
        yy, xx = np.mgrid[0:n, 0:n]
        centre = (n - 1) / 2.0
        r_px = radius_cells * supersample
        return ((yy - centre) ** 2 + (xx - centre) ** 2 <= r_px ** 2).astype(float)

    raise ValueError("no design for mask %r" % mask_name)


def sensor_px_per_cell(blob_height, blob_width, grid_size=DESIGN_GRID_SIZE):
    """How many sensor pixels one design grid-cell covers, estimated
    from a captured PSF blob's measured pixel extent (see the "NOTE on
    design scaling" module docstring)."""
    return float(np.mean([blob_height, blob_width])) / grid_size


def scaled_design_crop(mask_name, cell_px, crop_size=CROP_SIZE,
                        grid_size=DESIGN_GRID_SIZE, supersample=DESIGN_SUPERSAMPLE):
    """Resize this mask's design canvas so ONE GRID CELL is `cell_px`
    sensor pixels, then centre-crop/pad to `crop_size` -- the SAME
    window as the captured-PSF crop (extract_psf), so the two rows sit
    at an identical physical scale and are genuinely comparable pixel
    for pixel, not just visually similar.

    This means that at a small CROP_SIZE (e.g. 64), the design panel
    can legitimately show little more than a solid black/white block --
    same as the captured-PSF panel above it does when saturated. That
    is not a rendering bug: the measured mask projection is ~160-230px
    across at these depths (see the printed blob=HxW), i.e. genuinely
    larger than a 64px window, on both rows, for the same physical
    reason. Pick a larger CROP_SIZE (e.g. 256) if you want this
    comparison to actually show the code pattern.

    Returns the scaled+cropped design array.
    """
    canvas = build_design_canvas(mask_name, grid_size, supersample)
    zoom_factor = cell_px / supersample
    zoomed = zoom(canvas, zoom_factor, order=0)
    zc_y, zc_x = zoomed.shape[0] / 2.0, zoomed.shape[1] / 2.0
    return crop_centred(zoomed, zc_y, zc_x, crop_size)


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    psfs = {}
    for mask, depths in MASKS.items():
        psfs[mask] = {}
        fig, axes = plt.subplots(2, len(depths), figsize=(3.2 * len(depths), 7.4))
        for col, (depth_label, filename) in enumerate(depths.items()):
            path = os.path.join(DATA_ROOT, mask, "part2", filename)
            psf, (cy, cx), (height, width) = extract_psf(path, fpn, colors, black, white)
            psfs[mask][depth_label] = psf
            np.save(os.path.join(OUTDIR, "psf_%s_%s.npy" % (mask, depth_label)), psf)

            cell_px = sensor_px_per_cell(height, width)
            design = scaled_design_crop(mask, cell_px)

            clipped_pct = 100.0 * np.mean(psf >= SATURATION_LEVEL)

            # Fixed 0-1 scale (not per-panel autoscale): a saturated crop
            # has almost no internal contrast (all values ~1.0 plus tiny
            # noise), so autoscaling would stretch that noise to look
            # like real structure. A shared scale renders it honestly as
            # a flat, clipped block instead.
            ax_top, ax_bot = axes[0, col], axes[1, col]
            ax_top.imshow(psf, cmap="inferno", vmin=0.0, vmax=1.0)
            ax_top.set_title("%s\n%s\n%.0f%% clipped" % (depth_label, filename, clipped_pct))
            ax_top.axis("off")

            ax_bot.imshow(design, cmap="gray", vmin=0.0, vmax=1.0)
            ax_bot.set_title("design (scaled to blob)\n%.1f px/cell" % cell_px)
            ax_bot.axis("off")

            print("%-8s %-8s centroid=(row=%.1f, col=%.1f)  blob=%dx%d px  clipped=%5.1f%%  file=%s"
                  % (mask, depth_label, cy, cx, height, width, clipped_pct, filename))

        fig.suptitle("Captured PSF (top) vs. scaled design (bottom), green channel only -- %s mask" % mask)
        fig.tight_layout(h_pad=4.0)
        fig.subplots_adjust(top=0.88)
        fig.savefig(os.path.join(OUTDIR, "psf_grid_%s.png" % mask), dpi=150)

    plt.show()
