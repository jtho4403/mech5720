"""
MECH5720 Lab 3 -- Part 7: MTF via the Siemens Star target.

For each mask's Part 3 capture (Siemens Star, furthest distance marking,
same focus as Part 1), corrects for fixed pattern noise and black level
(Part 0) and debayers to the green channel only (reusing read_raw_bayer,
debayer_green and compute_fixed_pattern_noise from lab3_6.py), then:

1. Locates the star's centre and the largest radius that stays inside
   the frame at every angle (the star's true printed radius extends
   past the top/left/bottom edges in these captures -- only the
   right-hand side is fully visible -- so the usable radius for a full
   360 degree sweep is bounded by the nearest frame edge, not by the
   target itself).
2. Samples the green channel around concentric circles by bilinear
   interpolation (scipy.ndimage.map_coordinates, order=1) at angles
   evenly spaced around each circle.
3. Estimates the star's spoke count N from the (radius-independent)
   number of bright wedges resolved near the outer, best-resolved
   radii -- needed for the f(r) = N / (2*pi*r) frequency mapping.
4. Computes the Michelson contrast C(r) = (Imax - Imin) / (Imax + Imin)
   at each radius from the mean peak/trough intensity around that
   circle (scipy.signal.find_peaks on the angular profile), and treats
   C(r) as the MTF magnitude at f(r).

Radii are swept log-spaced from near the centre (highest frequency) out
to the frame-limited maximum (lowest frequency), per the brief.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, uniform_filter, uniform_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from lab3_6 import read_raw_bayer, debayer_green, compute_fixed_pattern_noise

# ======================================================================
# Configuration
# ======================================================================
DARK_DIR = "data/lab3/part0"
OUTDIR = "out/lab3_part7"

MASKS = {
    "circle": "data/lab3/circle/part3/mtf.dng",
    "levin":  "data/lab3/levin/part3/mtf.dng",
    "nayar":  "data/lab3/nayar/part3/mtf.dng",
}

# Search box for the star centre, as a fraction of (height, width). The
# star sits left-of-centre in every capture here (camera/target were not
# re-centred between masks) -- this box comfortably contains all three
# measured centres while excluding the blank background at the frame's
# right edge, which would otherwise win a naive global search (a large
# flat region is "lower contrast" than the star's own centre hub).
CENTRE_SEARCH_BOX_FRAC = (0.30, 0.65, 0.20, 0.50)  # (row0, row1, col0, col1)
RADIUS_MARGIN_PX = 10          # keep every sampled circle strictly inside the frame
N_THETA = 2880                 # angular samples per circle (0.125 deg resolution)
SPOKE_COUNT_RADIUS_FRACS = (0.80, 0.85, 0.90, 0.95, 0.99)  # outer radii used to count spokes
DEMO_RADIUS_FRACS = {"centre": 0.15, "middle": 0.50, "edge": 0.90}
N_MTF_RADII = 40
R_MIN_FRAC = 0.03               # avoid the sub-pixel-blended hub at r ~ 0


# ======================================================================
# Star centre + radius
# ======================================================================
def find_star_centre(green, box_frac=CENTRE_SEARCH_BOX_FRAC, small_win=9, smooth_win=25, refine_half=60):
    """Locate the Siemens Star's centre.

    Right at the centre, the wedges narrow below the optical/sensor
    resolution and blend into a smooth low-contrast hub, so the centre
    is the local MINIMUM of local intensity contrast -- everywhere else
    on the star, the alternating wedges are locally high-contrast. A
    coarse minimum is found on a smoothed local-std map restricted to
    box_frac (to exclude the blank background, which is even lower
    contrast over a much larger area and would otherwise win), then
    refined to sub-pixel precision as an inverse-contrast-weighted
    centroid in a small window around that coarse point.
    """
    h, w = green.shape
    mean = uniform_filter(green, small_win)
    meansq = uniform_filter(green ** 2, small_win)
    localstd = np.sqrt(np.clip(meansq - mean ** 2, 0, None))
    smoothed = uniform_filter(localstd, smooth_win)

    r0f, r1f, c0f, c1f = box_frac
    r0, r1 = int(r0f * h), int(r1f * h)
    c0, c1 = int(c0f * w), int(c1f * w)
    region = smoothed[r0:r1, c0:c1]
    idx = np.unravel_index(np.argmin(region), region.shape)
    cy0, cx0 = idx[0] + r0, idx[1] + c0

    r0r, r1r = cy0 - refine_half, cy0 + refine_half
    c0r, c1r = cx0 - refine_half, cx0 + refine_half
    patch = smoothed[r0r:r1r, c0r:c1r]
    weight = 1.0 / (patch + 1e-4)
    weight -= weight.min()
    ys, xs = np.mgrid[r0r:r1r, c0r:c1r]
    cy = float(np.sum(ys * weight) / np.sum(weight))
    cx = float(np.sum(xs * weight) / np.sum(weight))
    return cy, cx


def max_valid_radius(cy, cx, h, w, margin=RADIUS_MARGIN_PX):
    """Largest radius whose full 360 degree circle stays inside the frame."""
    return min(cy, cx, h - 1 - cy, w - 1 - cx) - margin


# ======================================================================
# Circle sampling (bilinear interpolation) + contrast
# ======================================================================
def sample_circle(green, cy, cx, r, n_theta=N_THETA):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    vals = map_coordinates(green, [y, x], order=1, mode="nearest")
    return theta, vals


def estimate_spoke_count(green, cy, cx, r_max, fracs=SPOKE_COUNT_RADIUS_FRACS,
                          n_theta=N_THETA, smooth_size=9):
    """Count the star's bright wedges (N in f(r) = N / 2*pi*r) at several
    large, well-resolved radii and take the median for robustness."""
    counts = []
    for frac in fracs:
        _, vals = sample_circle(green, cy, cx, frac * r_max, n_theta)
        smooth = uniform_filter1d(vals, size=smooth_size, mode="wrap")
        tiled = np.concatenate([smooth, smooth, smooth])
        peaks, _ = find_peaks(tiled, distance=n_theta // 60)
        peaks = peaks[(peaks >= n_theta) & (peaks < 2 * n_theta)]
        counts.append(len(peaks))
    return int(np.median(counts))


def circle_contrast(vals, n_theta=N_THETA, smooth_size=9, min_distance_frac=60):
    """Michelson contrast from the mean peak / trough intensity around one
    circle. A light smoothing pass rejects sensor read noise before peak
    finding (angular sampling is dense: N_THETA/spoke_count samples per
    period at every radius, since sampling is angular not radial, so this
    does not blur out genuine wedge structure)."""
    smooth = uniform_filter1d(vals, size=smooth_size, mode="wrap")
    tiled = np.concatenate([smooth, smooth, smooth])
    distance = max(1, n_theta // min_distance_frac)
    peak_idx, _ = find_peaks(tiled, distance=distance)
    trough_idx, _ = find_peaks(-tiled, distance=distance)
    peak_idx = peak_idx[(peak_idx >= n_theta) & (peak_idx < 2 * n_theta)] - n_theta
    trough_idx = trough_idx[(trough_idx >= n_theta) & (trough_idx < 2 * n_theta)] - n_theta
    if len(peak_idx) == 0 or len(trough_idx) == 0:
        return np.nan, np.nan, np.nan
    i_max = float(np.mean(vals[peak_idx]))
    i_min = float(np.mean(vals[trough_idx]))
    contrast = (i_max - i_min) / (i_max + i_min)
    return i_max, i_min, contrast


def mtf_curve(green, cy, cx, r_max, spoke_count, n_radii=N_MTF_RADII,
              r_min_frac=R_MIN_FRAC, n_theta=N_THETA):
    radii = np.geomspace(r_min_frac * r_max, r_max, n_radii)
    contrasts = np.full(n_radii, np.nan)
    for i, r in enumerate(radii):
        _, vals = sample_circle(green, cy, cx, r, n_theta)
        _, _, c = circle_contrast(vals, n_theta)
        contrasts[i] = c
    freqs = spoke_count / (2 * np.pi * radii)   # cycles/pixel
    return radii, freqs, contrasts


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    fpn, colors, black, white = compute_fixed_pattern_noise(DARK_DIR)

    results = {}
    for mask, path in MASKS.items():
        bayer, _, _, _ = read_raw_bayer(path)
        bayer = (bayer - black - fpn) / (white - black)
        green = debayer_green(bayer, colors)
        h, w = green.shape

        cy, cx = find_star_centre(green)
        r_max = max_valid_radius(cy, cx, h, w)
        spoke_count = estimate_spoke_count(green, cy, cx, r_max)
        radii, freqs, contrasts = mtf_curve(green, cy, cx, r_max, spoke_count)

        results[mask] = dict(green=green, cy=cy, cx=cx, r_max=r_max,
                              spoke_count=spoke_count, radii=radii,
                              freqs=freqs, contrasts=contrasts)
        np.savez(os.path.join(OUTDIR, "mtf_%s.npz" % mask),
                 radii=radii, freqs=freqs, contrasts=contrasts,
                 cy=cy, cx=cx, r_max=r_max, spoke_count=spoke_count)

        print("%-8s centre=(row=%.1f, col=%.1f)  r_max=%.1f px  spokes(N)=%d"
              % (mask, cy, cx, r_max, spoke_count))

    # ------------------------------------------------------------------
    # Figure A: intensity vs angle at three radii (centre/middle/edge) --
    # one standalone figure per mask.
    # ------------------------------------------------------------------
    for mask, res in results.items():
        fig_a, ax = plt.subplots(figsize=(7, 4.5))
        for label, frac in DEMO_RADIUS_FRACS.items():
            r = frac * res["r_max"]
            theta, vals = sample_circle(res["green"], res["cy"], res["cx"], r)
            ax.plot(np.degrees(theta), vals, label="%s (r=%.0fpx)" % (label, r), linewidth=1)
        ax.set_title("Siemens Star intensity vs. angle -- %s" % mask)
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("green intensity")
        ax.set_xlim(0, 360)
        ax.legend(fontsize=8)
        fig_a.tight_layout()
        fig_a.savefig(os.path.join(OUTDIR, "intensity_vs_angle_%s.png" % mask), dpi=150)

    # ------------------------------------------------------------------
    # Figure B: MTF vs frequency, all masks overlaid
    # ------------------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    for mask, res in results.items():
        ax_b.plot(res["freqs"], res["contrasts"], marker="o", markersize=3, label=mask)
    ax_b.set_xlabel("spatial frequency (cycles/pixel)")
    ax_b.set_ylabel("contrast (MTF magnitude)")
    ax_b.set_xscale("log")
    ax_b.set_title("MTF from Siemens Star, per mask")
    ax_b.legend()
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(OUTDIR, "mtf_vs_frequency.png"), dpi=150)

    plt.show()
