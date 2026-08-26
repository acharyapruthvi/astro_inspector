#!/usr/bin/env python3
"""
FITS Astrophotography Quality Inspector (with plots)
====================================================

Loads all FITS light frames from a folder, measures per-frame quality
metrics, combines them into a single 0-100 quality score
(percentile-ranked within the batch) based on three criteria:

  1. Star shape/count - weighted FWHM (FWHM with a star-count penalty,
                        Siril-style) + eccentricity (roundness)
  2. Background level - median sky background (lower = better)
  3. Noise level      - background standard deviation (lower = better)

The best frames (top X% by quality score, default 80%) are COPIED to a
"best frames" folder by default (see --good-dir / --top-percent /
--no-export / --move), organised as:

    <good-dir>/<date>/LIGHTS/<original_filename>.fits

The raw FLAT/DARK/DARKFLAT/BIAS frames are left in place and are NOT
copied unless --copy-calibration is passed.

Outputs:
  1. quality_report.csv         - full metric table, ranked best-first
  2. quality_scores.png         - ranked bar chart of scores per frame
  3. metric_distributions.png   - histograms of each metric
  4. fwhm_vs_score.png          - FWHM vs quality score scatter
  5. metrics_over_sequence.png  - metrics vs frame order (drift detection)
  6. <good-dir>/<date>/LIGHTS/  - copies of the highest-quality frames
  7. <good-dir>/<date>/<CALIB>/ - raw calibration frames, only when
                                  --copy-calibration is passed

Note: frames are analyzed as-is (no bias/flat calibration). Run on
already-calibrated frames, or use it directly on raw lights for a quick
relative ranking.

Requirements:
    pip install astropy numpy pandas matplotlib photutils

Usage:
    python fits_quality_plots.py --lights ./lights --outdir ./quality_report
"""

import argparse
import glob
import os
import re
import sys
import warnings
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings("ignore")

try:
    from photutils.detection import DAOStarFinder
except ImportError:
    sys.exit("This script requires photutils. Install with: pip install photutils")


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def list_fits(folder, recursive=False):
    """List FITS files in a folder. With recursive=True, also descend into
    any subdirectories (e.g. $DATE/LIGHTS/<target>/*.fits)."""
    exts = ("*.fits", "*.fit", "*.fts", "*.FIT", "*.FITS")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
        if recursive:
            files.extend(glob.glob(os.path.join(folder, "**", e),
                                    recursive=True))
    return sorted(set(files))


def load_data(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
    return data, header


# --------------------------------------------------------------------------
# Star detection & shape measurement
# --------------------------------------------------------------------------

def detect_stars(data, fwhm_guess=4.0, thresh_sigma=5.0):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm_guess, threshold=thresh_sigma * std)
    sources = finder(data - median)
    return sources, median, std


def star_moments(data, x, y, box=15, bg=0.0):
    """Flux-weighted second moments -> FWHM (px), eccentricity, peak."""
    half = box // 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - half), min(data.shape[0], yi + half + 1)
    x0, x1 = max(0, xi - half), min(data.shape[1], xi + half + 1)
    cutout = data[y0:y1, x0:x1].astype(np.float64) - bg
    if cutout.size == 0:
        return None
    cutout = np.clip(cutout, 0, None)
    total = cutout.sum()
    if total <= 0:
        return None

    yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
    xc = (xx * cutout).sum() / total
    yc = (yy * cutout).sum() / total

    dx, dy = xx - xc, yy - yc
    ixx = (cutout * dx * dx).sum() / total
    iyy = (cutout * dy * dy).sum() / total
    ixy = (cutout * dx * dy).sum() / total

    eigvals = np.linalg.eigvalsh(np.array([[ixx, ixy], [ixy, iyy]]))
    eigvals = np.clip(eigvals, 1e-6, None)
    sigma_major = np.sqrt(eigvals.max())
    sigma_minor = np.sqrt(eigvals.min())

    fwhm = 2.3548 * sigma_major
    ecc = np.sqrt(1 - (sigma_minor / sigma_major) ** 2)
    return dict(fwhm=fwhm, eccentricity=ecc, peak=cutout.max())


# --------------------------------------------------------------------------
# Per-frame metrics
# --------------------------------------------------------------------------

def analyze_frame(data, saturation_level=None):
    sources, bg_median, bg_std = detect_stars(data)

    metrics = dict(
        n_stars=0,
        fwhm_median=np.nan,
        fwhm_std=np.nan,
        eccentricity_median=np.nan,
        background_median=bg_median,
        background_std=bg_std,
        snr_median=np.nan,
        saturation_frac=np.nan,
    )

    if saturation_level is None:
        saturation_level = np.nanmax(data) if np.isfinite(data).any() else np.inf
    metrics["saturation_frac"] = float(np.mean(data >= 0.98 * saturation_level))

    if sources is None or len(sources) == 0:
        return metrics

    # photutils renamed centroid columns in v3.0 (xcentroid -> x_centroid)
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    fwhms, eccs, snrs = [], [], []
    for src in sources:
        m = star_moments(data, src[xcol], src[ycol], bg=bg_median)
        if m is None:
            continue
        fwhms.append(m["fwhm"])
        eccs.append(m["eccentricity"])
        snrs.append(m["peak"] / bg_std if bg_std > 0 else np.nan)

    if fwhms:
        metrics["n_stars"] = len(fwhms)
        metrics["fwhm_median"] = float(np.median(fwhms))
        metrics["fwhm_std"] = float(np.std(fwhms))
        metrics["eccentricity_median"] = float(np.nanmedian(eccs))
        metrics["snr_median"] = float(np.nanmedian(snrs))

    return metrics


# --------------------------------------------------------------------------
# Composite score: star shape + background level + noise level
# --------------------------------------------------------------------------

def compute_quality_scores(df, wfwhm_k=1.0):
    df = df.copy()

    # Siril-style weighted FWHM: FWHM penalized when a frame detects fewer
    # stars than the best frame in the batch. Lower is better. This is the
    # main component of the quality score (covers FWHM + star count).
    #   wFWHM = FWHM * (1 + k * (1 - n_stars / n_stars_max))
    n_max = df["n_stars"].max()
    if n_max and n_max > 0:
        df["wfwhm"] = df["fwhm_median"] * (
            1 + wfwhm_k * (1 - df["n_stars"] / n_max)
        )
    else:
        df["wfwhm"] = df["fwhm_median"]

    def rank_pct(series, higher_is_better=True):
        r = series.rank(pct=True, na_option="bottom") * 100
        return r if higher_is_better else 100 - r

    # --- 1. Star shape & count (55%): wFWHM combines FWHM tightness with a
    #        star-count penalty (see formula above); eccentricity = roundness
    df["score_wfwhm"] = rank_pct(df["wfwhm"], higher_is_better=False)
    df["score_eccentricity"] = rank_pct(df["eccentricity_median"], higher_is_better=False)

    # --- 2. Background level (15%): lower sky background = better
    df["score_background_level"] = rank_pct(df["background_median"], higher_is_better=False)

    # --- 3. Noise level (30%): lower background noise = better
    df["score_noise"] = rank_pct(df["background_std"], higher_is_better=False)

    weights = dict(
        score_wfwhm=0.40,             # FWHM + star count (weighted up)
        score_eccentricity=0.15,      # star roundness
        score_background_level=0.15,  # background level
        score_noise=0.30,             # noise level
    )
    df["quality_score"] = sum(df[k] * w for k, w in weights.items())
    return df


# --------------------------------------------------------------------------
# FITS preview (per-frame PNG for visual inspection)
# --------------------------------------------------------------------------

# Mapping from FITS BAYERPAT keyword values to OpenCV Bayer codes.
try:
    import cv2 as _cv2
    _BAYER_CODES = {
        "BGGR": _cv2.COLOR_BAYER_BG2RGB,
    }
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    _BAYER_CODES = {}


def debayer(data, bayer_pattern):
    """Demosaic a raw 2-D Bayer array.

    Returns a float64 HxWx3 RGB array, or None if the pattern is unknown
    or cv2 is unavailable.
    """
    if not _CV2_AVAILABLE:
        return None
    code = _BAYER_CODES.get(str(bayer_pattern).upper().strip())
    if code is None:
        return None
    # cv2 requires uint16 input for 16-bit Bayer data
    raw = np.clip(data, 0, None)
    if raw.max() <= 255:
        raw8 = raw.astype(np.uint8)
        rgb = _cv2.cvtColor(raw8, code)
    else:
        raw16 = (raw / raw.max() * 65535).astype(np.uint16)
        # Use high-quality demosaic
        rgb16 = _cv2.cvtColor(raw16, code)
        rgb = rgb16.astype(np.float64) / 65535.0
        return rgb  # already 0-1 float
    return rgb.astype(np.float64) / 255.0


def _stretch(arr, low_pct, high_pct):
    """Percentile linear stretch to 0-1, applied per-channel for RGB."""
    vmin = np.nanpercentile(arr, low_pct)
    vmax = np.nanpercentile(arr, high_pct)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def save_fits_preview(data, outpath, header=None, low_pct=1.0, high_pct=99.5,
                       jpeg_quality=80):
    """Save a percentile-stretched preview image (format from extension;
    .jpg recommended to save space).

    Debayers as BGGR when cv2 is available and saves an RGB image; otherwise
    a grayscale image is saved.
    """
    save_kw = {}
    if outpath.lower().endswith((".jpg", ".jpeg")):
        save_kw["pil_kwargs"] = {"quality": jpeg_quality}

    rgb = debayer(data, "BGGR")

    if rgb is not None:
        # Stretch each channel independently then stack
        stretched = np.stack(
            [_stretch(rgb[:, :, c], low_pct, high_pct) for c in range(3)],
            axis=-1,
        )
        # Flip vertically: FITS origin is bottom-left, image origin is top-left
        stretched = np.flipud(stretched)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(stretched, interpolation="nearest", aspect="equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(outpath, dpi=150, bbox_inches="tight", pad_inches=0,
                    **save_kw)
        plt.close(fig)
        return True  # colour save

    # Fallback: grayscale
    vmin = np.nanpercentile(data, low_pct)
    vmax = np.nanpercentile(data, high_pct)
    if vmax <= vmin:
        vmax = vmin + 1.0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(outpath, dpi=150, bbox_inches="tight", pad_inches=0,
                **save_kw)
    plt.close(fig)
    return False  # grayscale save


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def shorten(name, n=22):
    return name if len(name) <= n else name[: n - 3] + "..."


def plot_ranked_scores(df, outdir):
    d = df.sort_values("quality_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(d))))
    colors = plt.cm.RdYlGn(d["quality_score"] / 100)
    ax.barh([shorten(f) for f in d["file"]], d["quality_score"], color=colors)
    mu, sd = d["quality_score"].mean(), d["quality_score"].std()
    ax.axvline(mu, color="k", ls="--", lw=1.5,
               label=f"mean = {mu:.1f}")
    ax.axvspan(mu - sd, mu + sd, color="gray", alpha=0.15,
               label=f"±1 std = {sd:.1f}")
    ax.legend(loc="lower right")
    ax.set_xlabel("Quality score (0-100, batch-relative)")
    ax.set_title("Frame quality ranking\n"
                 "(wFWHM 40% + roundness 15% + background 15% + noise 30%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "quality_scores.png"), dpi=150)
    plt.close(fig)


def plot_metric_distributions(df, outdir):
    metrics = [
        ("fwhm_median", "Median FWHM (px)"),
        ("eccentricity_median", "Median eccentricity"),
        ("background_median", "Background level (ADU)"),
        ("background_std", "Background noise (ADU)"),
        ("n_stars", "Detected stars"),
        ("quality_score", "Quality score"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, label) in zip(axes.ravel(), metrics):
        vals = df[col].dropna()
        if len(vals):
            ax.hist(vals, bins=min(20, max(5, len(vals) // 2)),
                    color="steelblue", edgecolor="white")
            mu, sd = vals.mean(), vals.std()
            ax.axvline(mu, color="crimson", ls="--", lw=1.5,
                       label=f"mean = {mu:.2f}")
            ax.axvspan(mu - sd, mu + sd, color="crimson", alpha=0.10,
                       label=f"±1 std = {sd:.2f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("Frames")
        ax.grid(alpha=0.3)
    fig.suptitle("Metric distributions across the batch")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "metric_distributions.png"), dpi=150)
    plt.close(fig)


def plot_fwhm_vs_score(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["fwhm_median"], df["quality_score"],
                    c=df["eccentricity_median"], cmap="viridis", s=60,
                    edgecolor="k", linewidth=0.5)
    fig.colorbar(sc, ax=ax, label="Median eccentricity")
    ax.set_xlabel("Median FWHM (px)")
    ax.set_ylabel("Quality score")
    ax.set_title("FWHM vs quality score (color = eccentricity)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fwhm_vs_score.png"), dpi=150)
    plt.close(fig)


def plot_sequence(df, outdir):
    """Metrics in file order - useful for spotting drift (focus, clouds,
    tracking degradation) over the imaging session."""
    d = df.sort_values("file").reset_index(drop=True)
    x = np.arange(len(d))
    metrics = [
        ("fwhm_median", "Median FWHM (px)"),
        ("eccentricity_median", "Median eccentricity"),
        ("background_median", "Background level (ADU)"),
        ("background_std", "Background noise (ADU)"),
        ("quality_score", "Quality score"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 12), sharex=True)
    for ax, (col, label) in zip(axes, metrics):
        ax.plot(x, d[col], "o-", ms=4, color="steelblue")
        vals = d[col].dropna()
        if len(vals):
            mu, sd = vals.mean(), vals.std()
            ax.axhline(mu, color="crimson", ls="--", lw=1.5,
                       label=f"mean = {mu:.2f}")
            ax.fill_between(x, mu - sd, mu + sd, color="crimson", alpha=0.10,
                            label=f"±1 std = {sd:.2f}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frame index (sorted by filename)")
    fig.suptitle("Metrics over the imaging sequence")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "metrics_over_sequence.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Interactive review (browse JPEG previews, veto individual frames)
# --------------------------------------------------------------------------

def review_best_frames(good_df, previews_dir, label):
    """Quick Tk image browser over a batch of JPEG previews, so you can flip
    through them and veto individual frames by eye. Works with or without
    quality-score columns present (score/FWHM/etc. are shown when available,
    e.g. when called after scoring instead of before it).

    Every frame starts as "keep". Displays the saved JPEG preview (not the
    FITS file) for speed.

    Keys:
      Left / Right  - previous / next frame
      Space         - toggle keep/reject for the current frame
      Enter / Esc   - finish review and return the surviving frames

    Returns a DataFrame containing only the rows the user kept.
    """
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        print(f"[{label}] --review needs tkinter and Pillow "
              "(pip install pillow) to display JPEGs; skipping review and "
              "exporting all top-percent frames as-is.")
        return good_df

    rows = good_df.reset_index(drop=True)
    paths = []
    missing = []
    for _, row in rows.iterrows():
        stem = os.path.splitext(row["file"])[0]
        jpg = os.path.join(previews_dir, row["date"], stem + ".jpg")
        paths.append(jpg if os.path.isfile(jpg) else None)
        if paths[-1] is None:
            missing.append(row["file"])
    if missing:
        print(f"[{label}] Warning: {len(missing)} preview JPEG(s) missing "
              "(need --previews on); those frame(s) are shown blank but "
              "still reviewable: " + ", ".join(missing))

    keep = {i: True for i in range(len(rows))}
    idx_holder = {"i": 0}
    photo_ref = {"img": None}  # keep a reference so Tk doesn't garbage-collect it

    root = tk.Tk()
    root.title(f"Review best frames - {label}")
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    max_w, max_h = int(screen_w * 0.85), int(screen_h * 0.75)

    image_lbl = tk.Label(root, bg="black")
    image_lbl.pack()
    info_lbl = tk.Label(root, font=("TkDefaultFont", 12), justify="left", anchor="w")
    info_lbl.pack(fill="x", padx=8, pady=4)
    help_lbl = tk.Label(
        root,
        text="\u2190 / \u2192  prev / next      space  toggle keep / reject      "
             "Enter / Esc  finish review",
        font=("TkDefaultFont", 10), fg="gray30",
    )
    help_lbl.pack(pady=(0, 6))

    def render():
        i = idx_holder["i"]
        row = rows.iloc[i]
        path = paths[i]
        if path is not None:
            im = Image.open(path)
            im.thumbnail((max_w, max_h), Image.LANCZOS)
            photo_ref["img"] = ImageTk.PhotoImage(im)
            image_lbl.configure(image=photo_ref["img"], text="",
                                 width=im.width, height=im.height)
        else:
            photo_ref["img"] = None
            image_lbl.configure(image="", text="(no preview JPEG found)",
                                 fg="white", width=60, height=20)
        status = "KEEP" if keep[i] else "REJECT"
        header = f"[{i + 1}/{len(rows)}]  {row['file']}\ndate={row['date']}"
        if "quality_score" in rows.columns:
            fwhm = row.get("fwhm_median", float("nan"))
            ecc = row.get("eccentricity_median", float("nan"))
            n_stars = row.get("n_stars", 0)
            header += (f"   score={row['quality_score']:.1f}   "
                       f"FWHM={fwhm:.2f}   ecc={ecc:.2f}   stars={int(n_stars)}")
        info_lbl.configure(
            text=f"{header}\nstatus: {status}",
            fg="darkgreen" if keep[i] else "firebrick",
        )
        root.title(f"Review best frames - {label}  ({i + 1}/{len(rows)})  [{status}]")

    def go(delta):
        idx_holder["i"] = max(0, min(len(rows) - 1, idx_holder["i"] + delta))
        render()

    def toggle(_event=None):
        keep[idx_holder["i"]] = not keep[idx_holder["i"]]
        render()

    def finish(_event=None):
        root.destroy()

    root.bind("<Left>", lambda e: go(-1))
    root.bind("<Right>", lambda e: go(1))
    root.bind("<space>", toggle)
    root.bind("<Return>", finish)
    root.bind("<Escape>", finish)

    render()
    root.mainloop()

    kept_idx = [i for i, k in keep.items() if k]
    rejected_files = rows.loc[[i for i, k in keep.items() if not k], "file"].tolist()
    if rejected_files:
        print(f"[{label}] Review: rejected {len(rejected_files)} frame(s): "
              + ", ".join(rejected_files))
    else:
        print(f"[{label}] Review: kept all {len(rows)} frames")
    return rows.iloc[kept_idx].reset_index(drop=True)


# --------------------------------------------------------------------------
# Session discovery & date handling
# --------------------------------------------------------------------------

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def extract_date(filename, fallback="unknown-date"):
    """Pull the leading YYYY-MM-DD out of a frame filename, e.g.
    '2026-08-05_00-13-32__-10.10_180.00s_0000.fits' -> '2026-08-05'."""
    m = DATE_RE.match(filename)
    return m.group(1) if m else fallback


def extract_session_date(path, fallback="unknown-date"):
    """Determine which $DATE a frame belongs to.

    Prefers a YYYY-MM-DD directory component in the file's path (the NINA
    session folder), because frames captured after midnight carry the NEXT
    day's date in their filename while still belonging to the session that
    started the evening before, e.g.:
        .../2026-08-08/LIGHT/North America Nebula/2026-08-09_00-49-24_...fits
    -> '2026-08-08'.
    Falls back to the filename date, then to `fallback`.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)[:-1]
    for p in reversed(parts):
        m = DATE_RE.fullmatch(p)
        if m:
            return m.group(1)
    return extract_date(os.path.basename(path), fallback=fallback)


def find_sessions(root):
    """Scan root for $DATE/LIGHTS folders. Returns [(label, lights_path)]."""
    sessions = []
    for entry in sorted(os.listdir(root)):
        session_dir = os.path.join(root, entry)
        if not os.path.isdir(session_dir):
            continue
        for sub in ("LIGHT", "Light", "light", "LIGHTS", "Lights", "lights"):
            lights_dir = os.path.join(session_dir, sub)
            if os.path.isdir(lights_dir) and list_fits(lights_dir,
                                                        recursive=True):
                sessions.append((entry, lights_dir))
                break
    return sessions


LIGHT_DIR_NAMES = {"light", "lights"}

# Calibration folder names copied alongside the exported lights.
CALIBRATION_DIR_NAMES = {
    "flat", "flats", "dark", "darks", "bias", "biases",
    "darkflat", "darkflats", "flatdark", "flatdarks",
}


def find_session_dir(path):
    """Walk up from a frame's path to its $DATE session directory (the
    directory whose name is exactly YYYY-MM-DD). Returns None if no such
    directory exists in the path."""
    d = os.path.dirname(os.path.normpath(os.path.abspath(path)))
    while True:
        if DATE_RE.fullmatch(os.path.basename(d)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def copy_calibration_frames(session_dir, dest_date_dir):
    """Copy calibration subfolders (FLAT, DARK, BIAS, ...) from a session's
    $DATE directory into the export folder for that date, preserving the
    folder names and internal structure:

        <session_dir>/FLAT/...  ->  <dest_date_dir>/FLAT/...

    Calibration frames are always COPIED (never moved), even when --move is
    used for the lights, so the originals stay intact. Returns the list of
    calibration folder names that were copied."""
    import shutil
    copied = []
    for entry in sorted(os.listdir(session_dir)):
        src = os.path.join(session_dir, entry)
        if not os.path.isdir(src):
            continue
        if entry.lower() not in CALIBRATION_DIR_NAMES:
            continue
        if not list_fits(src, recursive=True):
            continue
        dst = os.path.join(dest_date_dir, entry)
        os.makedirs(dest_date_dir, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied.append(entry)
    return copied


# --------------------------------------------------------------------------
# Per-session processing
# --------------------------------------------------------------------------

def process_session(label, light_files, outdir, args, before_cutoff):
    """Analyze one night's LIGHTS folder: metrics, plots, JPEG previews,
    and export of the best frames into <good-dir>/<date>/LIGHTS/ (original
    filenames kept; the per-date folder prevents collisions between
    nights). Calibration folders (FLAT, DARK, BIAS, ...) from the session
    directory are copied into <good-dir>/<date>/ when --copy-calibration
    is passed. Scoring is percentile-ranked WITHIN this session only, so
    nights are judged on their own conditions."""
    os.makedirs(outdir, exist_ok=True)
    previews_dir = os.path.join(outdir, "previews")
    os.makedirs(previews_dir, exist_ok=True)

    # ---- Pass 1: load frames, save previews (needed for manual inspection) ----
    frame_records = []
    for i, path in enumerate(light_files, 1):
        name = os.path.basename(path)
        print(f"[{label}] [{i}/{len(light_files)}] {name}")
        try:
            data, header = load_data(path)
        except Exception as e:
            print(f"  Skipped (read error): {e}")
            continue
        if data is None or data.ndim != 2:
            print("  Skipped: not a 2D image HDU")
            continue

        if before_cutoff is not None:
            # Prefer TIME-OBS if present; fall back to the time portion of DATE-OBS
            time_str = header.get("TIME-OBS", "")
            if not time_str:
                date_obs_str = header.get("DATE-OBS", "")
                # DATE-OBS is typically YYYY-MM-DDTHH:MM:SS[.sss]
                if "T" in date_obs_str:
                    time_str = date_obs_str.split("T")[1]
                elif " " in date_obs_str:
                    time_str = date_obs_str.split(" ")[1]
            if time_str:
                try:
                    frame_time = datetime.strptime(
                        time_str.split(".")[0], "%H:%M:%S"
                    ).time()
                    if frame_time >= before_cutoff:
                        print(f"  Skipped (time {frame_time.strftime('%H:%M:%S')} >= cutoff {before_cutoff.strftime('%H:%M:%S')})")
                        continue
                except ValueError:
                    print(f"  Warning: could not parse time '{time_str}'; frame included")
            else:
                print("  Warning: no time found in header; frame included")

        stem = os.path.splitext(name)[0]
        frame_date = extract_session_date(path, fallback=label)
        # A frame is "new" if its preview JPEG doesn't already exist on disk,
        # i.e. it wasn't saved (and presumably reviewed) in a previous run.
        is_new = True
        if args.previews:
            date_prev_dir = os.path.join(previews_dir, frame_date)
            preview_path = os.path.join(date_prev_dir, stem + ".jpg")
            if os.path.isfile(preview_path):
                is_new = False
                print("  Preview skipped (already saved)")
            else:
                os.makedirs(date_prev_dir, exist_ok=True)
                colour = save_fits_preview(
                    data, preview_path,
                    header=header, jpeg_quality=args.jpeg_quality,
                )
                print(f"  Preview saved ({'RGB' if colour else 'grayscale'} JPEG)")

        frame_records.append(dict(file=name, path=path, date=frame_date,
                                   data=data, header=header, is_new=is_new))

    if not frame_records:
        print(f"[{label}] No frames were successfully processed; skipping.")
        return None
    n_loaded = len(frame_records)

    # ---- Manual inspection: veto frames by eye BEFORE quality scoring ----
    # Only frames whose preview JPEG was newly saved this run are shown;
    # frames whose preview already existed (saved/reviewed in a previous
    # run) are kept automatically and still quality-scored below.
    if args.review:
        if not args.previews:
            print(f"[{label}] --review needs JPEG previews to display; "
                  "skipping review (run without --no-previews to enable "
                  "it). All frames will be quality-scored.")
        else:
            new_records = [r for r in frame_records if r["is_new"]]
            old_records = [r for r in frame_records if not r["is_new"]]
            if not new_records:
                print(f"[{label}] No new frames to review (all previews "
                      "already existed); skipping manual inspection.")
            else:
                review_df = pd.DataFrame([
                    {"file": r["file"], "date": r["date"], "path": r["path"]}
                    for r in new_records
                ])
                print(f"\n=== [{label}] Manual inspection: {len(review_df)} "
                      f"new frame(s) (of {n_loaded} total, "
                      f"{len(old_records)} previously saved), opening viewer... ===")
                survivors = review_best_frames(review_df, previews_dir, label)
                keep_paths = set(survivors["path"])
                frame_records = old_records + [
                    r for r in new_records if r["path"] in keep_paths
                ]
                print(f"[{label}] Manual inspection kept "
                      f"{len(frame_records)}/{n_loaded} frame(s)")

    if not frame_records:
        print(f"[{label}] All frames rejected during manual inspection; skipping.")
        return None

    # ---- Pass 2: quality metrics + scoring, surviving frames only ----
    rows = []
    for r in frame_records:
        metrics = analyze_frame(r["data"], saturation_level=args.saturation_level)
        metrics["file"] = r["file"]
        metrics["path"] = r["path"]
        metrics["date"] = r["date"]
        metrics["exptime"] = r["header"].get("EXPTIME", r["header"].get("EXPOSURE", np.nan))
        metrics["filter"] = r["header"].get("FILTER", "")
        metrics["date_obs"] = r["header"].get("DATE-OBS", "")
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = compute_quality_scores(df)
    df_ranked = df.sort_values("quality_score", ascending=False).reset_index(drop=True)

    cols = ["file", "date", "quality_score",
            "score_wfwhm", "score_eccentricity",
            "score_background_level", "score_noise",
            "fwhm_median", "eccentricity_median",
            "background_median", "background_std",
            "wfwhm", "n_stars", "snr_median",
            "saturation_frac", "exptime", "filter", "date_obs"]
    df_ranked[cols].to_csv(os.path.join(outdir, "quality_report.csv"),
                            index=False)

    plot_ranked_scores(df, outdir)
    plot_metric_distributions(df, outdir)
    plot_fwhm_vs_score(df, outdir)
    plot_sequence(df, outdir)

    print(f"\n=== [{label}] Quality ranking (best first) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(df_ranked[["file", "quality_score", "wfwhm", "fwhm_median",
                          "eccentricity_median", "background_median",
                          "background_std", "n_stars"]]
              .round(2).to_string(index=False))

    # ---- Copy best frames into <good-dir>/<date>/LIGHTS/ ----------------
    if not args.no_export:
        import shutil
        # Keep the top X% of frames by quality score (at least 1 frame).
        n_keep = max(1, int(round(len(df_ranked) * args.top_percent / 100.0)))
        good = df_ranked.head(n_keep)

        verb = "Moved" if args.move else "Copied"
        print(f"\n=== [{label}] Best frame export -> "
              f"{args.good_dir}/<date>/LIGHTS/ ===")
        print(f"Keeping top {args.top_percent:.0f}% = {n_keep}/{len(df_ranked)} "
              f"frames from scoring")
        for _, row in good.iterrows():
            date_dir = os.path.join(args.good_dir, row["date"])
            lights_dst_dir = os.path.join(date_dir, "LIGHTS")
            os.makedirs(lights_dst_dir, exist_ok=True)
            dst = os.path.join(lights_dst_dir, row["file"])
            if args.move:
                shutil.move(row["path"], dst)
            else:
                shutil.copy2(row["path"], dst)
            # Bring the frame's JPEG preview along, into the date folder
            stem = os.path.splitext(row["file"])[0]
            preview_src = os.path.join(previews_dir, row["date"],
                                        stem + ".jpg")
            if os.path.isfile(preview_src):
                good_prev_dir = os.path.join(date_dir, "previews")
                os.makedirs(good_prev_dir, exist_ok=True)
                shutil.copy2(preview_src,
                              os.path.join(good_prev_dir, stem + ".jpg"))
            print(f"  {verb}: {row['date']}/LIGHTS/{row['file']}  "
                  f"(score {row['quality_score']:.1f})")
        rejected = df_ranked.iloc[n_keep:]
        if len(rejected):
            print(f"Left behind ({len(rejected)}): "
                  + ", ".join(rejected["file"].tolist()))

        # ---- Raw calibration frames (only when --copy-calibration) ------
        if args.copy_calibration:
            print(f"\n=== [{label}] Calibration frames ===")
            for date in sorted(good["date"].unique()):
                sample_path = good[good["date"] == date].iloc[0]["path"]
                session_dir = find_session_dir(sample_path)
                if session_dir is None:
                    print(f"  [{date}] No $DATE session folder found in path; "
                          f"calibration frames not copied")
                    continue
                dest_date_dir = os.path.join(args.good_dir, date)
                copied = copy_calibration_frames(session_dir, dest_date_dir)
                if copied:
                    print(f"  [{date}] Copied {', '.join(copied)} "
                          f"from {session_dir} -> {dest_date_dir}/")
                else:
                    print(f"  [{date}] No calibration folders "
                          f"(FLAT/DARK/BIAS/...) found in {session_dir}")

    print(f"\n[{label}] Outputs written to {outdir}:")
    for f in ["quality_report.csv", "quality_scores.png",
              "metric_distributions.png", "fwhm_vs_score.png",
              "metrics_over_sequence.png"]:
        print(f"  - {f}")
    if args.previews:
        print(f"  - previews/<date>/<frame>.jpg  ({n_loaded} FITS previews)")
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="FITS quality inspector with plots. Point --root at a "
                    "directory laid out as $ROOT/$DATE/LIGHTS/*.fits to "
                    "process every night, or --lights at a single folder.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", default=None,
                      help="Main directory containing $DATE/LIGHTS/ subfolders")
    src.add_argument("--lights", default=None,
                      help="Single folder of FITS frames (old behaviour)")
    ap.add_argument("--outdir", default="./quality_report",
                     help="Report folder; per-date subfolders are created "
                          "under it when using --root")
    ap.add_argument("--saturation-level", type=float, default=None,
                     help="ADU value considered saturated (default: per-frame max)")
    ap.add_argument("--good-dir", default="./best_frames",
                     help="Folder to receive the best frames, organised as "
                          "<good-dir>/<date>/LIGHTS/frame.fits. "
                          "Default: ./best_frames")
    ap.add_argument("--copy-calibration", action=argparse.BooleanOptionalAction,
                     default=False,
                     help="Also copy the raw calibration subfolders (FLAT, "
                          "DARK, BIAS, DARKFLAT, ...) into "
                          "<good-dir>/<date>/<CALIB_NAME>/. Off by default; "
                          "pass --copy-calibration to keep the raw frames "
                          "alongside the exported lights.")
    ap.add_argument("--no-export", action="store_true",
                     help="Skip copying best frames entirely")
    ap.add_argument("--top-percent", type=float, default=80.0,
                     help="Copy the best X%% of frames by quality score "
                          "(default 80, i.e. drop the worst 20%%)")
    ap.add_argument("--move", action="store_true",
                     help="Move light frames into --good-dir instead of "
                          "copying (default: copy, originals untouched). "
                          "Raw calibration frames (if --copy-calibration "
                          "is used) are always copied, never moved.")
    ap.add_argument("--before", default=None,
                     help="Ignore frames whose local time-of-day is at or after "
                          "this time. Format: HH:MM:SS (e.g. 02:30:00). "
                          "The time is read from DATE-OBS (or TIME-OBS) in the "
                          "FITS header as local time.")
    ap.add_argument("--previews", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="Write per-frame JPEG previews (on by default; "
                          "use --no-previews to turn off)")
    ap.add_argument("--jpeg-quality", type=int, default=80,
                     help="JPEG quality for previews, 1-95 (default 80)")
    ap.add_argument("--review", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="After previews are generated, open a viewer over "
                          "ALL frames so you can veto bad ones by eye "
                          "before quality scoring; only surviving frames "
                          "are scored and exported (on by default; "
                          "requires --previews and Pillow; use --no-review "
                          "to score/export automatically without "
                          "prompting)")
    args = ap.parse_args()

    # Parse the --before cutoff (time-only) once
    before_cutoff = None
    if args.before:
        try:
            before_cutoff = datetime.strptime(args.before, "%H:%M:%S").time()
        except ValueError:
            sys.exit(
                f"--before '{args.before}' is not a valid time. Use HH:MM:SS, e.g. 02:30:00."
            )
        print(f"Ignoring frames taken at or after {before_cutoff.strftime('%H:%M:%S')} (local time)")

    if args.root:
        sessions = find_sessions(args.root)
        if not sessions:
            sys.exit(f"No $DATE/LIGHTS folders with FITS files found under {args.root}")
        print(f"Found {len(sessions)} session(s): "
              + ", ".join(lbl for lbl, _ in sessions))
        for label, lights_dir in sessions:
            files = list_fits(lights_dir, recursive=True)
            session_outdir = os.path.join(args.outdir, label)
            print(f"\n{'='*60}\nProcessing session {label} "
                  f"({len(files)} frames)\n{'='*60}")
            process_session(label, files, session_outdir, args, before_cutoff)
    else:
        light_files, how = collect_light_files(args.lights)
        print(f"File selection: {how}")
        if not light_files:
            sys.exit(f"No FITS files found in {args.lights}")
        process_session(os.path.basename(os.path.normpath(args.lights)) or "session",
                         light_files, args.outdir, args, before_cutoff)


def collect_light_files(base):
    """Collect FITS under base, restricted to LIGHT/LIGHTS subdirectories.

    Rules:
      - If base itself is (or is inside) a LIGHT/LIGHTS folder, take
        everything under it.
      - Otherwise, keep ONLY files that sit under a LIGHT/LIGHTS
        subdirectory somewhere below base (BIAS, FLAT, DARK etc. ignored).
      - If no LIGHT/LIGHTS folder exists anywhere in the tree, fall back
        to treating base itself as the lights folder (old behaviour).
    """
    base_abs = os.path.normpath(os.path.abspath(base))
    base_is_light = any(p.lower() in LIGHT_DIR_NAMES
                         for p in base_abs.split(os.sep))
    files = list_fits(base, recursive=True)
    if base_is_light:
        return files, "base is a LIGHT folder"

    def in_light_dir(path):
        rel_dirs = os.path.relpath(path, base).split(os.sep)[:-1]
        return any(p.lower() in LIGHT_DIR_NAMES for p in rel_dirs)

    light_files = [f for f in files if in_light_dir(f)]
    if light_files:
        n_skipped = len(files) - len(light_files)
        return light_files, (f"restricted to LIGHT subfolders "
                              f"({n_skipped} non-LIGHT FITS ignored)")
    return files, "no LIGHT subfolders found; using all FITS in folder"


if __name__ == "__main__":
    main()