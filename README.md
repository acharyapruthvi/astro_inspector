# Astro Utils

A growing collection of small command-line tools for astrophotography workflows — calibration, quality control, and general FITS file wrangling. Each tool is a standalone script; see the sections below for what's currently available.

## Tools

### `fits_quality_plots.py` — FITS Frame Quality Inspector

Scores a batch of FITS light frames on image quality, ranks them, generates diagnostic plots, and copies (or moves) the best ones into a clean output folder — so you can quickly throw out bad subs before stacking.

#### What it does

1. **Loads** every FITS light frame from a folder (or a whole `$ROOT/$DATE/LIGHTS/` tree covering multiple nights).
2. **Measures per-frame metrics**:
   - Star shape & count — flux-weighted FWHM and eccentricity (roundness), via `DAOStarFinder` + second-moment fitting
   - Background level — median sky background (lower is better)
   - Noise level — background standard deviation (lower is better)
3. **Combines these into a single 0–100 quality score**, percentile-ranked within the batch (so it's a *relative* ranking, not an absolute standard):

   | Component | Weight | Notes |
   |---|---|---|
   | Weighted FWHM (wFWHM) | 40% | FWHM penalized when a frame has fewer detected stars than the best frame in the batch (Siril-style) |
   | Eccentricity | 15% | Star roundness |
   | Background level | 15% | Lower sky background scores higher |
   | Noise | 30% | Lower background std-dev scores higher |

4. **Optionally lets you eye-ball and veto frames** in a lightweight Tk viewer before scoring.
5. **Exports the best frames** (default: top 80% by score) into an organized output folder, and can optionally bring along the night's raw calibration frames (flats/darks/bias).

#### Outputs

Written to `--outdir` (per-session subfolder when using `--root`):

| File | Description |
|---|---|
| `quality_report.csv` | Full metric table, ranked best-first |
| `quality_scores.png` | Ranked bar chart of scores per frame |
| `metric_distributions.png` | Histograms of each metric across the batch |
| `fwhm_vs_score.png` | FWHM vs. quality score scatter, colored by eccentricity |
| `metrics_over_sequence.png` | Metrics vs. frame order — useful for spotting drift (focus shift, clouds, tracking degradation) over a session |
| `previews/<date>/<frame>.jpg` | Per-frame stretched JPEG preview (on by default) |
| `<good-dir>/<date>/LIGHTS/` | Copies (or moves) of the highest-quality frames |
| `<good-dir>/<date>/<CALIB>/` | Raw calibration frames, only with `--copy-calibration` |

#### Requirements

```bash
pip install astropy numpy pandas matplotlib photutils
```

Optional, for extra functionality:
- `opencv-python` — enables debayering of OSC (one-shot-color) BGGR raw frames for RGB previews (falls back to grayscale without it)
- `pillow` + `tkinter` — enables the interactive `--review` veto viewer (falls back to skipping review without it)

#### Usage

Single folder of lights:

```bash
python fits_quality_plots.py --lights ./lights --outdir ./quality_report
```

A whole imaging root organized as `$ROOT/$DATE/LIGHTS/*.fits` (processes every night found, each scored independently):

```bash
python fits_quality_plots.py --root ./imaging_data --outdir ./quality_report
```

#### Key options

| Flag | Default | Description |
|---|---|---|
| `--root` / `--lights` | — | One is required. `--root` walks `$DATE/LIGHTS/` subfolders; `--lights` scores a single folder |
| `--outdir` | `./quality_report` | Where reports/plots/previews are written |
| `--good-dir` | `./best_frames` | Where the best frames are exported, as `<good-dir>/<date>/LIGHTS/` |
| `--top-percent` | `80` | Keep the top X% of frames by score |
| `--no-export` | off | Skip exporting best frames entirely (analysis only) |
| `--move` | off (copy) | Move instead of copy the exported light frames |
| `--copy-calibration` | off | Also copy that night's FLAT/DARK/BIAS/DARKFLAT folders alongside the exported lights |
| `--saturation-level` | per-frame max | ADU value treated as saturated |
| `--before HH:MM:SS` | — | Drop frames taken at/after this local time (read from `DATE-OBS`/`TIME-OBS`) |
| `--previews` / `--no-previews` | on | Write per-frame JPEG previews |
| `--jpeg-quality` | `80` | JPEG quality (1–95) for previews |
| `--review` / `--no-review` | on | Open a Tk viewer to manually veto frames before scoring (requires previews + Pillow) |

Run `python fits_quality_plots.py --help` for the full list.

#### Notes & caveats

- Frames are analyzed **as-is** — there's no bias/flat calibration step. Run it on already-calibrated frames, or use it directly on raw lights for a quick relative ranking within that batch.
- Scoring is **relative to the batch/session** — a score of 90 means "one of the best frames in this particular set," not an absolute quality benchmark. Sessions are scored independently when using `--root`.
- Re-running on the same output folder skips regenerating previews for frames it's already saved a JPEG for (and auto-keeps them past the review step), so you can incrementally add new frames to a session without re-reviewing old ones.

---

## Planned / upcoming tools

More utilities for the astrophotography pipeline (calibration frame management, stacking helpers, etc.) will be added here over time.
