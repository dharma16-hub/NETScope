"""
NETScope - step 2: find each cell, then follow its green signal over time.

Why a max projection instead of per-frame segmentation:
nuclei decondense and dim as the experiment runs, so a per-frame Otsu
threshold loses the faint ones (19 objects at t=0, only 6 by the end).
The cells barely move in this field, so we instead define each cell ONCE
from the maximum-intensity projection of the DNA channel, and measure
both channels inside that fixed region at every frame.

Produces:
  results/tables/02_cell_traces.csv      one row per cell per frame
  results/tables/02_cell_summary.csv     one row per cell
  results/figures/02_segmentation.png
  results/figures/02_traces.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import yaml
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]

SMOOTH_SIGMA = 2        # px; merges the lobes of one nucleus
CLOSE_RADIUS = 3        # px; fills small gaps inside a nucleus
MIN_AREA_PX = 100       # drop debris
BASELINE_FRAMES = 5     # frames averaged to define each cell's own baseline
DECAY_CUTOFF = 0.3      # final/peak below this = signal dispersed


def load_config():
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def load_stack(cfg):
    with tifffile.TiffFile(ROOT / cfg["data"]["path"]) as tif:
        series = tif.series[0]
        if series.axes != "TCYX":
            raise ValueError(f"expected axes TCYX, got {series.axes}")
        return series.asarray()


def segment_cells(dna):
    """Define cell regions once, from the max projection over time."""
    projection = dna.max(axis=0).astype(float)
    smoothed = gaussian(projection, sigma=SMOOTH_SIGMA)
    threshold = threshold_otsu(smoothed)
    mask = closing(smoothed > threshold, disk(CLOSE_RADIUS))
    # scikit-image 0.26 renamed min_size -> max_size (same number here:
    # objects up to and including this many pixels are dropped)
    mask = remove_small_objects(mask, max_size=MIN_AREA_PX)
    return label(mask), projection, threshold


def measure(labels, dna, release, cfg):
    """Mean intensity of both channels inside every cell, at every frame."""
    dt_min = cfg["data"]["frame_interval_s"] / 60.0
    px_area = cfg["data"]["pixel_size_um"] ** 2
    rows = []
    for region in regionprops(labels):
        sl = region.slice
        mask = labels[sl] == region.label
        for t in range(dna.shape[0]):
            rows.append({
                "cell_id": region.label,
                "frame": t,
                "time_min": round(t * dt_min, 1),
                "area_um2": round(region.area * px_area, 2),
                "dna_mean": dna[t][sl][mask].mean(),
                "release_mean": release[t][sl][mask].mean(),
            })
    return pd.DataFrame(rows)


def summarise(traces):
    """One row per cell: when it lit up, how brightly, and whether it stayed lit."""
    out = []
    for cell_id, grp in traces.groupby("cell_id"):
        grp = grp.sort_values("frame")
        signal = grp["release_mean"].to_numpy()
        baseline = signal[:BASELINE_FRAMES].mean()
        fold = signal / baseline
        peak_i = int(fold.argmax())
        out.append({
            "cell_id": cell_id,
            "area_um2": grp["area_um2"].iloc[0],
            "baseline": round(baseline, 1),
            "peak_fold": round(float(fold.max()), 1),
            "peak_time_min": float(grp["time_min"].iloc[peak_i]),
            "final_fold": round(float(fold[-1]), 1),
            "decay_ratio": round(float(fold[-1] / fold.max()), 3),
        })
    df = pd.DataFrame(out)
    df["class"] = np.where(df["decay_ratio"] < DECAY_CUTOFF, "dispersing", "persistent")
    return df


def plot_segmentation(projection, labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    lo, hi = np.percentile(projection, [1, 99.5])
    axes[0].imshow(projection, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("DNA channel, max projection")

    overlay = np.zeros((*projection.shape, 3))
    norm = np.clip((projection - lo) / (hi - lo), 0, 1)
    overlay[..., 0] = overlay[..., 1] = overlay[..., 2] = norm
    overlay[find_boundaries(labels, mode="outer")] = [1, 0.3, 0]
    axes[1].imshow(overlay)
    axes[1].set_title(f"{labels.max()} cells found")

    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    for region in regionprops(labels):
        y, x = region.centroid
        axes[1].text(x, y, str(region.label), color="yellow",
                     fontsize=7, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_traces(traces, summary, out_path):
    colours = {"dispersing": "tab:orange", "persistent": "tab:blue"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    cls = dict(zip(summary["cell_id"], summary["class"]))
    for cell_id, grp in traces.groupby("cell_id"):
        grp = grp.sort_values("frame")
        baseline = grp["release_mean"].iloc[:BASELINE_FRAMES].mean()
        axes[0].plot(grp["time_min"], grp["release_mean"] / baseline,
                     color=colours[cls[cell_id]], alpha=0.75, lw=1.2)
    axes[0].set_xlabel("Time (min)"), axes[0].set_ylabel("Release signal (fold over own baseline)")
    axes[0].set_title("Per-cell release trajectories")
    for name, colour in colours.items():
        axes[0].plot([], [], color=colour, label=name)
    axes[0].legend(frameon=False)

    for cell_id, grp in traces.groupby("cell_id"):
        grp = grp.sort_values("frame")
        axes[1].plot(grp["time_min"], grp["dna_mean"],
                     color=colours[cls[cell_id]], alpha=0.75, lw=1.2)
    axes[1].set_xlabel("Time (min)"), axes[1].set_ylabel("DNA mean intensity (a.u.)")
    axes[1].set_title("Per-cell DNA intensity")

    axes[2].hist(summary["decay_ratio"], bins=20, color="grey")
    axes[2].axvline(DECAY_CUTOFF, color="red", ls="--")
    axes[2].set_xlabel("final / peak signal"), axes[2].set_ylabel("cells")
    axes[2].set_title("Two populations, split at %.1f" % DECAY_CUTOFF)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    stack = load_stack(cfg)
    dna = stack[:, cfg["data"]["channel_dna"]].astype(float)
    release = stack[:, cfg["data"]["channel_release"]].astype(float)

    labels, projection, threshold = segment_cells(dna)
    print(f"Otsu threshold on projection: {threshold:.0f}")
    print(f"cells found: {labels.max()}")

    traces = measure(labels, dna, release, cfg)
    summary = summarise(traces)

    tables = ROOT / cfg["output"]["tables"]
    figures = ROOT / cfg["output"]["figures"]
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    traces.to_csv(tables / "02_cell_traces.csv", index=False)
    summary.to_csv(tables / "02_cell_summary.csv", index=False)

    plot_segmentation(projection, labels, figures / "02_segmentation.png")
    plot_traces(traces, summary, figures / "02_traces.png")

    counts = summary["class"].value_counts()
    print("\nclasses:")
    for name, n in counts.items():
        sub = summary[summary["class"] == name]
        print(f"  {name:11s} n={n:2d}   peak time median "
              f"{sub['peak_time_min'].median():.0f} min")
    print("\nper cell:")
    print(summary.sort_values("peak_time_min").to_string(index=False))


if __name__ == "__main__":
    main()
