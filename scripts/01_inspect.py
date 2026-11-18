"""
NETScope - step 1: load the time-lapse and look at what we have.

Produces:
  results/tables/01_frame_summary.csv   one row per frame per channel
  results/figures/01_intensity_trace.png
  results/figures/01_montage.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # draw to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_stack(cfg):
    """Return the movie as an array with axes (T, C, Y, X)."""
    tif_path = ROOT / cfg["data"]["path"]
    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        if series.axes != "TCYX":
            raise ValueError(f"expected axes TCYX, got {series.axes}")
        stack = series.asarray()
    return stack


def frame_summary(stack, cfg):
    """One row per frame per channel: the simplest possible description."""
    dt_min = cfg["data"]["frame_interval_s"] / 60.0
    names = {cfg["data"]["channel_dna"]: "DNA (DAPI)",
             cfg["data"]["channel_release"]: "release (FITC)"}

    rows = []
    for t in range(stack.shape[0]):
        for c in range(stack.shape[1]):
            plane = stack[t, c]
            rows.append({
                "frame": t,
                "time_min": round(t * dt_min, 1),
                "channel": names.get(c, str(c)),
                "mean": plane.mean(),
                "median": float(np.median(plane)),
                "p99": float(np.percentile(plane, 99)),
                "max": int(plane.max()),
            })
    return pd.DataFrame(rows)


def plot_traces(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, stat, label in zip(axes, ["mean", "p99"],
                               ["Mean intensity", "99th percentile intensity"]):
        for name, grp in df.groupby("channel", sort=False):
            ax.plot(grp["time_min"], grp[stat], marker="o", ms=3, label=name)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{label} (a.u.)")
        ax.set_title(label)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_montage(stack, cfg, out_path, n=5):
    """A few frames across the movie, both channels, so we can see it."""
    dt_min = cfg["data"]["frame_interval_s"] / 60.0
    picks = np.linspace(0, stack.shape[0] - 1, n).astype(int)
    c_dna = cfg["data"]["channel_dna"]
    c_rel = cfg["data"]["channel_release"]

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    for col, t in enumerate(picks):
        for row, (c, cmap, label) in enumerate(
                [(c_dna, "gray", "DNA (DAPI)"), (c_rel, "viridis", "release (FITC)")]):
            plane = stack[t, c]
            # percentile scaling: robust to a few very bright pixels
            lo, hi = np.percentile(plane, [1, 99.5])
            ax = axes[row, col]
            ax.imshow(plane, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {t * dt_min:.0f} min", fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    stack = load_stack(cfg)

    t, c, y, x = stack.shape
    dt_min = cfg["data"]["frame_interval_s"] / 60.0
    px = cfg["data"]["pixel_size_um"]
    print(f"stack: {t} frames x {c} channels x {y} x {x} px, {stack.dtype}")
    print(f"interval {dt_min:.1f} min -> {(t - 1) * dt_min / 60:.1f} h total")
    print(f"pixel {px} um -> field {x * px:.0f} x {y * px:.0f} um")

    tables = ROOT / cfg["output"]["tables"]
    figures = ROOT / cfg["output"]["figures"]
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = frame_summary(stack, cfg)
    df.to_csv(tables / "01_frame_summary.csv", index=False)

    plot_traces(df, figures / "01_intensity_trace.png")
    plot_montage(stack, cfg, figures / "01_montage.png")

    print("\nfold change, first frame to last:")
    for name, grp in df.groupby("channel", sort=False):
        g = grp.sort_values("frame")
        print(f"  {name:18s} mean {g['mean'].iloc[-1] / g['mean'].iloc[0]:.2f}x"
              f"   peak at t = {g.loc[g['mean'].idxmax(), 'time_min']:.0f} min")


if __name__ == "__main__":
    main()
