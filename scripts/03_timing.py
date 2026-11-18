"""
NETScope - step 3: when does each cell go, and in what order do the two
events happen?

Two questions, with two very different answers:

  1. WHEN does each cell release its DNA?  -> robust, gives a kinetics curve.
  2. Does the DNA collapse happen BEFORE the membrane opens?  -> not
     answerable with this dataset, and the script shows why.

Onset is defined per cell as the half-maximum crossing of that cell's own
trace. Half-max is used rather than "first departure from baseline" because
the peak is 30-150x baseline here, so the midpoint sits far above the noise
and its crossing time is stable.

Produces:
  results/tables/03_timing.csv
  results/figures/03_kinetics.png
  results/figures/03_lag.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FRAMES = 5


def load_config():
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def half_max_crossing(time, signal, baseline, extreme, rising):
    """Time at which `signal` first crosses halfway between baseline and extreme."""
    midpoint = (baseline + extreme) / 2.0
    crossed = signal > midpoint if rising else signal < midpoint
    if not crossed.any():
        return np.nan
    return float(time[int(np.argmax(crossed))])


def timing_table(traces, summary):
    cls = dict(zip(summary["cell_id"], summary["class"]))
    rows = []
    for cell_id, grp in traces.groupby("cell_id"):
        grp = grp.sort_values("frame")
        t = grp["time_min"].to_numpy()
        dna = grp["dna_mean"].to_numpy()
        rel = grp["release_mean"].to_numpy()

        dna_base = dna[:BASELINE_FRAMES].mean()
        rel_base = rel[:BASELINE_FRAMES].mean()

        t_release = half_max_crossing(t, rel, rel_base, rel.max(), rising=True)
        t_dna = half_max_crossing(t, dna, dna_base, dna.min(), rising=False)

        rows.append({
            "cell_id": cell_id,
            "class": cls[cell_id],
            "t_dna_collapse_min": t_dna,
            "t_release_min": t_release,
            "lag_min": t_release - t_dna,
        })
    return pd.DataFrame(rows).sort_values("t_release_min")


def plot_kinetics(timing, interval_min, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # cumulative curve: fraction of cells that have gone by time t
    for name, colour in [("dispersing", "tab:orange"), ("persistent", "tab:blue")]:
        sub = timing[timing["class"] == name].sort_values("t_release_min")
        times = np.concatenate([[0], sub["t_release_min"].to_numpy()])
        frac = np.arange(len(times)) / len(sub)
        axes[0].step(times, frac, where="post", color=colour, lw=2, label=name)
    allsub = timing.sort_values("t_release_min")
    times = np.concatenate([[0], allsub["t_release_min"].to_numpy()])
    axes[0].step(times, np.arange(len(times)) / len(allsub), where="post",
                 color="k", lw=1, ls="--", label="all cells")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Cumulative fraction of cells released")
    axes[0].set_title("Release kinetics")
    axes[0].legend(frameon=False)

    # onset time per cell
    for name, colour in [("dispersing", "tab:orange"), ("persistent", "tab:blue")]:
        sub = timing[timing["class"] == name]
        axes[1].scatter(sub["t_release_min"],
                        np.random.default_rng(0).normal(0 if name == "dispersing" else 1,
                                                        0.06, len(sub)),
                        color=colour, s=40, alpha=0.85)
    axes[1].set_yticks([0, 1], ["dispersing", "persistent"])
    axes[1].set_xlabel("Release half-max time (min)")
    axes[1].set_title("Onset time per cell")
    axes[1].set_ylim(-0.5, 1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lag(timing, interval_min, out_path):
    sub = timing[timing["class"] == "dispersing"].dropna(subset=["lag_min"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].scatter(sub["t_dna_collapse_min"], sub["t_release_min"],
                    color="tab:orange", s=45)
    lo = min(sub["t_dna_collapse_min"].min(), sub["t_release_min"].min()) - 10
    hi = max(sub["t_dna_collapse_min"].max(), sub["t_release_min"].max()) + 10
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("DNA collapse, half-max (min)")
    axes[0].set_ylabel("Release rise, half-max (min)")
    axes[0].set_title("The two events, per cell")

    axes[1].hist(sub["lag_min"], bins=np.arange(-20, 25, interval_min),
                 color="tab:orange", edgecolor="white")
    axes[1].axvline(0, color="k", lw=1)
    axes[1].axvspan(-interval_min, interval_min, color="grey", alpha=0.25)
    axes[1].set_xlabel("Lag: release minus DNA collapse (min)")
    axes[1].set_ylabel("cells")
    axes[1].set_title("Grey band = one sampling interval")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    interval_min = cfg["data"]["frame_interval_s"] / 60.0
    tables = ROOT / cfg["output"]["tables"]
    figures = ROOT / cfg["output"]["figures"]

    traces = pd.read_csv(tables / "02_cell_traces.csv")
    summary = pd.read_csv(tables / "02_cell_summary.csv")

    timing = timing_table(traces, summary)
    timing.to_csv(tables / "03_timing.csv", index=False)

    plot_kinetics(timing, interval_min, figures / "03_kinetics.png")
    plot_lag(timing, interval_min, figures / "03_lag.png")

    print(timing.to_string(index=False))
    for name in ("dispersing", "persistent"):
        sub = timing[timing["class"] == name]
        print(f"\n{name}: n={len(sub)}  release half-max "
              f"median {sub['t_release_min'].median():.0f} min "
              f"(range {sub['t_release_min'].min():.0f}-{sub['t_release_min'].max():.0f})")

    lag = timing[timing["class"] == "dispersing"]["lag_min"].dropna()
    print(f"\nlag (release - DNA collapse), dispersing cells only:")
    print(f"  median {lag.median():.1f} min, IQR {lag.quantile(.25):.1f} to "
          f"{lag.quantile(.75):.1f}, range {lag.min():.0f} to {lag.max():.0f}")
    print(f"  sampling interval is {interval_min:.1f} min: any ordering "
          f"smaller than this is not resolvable")
    within = (lag.abs() <= interval_min).sum()
    print(f"  {within}/{len(lag)} cells have |lag| within one interval")


if __name__ == "__main__":
    main()
