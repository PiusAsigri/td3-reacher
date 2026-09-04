"""
Plotting  (ORIGINAL GROUP WORK).

Every figure is generated ONLY from the committed CSV logs, so each figure
traces back to raw data in the repository, as the brief requires. No plot reads
live training state.

Figure 1: training return vs environment steps, mean across seeds with a shaded
+/- 1 std band, one line per algorithm.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CFG, LOG_DIR, FIG_DIR


def _binned_curve(seeds, algo, n_bins=100):
    """Interpolate each seed's (timestep, return) onto a common grid, then take
    the mean and std across seeds at each grid point."""
    curves, max_t = [], 0
    for seed in seeds:
        path = LOG_DIR / f"train_{algo}_seed{seed}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        curves.append(df)
        max_t = max(max_t, df["timestep"].max())
    if not curves:
        return None
    grid = np.linspace(0, max_t, n_bins)
    stacked = []
    for df in curves:
        # smooth episode returns a little before interpolating
        y = df["return"].rolling(10, min_periods=1).mean().to_numpy()
        stacked.append(np.interp(grid, df["timestep"].to_numpy(), y))
    stacked = np.vstack(stacked)
    return grid, stacked.mean(0), stacked.std(0, ddof=1) if len(stacked) > 1 else np.zeros_like(grid)


def plot_training_curves(out_path=None):
    out_path = out_path or (FIG_DIR / "fig1_training_curves.png")
    plt.figure(figsize=(7, 4.5))
    for algo in CFG.algorithms:
        res = _binned_curve(CFG.seeds, algo)
        if res is None:
            print(f"[plot] no logs for {algo}; skipping")
            continue
        grid, mean, std = res
        plt.plot(grid, mean, label=algo)
        plt.fill_between(grid, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Environment steps")
    plt.ylabel("Episode return (training)")
    plt.title(f"Training return on {CFG.env.env_id} "
              f"(mean ± 1 std over {len(CFG.seeds)} seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    plot_training_curves()
