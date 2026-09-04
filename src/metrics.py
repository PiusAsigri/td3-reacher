"""
Metric computation and aggregation  (ORIGINAL GROUP WORK).

Per-episode metrics are computed from a trajectory record; per-seed metrics are
the mean over that seed's evaluation episodes; the headline figure for each
metric is the mean over seeds reported WITH the standard deviation over seeds.
`aggregate_over_seeds` always returns (mean, std, n_seeds) as required by the task.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class EpisodeRecord:
    """Everything needed to score one evaluation episode."""
    distances: list          # d_t for t = 1..T
    control_efforts: list    # ||a_t||^2 for t = 1..T
    reward: float            # cumulative *environment* reward for the episode
    length: int


def episode_metrics(rec: EpisodeRecord, success_threshold: float) -> dict:
    """Reduce one trajectory to the reported per-episode metrics."""
    d = np.asarray(rec.distances, dtype=float)
    reached = np.where(d < success_threshold)[0]
    return {
        "return": rec.reward,
        "final_distance": float(d[-1]),
        "min_distance": float(d.min()),
        "success": float(d[-1] < success_threshold),   # success = reached at end
        # time-to-target: first step index (1-based) at which target is reached;
        # NaN if never reached, so it is averaged only over successful episodes.
        "time_to_target": float(reached[0] + 1) if reached.size else np.nan,
        "control_effort": float(np.mean(rec.control_efforts)),
        "length": rec.length,
    }


def mean_over_episodes(per_episode: Sequence[dict]) -> dict:
    """Per-seed summary: mean of each metric over that seed's episodes.

    time_to_target is averaged over successful episodes only (NaNs ignored).
    """
    keys = per_episode[0].keys()
    out = {}
    for k in keys:
        vals = np.asarray([m[k] for m in per_episode], dtype=float)
        if k == "time_to_target":
            out[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
        else:
            out[k] = float(np.mean(vals))
    return out


def aggregate_over_seeds(per_seed: Sequence[dict]) -> dict:
    """Headline summary: mean and std of each metric across seeds."""
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        vals = np.asarray([s[k] for s in per_seed], dtype=float)
        finite = vals[~np.isnan(vals)]
        if finite.size == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n_seeds": 0}
        else:
            out[k] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "n_seeds": int(finite.size),
            }
    return out


def difference_exceeds_spread(agg_a: dict, agg_b: dict, metric: str) -> dict:
    """Cheap effect check the brief asks for: does |mean_A - mean_B| exceed the
    combined across-seed spread? This is a descriptive check, not a significance
    test, and should be reported as such.
    """
    a, b = agg_a[metric], agg_b[metric]
    diff = a["mean"] - b["mean"]
    combined = float(np.hypot(a["std"], b["std"]))
    return {
        "metric": metric,
        "diff": diff,
        "combined_std": combined,
        "exceeds_spread": bool(abs(diff) > combined),
    }
