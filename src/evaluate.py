"""
Evaluation harness  (ORIGINAL GROUP WORK).

Runs the required protocol identically for every method (TD3, DDPG, random):

  * >= 30 evaluation episodes per seed (config.eval.n_eval_episodes);
  * exploration disabled (deterministic=True) for the learned agents;
  * evaluation episodes drawn from a seed stream disjoint from training
    (eval seed = seed + eval_seed_offset), so no evaluated target configuration
    was seen during training;
  * the SAME rollout + metric code path for the agent and every baseline.

Outputs, all written to results/ and committed:
  * per-episode CSV per (method, seed)
  * per-seed summary CSV
  * headline aggregate (mean +/- std over seeds) as JSON and CSV
  * an effect check (does the TD3-DDPG gap exceed the across-seed spread?)
"""
from __future__ import annotations
import csv
import json

from stable_baselines3 import TD3, DDPG

from .config import CFG, MODEL_DIR, RESULTS_DIR
from .env import make_env
from .metrics import (EpisodeRecord, episode_metrics, mean_over_episodes,
                      aggregate_over_seeds, difference_exceeds_spread)
from .baseline_random import RandomController

_ALGOS = {"TD3": TD3, "DDPG": DDPG}


def _load_policy(method: str, seed: int, action_space):
    if method == "random":
        return RandomController(action_space, seed=seed + CFG.eval.eval_seed_offset)
    model_path = MODEL_DIR / f"{method}_seed{seed}.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run training before evaluation.")
    return _ALGOS[method].load(model_path)


def _rollout(policy, env) -> EpisodeRecord:
    obs, info = env.reset()
    distances, efforts = [info["distance"]], []
    total_reward, length = 0.0, 0
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=CFG.eval.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        distances.append(info["distance"])
        efforts.append(info["control_effort"])
        total_reward += reward
        length += 1
        done = terminated or truncated
    return EpisodeRecord(distances=distances, control_efforts=efforts,
                         reward=total_reward, length=length)


def evaluate_method_seed(method: str, seed: int) -> dict:
    """Return the per-seed mean metrics for one (method, seed)."""
    eval_seed = seed + CFG.eval.eval_seed_offset
    env = make_env(CFG.env, seed=eval_seed)
    policy = _load_policy(method, seed, env.action_space)

    per_episode = []
    per_episode_path = RESULTS_DIR / f"eval_{method}_seed{seed}.csv"
    with open(per_episode_path, "w", newline="") as f:
        writer = None
        for ep in range(CFG.eval.n_eval_episodes):
            rec = _rollout(policy, env)
            m = episode_metrics(rec, CFG.env.success_threshold)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=["episode", *m.keys()])
                writer.writeheader()
            writer.writerow({"episode": ep, **m})
            per_episode.append(m)
    env.close()
    return mean_over_episodes(per_episode)


def evaluate_all() -> dict:
    methods = list(CFG.algorithms)
    if CFG.include_random_baseline:
        methods.append("random")

    # method -> list of per-seed summaries
    per_seed_all: dict[str, list[dict]] = {m: [] for m in methods}
    rows = []
    for method in methods:
        for seed in CFG.seeds:
            print(f"[eval] {method} seed={seed} "
                  f"({CFG.eval.n_eval_episodes} episodes) ...")
            summary = evaluate_method_seed(method, seed)
            per_seed_all[method].append(summary)
            rows.append({"method": method, "seed": seed, **summary})

    # per-seed table
    with open(RESULTS_DIR / "per_seed_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # headline aggregate (mean +/- std over seeds)
    aggregate = {m: aggregate_over_seeds(s) for m, s in per_seed_all.items()}
    with open(RESULTS_DIR / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # flat aggregate CSV for the report table
    with open(RESULTS_DIR / "aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "mean", "std", "n_seeds"])
        for method, agg in aggregate.items():
            for metric, v in agg.items():
                w.writerow([method, metric, v["mean"], v["std"], v["n_seeds"]])

    # effect check: TD3 vs DDPG on the headline metrics
    checks = []
    if "TD3" in aggregate and "DDPG" in aggregate:
        for metric in ("return", "final_distance", "success", "control_effort"):
            checks.append(difference_exceeds_spread(
                aggregate["TD3"], aggregate["DDPG"], metric))
    with open(RESULTS_DIR / "effect_checks.json", "w") as f:
        json.dump(checks, f, indent=2)

    return {"aggregate": aggregate, "effect_checks": checks}


if __name__ == "__main__":
    out = evaluate_all()
    print(json.dumps(out, indent=2))
