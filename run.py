#!/usr/bin/env python3
"""
Single entry point.

Reproduces the headline result from a clean environment:

    python run.py --all        # train all seeds, evaluate, plot, print table
    python run.py --train      # training only
    python run.py --eval       # evaluation only (requires saved models)
    python run.py --plot       # figures only (requires committed logs)
    python run.py --smoke      # 2k-step end-to-end check that nothing is broken

Seeds, hyperparameters and the evaluation protocol are fixed in src/config.py.
"""
from __future__ import annotations
import argparse
import json

from src.config import CFG, RESULTS_DIR


def _print_headline(aggregate: dict):
    print("\n================ HEADLINE (mean ± std over seeds) ================")
    metrics = ["return", "final_distance", "success", "time_to_target",
               "control_effort"]
    header = f"{'method':>8} | " + " | ".join(f"{m:>15}" for m in metrics)
    print(header)
    print("-" * len(header))
    for method, agg in aggregate.items():
        cells = []
        for m in metrics:
            v = agg[m]
            cells.append(f"{v['mean']:7.3f}±{v['std']:5.3f}")
        print(f"{method:>8} | " + " | ".join(f"{c:>15}" for c in cells))
    print("==================================================================\n")


def main():
    p = argparse.ArgumentParser(description="TD3-2 Robotic Arm Target Reaching")
    p.add_argument("--all", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--parallel", nargs="?", type=int, const=0, default=None,
                   metavar="N",
                   help="train runs concurrently across CPU cores. "
                        "'--parallel' auto-picks min(#runs, #cores); "
                        "'--parallel 4' uses 4 workers. Omit for sequential.")
    args = p.parse_args()

    if args.smoke:
        # Fast end-to-end path with tiny budgets to prove the pipeline runs.
        import dataclasses
        from src import train as train_mod, evaluate as eval_mod, plots as plot_mod
        object.__setattr__(CFG.train, "total_timesteps", 2000)
        object.__setattr__(CFG.train, "learning_starts", 200)
        object.__setattr__(CFG.eval, "n_eval_episodes", 3)
        object.__setattr__(CFG, "seeds", (0, 1))
        object.__setattr__(CFG, "algorithms", ("TD3", "DDPG"))
        train_mod.train_all()
        out = eval_mod.evaluate_all()
        plot_mod.plot_training_curves()
        _print_headline(out["aggregate"])
        return

    do_all = args.all or not (args.train or args.eval or args.plot)
    if args.train or do_all:
        if args.parallel is not None:
            from src.train import train_all_parallel
            train_all_parallel(n_workers=args.parallel)
        else:
            from src.train import train_all
            train_all()
    if args.eval or do_all:
        from src.evaluate import evaluate_all
        out = evaluate_all()
        _print_headline(out["aggregate"])
        print("Effect checks (TD3 vs DDPG):")
        print(json.dumps(out["effect_checks"], indent=2))
    if args.plot or do_all:
        from src.plots import plot_training_curves
        plot_training_curves()

    print(f"\nArtifacts written under: results/, figures/, logs/, models/")
    print(f"Seeds: {CFG.seeds} | steps/seed: {CFG.train.total_timesteps}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
