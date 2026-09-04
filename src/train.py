"""
Training  (algorithm supplied by Stable-Baselines3; harness is ORIGINAL WORK).

Trains one algorithm for one seed with the shared hyperparameters from config.py
and logs training return per episode to a CSV that the plotting code reads. The
same function trains both the TD3 agent and the DDPG baseline; only the
algorithm class and the three TD3-only hyperparameters differ, and those
differences are declared in the report.

Attribution: TD3 and DDPG implementations, and the replay buffer, come from
Stable-Baselines3 (see requirements.txt for the pinned version).
"""
from __future__ import annotations
import csv
import numpy as np

import gymnasium as gym
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from .config import CFG, LOG_DIR, MODEL_DIR
from .env import make_env

_ALGOS = {"TD3": TD3, "DDPG": DDPG}


class EpisodeReturnLogger(BaseCallback):
    """Record (timestep, episode, return, length) at each episode end.

    Reads the episode summary that `Monitor` writes into `info["episode"]`.
    """

    def __init__(self):
        super().__init__()
        self.rows: list[tuple] = []
        self._ep = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep += 1
                self.rows.append((self.num_timesteps, self._ep,
                                  float(ep["r"]), int(ep["l"])))
        return True


def _algo_kwargs(algo: str) -> dict:
    t = CFG.train
    common = dict(
        learning_rate=t.learning_rate,
        buffer_size=t.buffer_size,
        learning_starts=t.learning_starts,
        batch_size=t.batch_size,
        tau=t.tau,
        gamma=t.gamma,
        train_freq=t.train_freq,
        gradient_steps=t.gradient_steps,
        policy_kwargs=dict(net_arch=list(t.net_arch)),
        verbose=0,
    )
    if algo == "TD3":
        common.update(
            policy_delay=t.policy_delay,
            target_policy_noise=t.target_policy_noise,
            target_noise_clip=t.target_noise_clip,
        )
    return common


def train_one(algo: str, seed: int) -> dict:
    """Train `algo` with `seed`; write log + weights; return a small summary."""
    assert algo in _ALGOS, algo
    set_random_seed(seed)

    env = Monitor(make_env(CFG.env, seed=seed))
    action_dim = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=CFG.train.exploration_sigma * np.ones(action_dim),
    )

    model = _ALGOS[algo](
        "MlpPolicy", env, seed=seed, action_noise=action_noise,
        **_algo_kwargs(algo),
    )

    logger = EpisodeReturnLogger()
    model.learn(total_timesteps=CFG.train.total_timesteps, callback=logger,
                progress_bar=False)

    # Persist raw training log (committed to the repo; figures trace to this).
    log_path = LOG_DIR / f"train_{algo}_seed{seed}.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "episode", "return", "length"])
        w.writerows(logger.rows)

    model_path = MODEL_DIR / f"{algo}_seed{seed}.zip"
    model.save(model_path)
    env.close()
    return {"algo": algo, "seed": seed, "episodes": len(logger.rows),
            "log": str(log_path), "model": str(model_path)}


def train_all() -> list[dict]:
    out = []
    for algo in CFG.algorithms:
        for seed in CFG.seeds:
            print(f"[train] {algo} seed={seed} "
                  f"steps={CFG.train.total_timesteps} ...")
            out.append(train_one(algo, seed))
    return out


# ---------------------------------------------------------------------------
# Parallel training across CPU cores
# ---------------------------------------------------------------------------
def _train_one_worker(job):
    """Pool worker: pin this process to one thread, then train one (algo, seed).

    Pinning to a single thread means `n` worker processes map onto `n` cores
    without oversubscription (each run is single-threaded; we run many at once).
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    algo, seed = job
    return train_one(algo, seed)


def train_all_parallel(n_workers: int | None = None) -> list[dict]:
    """Train all (algorithm, seed) runs concurrently across worker processes.

    n_workers=None (or <=0) auto-selects min(#runs, #cpu cores). Results are
    identical to train_all(); only the wall-clock differs, because each run is
    independent and writes its own log + weights.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(algo, seed) for algo in CFG.algorithms for seed in CFG.seeds]
    if not n_workers or n_workers <= 0:
        n_workers = min(len(jobs), os.cpu_count() or 1)
    n_workers = min(n_workers, len(jobs))

    print(f"[train] {len(jobs)} runs across {n_workers} worker process(es) "
          f"| steps/run={CFG.train.total_timesteps}")
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_train_one_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            algo, seed = futures[fut]
            r = fut.result()   # re-raises worker exceptions here
            print(f"[train] done {algo} seed={seed}: {r['episodes']} episodes")
            results.append(r)
    return results


if __name__ == "__main__":
    for r in train_all():
        print(r)
