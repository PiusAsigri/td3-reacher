"""
Environment construction and wrappers  (ORIGINAL GROUP WORK).

The TD3/DDPG algorithms are supplied by Stable-Baselines3, but the environment
definition, the observation/metric plumbing and the termination logic are ours.

What this module adds on top of Gymnasium's Reacher-v5:

1. `ReacherMetricWrapper` — computes the true 2-D fingertip->target distance
   directly from the MuJoCo bodies (robust to reward-weight changes) and the
   per-step control effort ||a||^2, and writes both into `info` so the
   evaluation harness can compute success rate, final distance, time-to-target
   and control effort without depending on the reward internals. It also
   optionally terminates the episode on success.

Design note (see report, Methodology): Reacher-v5 by default only *truncates*
at 50 steps and never *terminates*. We keep `success_termination=False` for the
headline run so training is not confounded by an altered episode structure, and
we treat "success" purely as an evaluation-time property of the trajectory.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym

from .config import EnvConfig


class ReacherMetricWrapper(gym.Wrapper):
    """Expose distance + control-effort in `info`; optional success termination.

    Adds to every `info` dict:
        info["distance"]        float, ||fingertip - target|| in the xy-plane
        info["control_effort"]  float, ||action||^2
        info["is_success"]      bool,  distance < success_threshold
    """

    def __init__(self, env: gym.Env, cfg: EnvConfig):
        super().__init__(env)
        self.cfg = cfg

    def _distance(self) -> float:
        u = self.env.unwrapped
        fingertip = u.get_body_com("fingertip")[:2]
        target = u.get_body_com("target")[:2]
        return float(np.linalg.norm(fingertip - target))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dist = self._distance()
        info["distance"] = dist
        info["control_effort"] = float(np.sum(np.square(action)))
        info["is_success"] = bool(dist < self.cfg.success_threshold)
        if self.cfg.success_termination and info["is_success"]:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["distance"] = self._distance()
        info["control_effort"] = 0.0
        info["is_success"] = bool(info["distance"] < self.cfg.success_threshold)
        return obs, info


def make_env(cfg: EnvConfig, seed: int | None = None, render_mode: str | None = None):
    """Build a single wrapped Reacher environment.

    The reward wrapper is applied on top so that the reward weighting is under
    our control; see reward.py.
    """
    from .reward import ShapedRewardWrapper  # local import to avoid cycle

    env = gym.make(cfg.env_id, render_mode=render_mode)
    env = ReacherMetricWrapper(env, cfg)
    env = ShapedRewardWrapper(env, cfg)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


if __name__ == "__main__":
    # Smoke test
    cfg = EnvConfig()
    env = make_env(cfg, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,), obs.shape
    total = 0.0
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        assert {"distance", "control_effort", "is_success"} <= set(info)
        if term or trunc:
            break
    print("smoke ok | last distance", round(info["distance"], 4),
          "| return", round(total, 3), "| trunc", trunc)
