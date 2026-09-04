#!/usr/bin/env python3
"""Print Reacher-v5 facts from the *installed* packages, not from memory.

Run after `pip install -r requirements.txt`:

    python verify_env.py

If any printed value disagrees with the report or with src/config.py comments,
the installed version wins. Re-check after any dependency change.
"""
from __future__ import annotations

import inspect

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
import torch
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv
from gymnasium.envs.registration import registry


def main() -> None:
    print("=== package versions ===")
    print("gymnasium", gym.__version__)
    print("mujoco", mujoco.__version__)
    print("stable_baselines3", stable_baselines3.__version__)
    print("torch", torch.__version__)

    spec = registry["Reacher-v5"]
    print("\n=== registry: Reacher-v5 ===")
    print("entry_point:", spec.entry_point)
    print("max_episode_steps:", spec.max_episode_steps)
    print("reward_threshold:", spec.reward_threshold)
    print("kwargs:", spec.kwargs)

    print("\n=== ReacherEnv.__init__ defaults (source constructor) ===")
    sig = inspect.signature(ReacherEnv.__init__)
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            print(f"  {name}={param.default!r}")

    env = gym.make("Reacher-v5")
    u = env.unwrapped
    print("\n=== live env ===")
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("action low:", env.action_space.low, "high:", env.action_space.high)
    print("reward_dist_weight:", u._reward_dist_weight)
    print("reward_control_weight:", u._reward_control_weight)
    print("frame_skip:", u.frame_skip, "dt:", u.dt)
    print("TimeLimit _max_episode_steps:", getattr(env, "_max_episode_steps", None))

    wrappers = []
    e = env
    while hasattr(e, "env"):
        wrappers.append(type(e).__name__)
        e = e.env
    wrappers.append(type(e).__name__)
    print("wrapper stack:", " -> ".join(wrappers))

    obs, info = env.reset(seed=0)
    print("\n=== reset(seed=0) ===")
    print("obs shape", obs.shape, "dtype", obs.dtype)
    print("obs", obs)
    print("info", info)

    obs2, reward, terminated, truncated, info2 = env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
    print("\n=== one zero-action step ===")
    print("reward", reward, "terminated", terminated, "truncated", truncated)
    print("info", info2)

    env.reset(seed=1)
    n = 0
    term = trunc = False
    while True:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        n += 1
        if term or trunc:
            break
    print("\n=== random episode end ===")
    print("steps", n, "terminated", term, "truncated", trunc)

    print("\n=== _get_obs layout (from installed reacher_v5.py) ===")
    print("file:", inspect.getfile(ReacherEnv))
    print("concat order: cos(theta[2]), sin(theta[2]), qpos[2:] (target xy),")
    print("              qvel[:2], (fingertip - target)[:2]  -> 10 dims")

    env.close()
    print("\nverify_env: ok")


if __name__ == "__main__":
    main()
