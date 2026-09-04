"""
Random continuous controller  (ORIGINAL GROUP WORK).

A sanity-floor baseline. It samples torques uniformly from the action space and
exposes a `predict(obs, deterministic=...)` method so that the SAME evaluation
harness that scores the learned agents also scores it, through the same code
path, as the brief requires.

The primary required baseline for TD3-2 is DDPG (trained through train.py), because TD3 is DDPG plus three specific
fixes and so the TD3-vs-DDPG comparison isolates the algorithm's contribution.
The random controller establishes the performance floor.
"""
from __future__ import annotations
import numpy as np


class RandomController:
    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        # `deterministic` is accepted for interface compatibility; a random
        # controller has no deterministic mode, which is noted in the project report.
        return self.action_space.sample(), None
