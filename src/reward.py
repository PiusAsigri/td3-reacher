"""
Reward function  (ORIGINAL GROUP WORK).

We replace the environment's built-in reward with our own so that the weighting
of the component terms is explicit and under our control, as required by the
brief. The functional form matches Reacher's standard dense reward; the point of
owning it is that the reward-shaping ablation in the report can vary the weights
and the optional success bonus from a single place.

Reward equation (written also in the report):

    r_t = - w_dist * d_t  -  w_ctrl * ||a_t||^2  +  b * 1[d_t < eps]

where
    d_t      = ||fingertip_t - target||   (xy-plane distance)
    ||a_t||^2 = squared L2 norm of the torque action
    w_dist   = reward_dist_weight
    w_ctrl   = reward_control_weight
    b        = success_bonus (0 for the headline configuration)
    eps      = success_threshold
"""
from __future__ import annotations
import gymnasium as gym

from .config import EnvConfig


class ShapedRewardWrapper(gym.Wrapper):
    """Recompute reward from the distance / control terms exposed upstream.

    Requires `ReacherMetricWrapper` to run first so that `info["distance"]` and
    `info["control_effort"]` are populated.
    """

    def __init__(self, env: gym.Env, cfg: EnvConfig):
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        d = info["distance"]
        ctrl = info["control_effort"]
        c = self.cfg
        reward = -(c.reward_dist_weight * d) - (c.reward_control_weight * ctrl)
        if info.get("is_success", False):
            reward += c.success_bonus
        # keep components in info for transparency / debugging
        info["r_dist"] = -(c.reward_dist_weight * d)
        info["r_ctrl"] = -(c.reward_control_weight * ctrl)
        return obs, reward, terminated, truncated, info
