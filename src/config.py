"""
Central configuration: the single source of truth for the experiment.

Every hyperparameter, seed and path used anywhere in the project is defined here
so that a reader can audit the full configuration in one place and so that the
training, evaluation and plotting code cannot silently disagree.

"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
for _d in (LOG_DIR, MODEL_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "Reacher-v5"
    # Distance (in MuJoCo length units) below which the fingertip is judged to
    # have "reached" the target. This is a DESIGN CHOICE used only to compute
    # the success-rate and time-to-target evaluation metrics; justify it in the
    # report and report a sensitivity check.
    success_threshold: float = 0.05
    # If True, the episode terminates the moment distance < success_threshold.
    # We keep this False for the headline result so that training is not
    # confounded by an altered termination structure (see report, Methodology).
    success_termination: bool = False
    # Reward component weights. Defaults match the Reacher-v5 defaults so the
    # headline agent is comparable with the literature; exposed here so the
    # reward-shaping ablation can vary them.
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 0.1
    success_bonus: float = 0.0  # optional terminal bonus for reaching target


# ---------------------------------------------------------------------------
# Shared training hyperparameters (held CONSTANT across seeds and algorithms)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 50_000   
    learning_rate: float = 1e-3
    buffer_size: int = 200_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    # gamma chosen against the effective horizon: 1/(1-0.98) = 50 steps, which
    # matches the 50-step Reacher episode. Justified in the report.
    gamma: float = 0.98
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: tuple = (256, 256)
    exploration_sigma: float = 0.1   # std of Gaussian action noise (training only)

    # --- TD3-specific (ignored by DDPG) ---
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    n_eval_episodes: int = 30        # rubric minimum; per seed
    deterministic: bool = True       # exploration disabled at evaluation
    # Evaluation episodes are drawn from a seed stream disjoint from training so
    # that no evaluated target configuration was seen during training.
    eval_seed_offset: int = 10_000


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple = (0, 1, 2)                 # >= 3 seeds; REPORT THESE VALUES
    algorithms: tuple = ("TD3", "DDPG")      # agent + required baseline
    include_random_baseline: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


CFG = ExperimentConfig()


def config_dict() -> dict:
    """Flat-ish dict of the whole configuration, for logging into result files."""
    return asdict(CFG)


if __name__ == "__main__":
    import json
    print(json.dumps(config_dict(), indent=2, default=str))
