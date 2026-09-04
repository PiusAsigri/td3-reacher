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


def set_artifact_dirs(root: Path) -> None:
    """Redirect logs/models/figures/results (used by --smoke so assessed artifacts stay intact).

    Must be called *before* importing src.train / src.evaluate / src.plots, because
    those modules bind LOG_DIR etc. at import time.
    """
    global LOG_DIR, MODEL_DIR, FIG_DIR, RESULTS_DIR
    root = Path(root)
    LOG_DIR = root / "logs"
    MODEL_DIR = root / "models"
    FIG_DIR = root / "figures"
    RESULTS_DIR = root / "results"
    for d in (LOG_DIR, MODEL_DIR, FIG_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


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
    # Reward component weights. These are a GROUP DESIGN CHOICE.
    # Do not change them after seeing evaluation numbers: that is HARKing.
    #
    # Verified against gymnasium 1.3.0 on the machine that ran verify_env.py:
    # ReacherEnv.__init__ defaults are reward_dist_weight=1, reward_control_weight=1.
    # The Gymnasium docstring still says control weight 0.1; the constructor
    # default is 1. Headline runs in this repo used control weight 0.1, which
    # is *not* the installed native default. State that honestly in the report.
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 0.1
    success_bonus: float = 0.0  # optional bonus; 0 in the headline configuration


# ---------------------------------------------------------------------------
# Shared training hyperparameters (held CONSTANT across seeds and algorithms)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 50_000   
    learning_rate: float = 1e-3
    buffer_size: int = 200_000        # SB3 2.9.0 default is 1_000_000
    learning_starts: int = 1_000      # SB3 2.9.0 default is 100
    batch_size: int = 256
    tau: float = 0.005
    # gamma chosen against the effective horizon: 1/(1-0.98) = 50 steps, which
    # matches the 50-step Reacher episode. Justified in the report.
    gamma: float = 0.98
    train_freq: int = 1
    gradient_steps: int = 1
    # Differs from SB3 2.9.0 MlpPolicy default [400, 300]; declare in the report.
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
