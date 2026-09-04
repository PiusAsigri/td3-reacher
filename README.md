# TD3-2 - Robotic Arm Target Reaching

Deep RL group project (DSCD 614). A **TD3** agent is trained to drive the
fingertip of a two-jointed arm to a randomly spawned target in Gymnasium's
MuJoCo **Reacher-v5**, and is compared under a controlled protocol against a
**DDPG** baseline (the required baseline) and a random continuous controller
(a performance floor).

> The algorithm implementations (TD3, DDPG, replay buffer) come from
> Stable-Baselines3. Everything else - the environment wrapper, the reward
> function, both baselines' harness integration, the evaluation harness,
> and all logging/plotting/analysis - is the group's own work.

---

## 1. Installation

Requires **Python 3.10+** (`python3 --version`). Create a virtual environment,
then install the pinned dependencies. After the venv is **activated**, `python`
and `pip` refer to the venv interpreter on every platform.

**Linux (Ubuntu/Debian)**
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**macOS** (Python 3.10+ from python.org or `brew install python@3.12`)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MuJoCo ships as a self-contained wheel via `gymnasium[mujoco]`; no separate
MuJoCo binary or licence is required. Verify the environment loads:

```bash
python -c "import gymnasium as gym; gym.make('Reacher-v5'); print('ok')"
```

> **On-screen rollout** (`render_mode="human"`, used only for the demonstration
> video) needs a display and OpenGL. It works on a desktop OS out of the box; on
> a headless Linux server, render to file with `render_mode="rgb_array"` instead.
> Training and evaluation never need a display.

After confirming training runs on your machine, regenerate the exact pins:

```bash
pip freeze > requirements.txt
```

**Troubleshooting (Linux).**
- `python: command not found` — use `python3` to create the venv; once the venv
  is **activated**, `python` and `pip` exist inside it.
- `.venv/bin/pip: No such file or directory` — the venv was created before
  `python3-venv` was installed. Fix it with `python -m ensurepip --upgrade`,
  or recreate: `deactivate; rm -rf .venv; python3 -m venv .venv; source .venv/bin/activate`.
  Deleting `.venv` is harmless since logs and results live in the
  project folder, not the venv.

## 2. Reproduce the headline result (clean environment)

```bash
python run.py --all
```

This trains TD3 and DDPG for every seed in `src/config.py`, evaluates all
methods through the shared harness, writes the aggregate results table, and
regenerates the training-curve figure. On CPU the default budget
(50k steps × 3 seeds × 2 algorithms) takes roughly 15–60 minutes depending on
hardware; reduce `total_timesteps` in `src/config.py` if compute is limited
(reduce steps, not seeds).

Individual stages:

```bash
python run.py --train    # training only  -> logs/, models/
python run.py --eval     # evaluation only -> results/   (needs models/)
python run.py --plot     # figures only    -> figures/   (needs logs/)
python run.py --smoke    # ~1-min end-to-end check that the pipeline runs
```

**Parallel training (use all your cores).** The six training runs
(2 algorithms x 3 seeds) are independent, so they can run concurrently. Each
worker is pinned to one thread, so `N` workers use `N` cores cleanly:

```bash
python run.py --all --parallel        # auto: min(#runs, #cores) workers
python run.py --all --parallel 4      # exactly 4 workers
```

Wall-clock drops roughly in proportion to the number of workers (capped at 6,
the number of runs). Each worker holds its own replay buffer and PyTorch
process, so budget a few hundred MB of RAM per worker. Evaluation and plotting
stay sequential (they are cheap). Omit `--parallel` for the sequential path.

> **Hardware note.** On a 4-core machine, `--parallel 3` often finishes the
> whole job faster than `--parallel 6`, because fewer workers avoid
> oversubscribing the physical cores. Worker processes use the `fork` start
> method (the Linux default); if parallel training stalls at startup on your
> platform, run sequentially by omitting `--parallel`.

## 3. Reproduce each figure / table in the report

| Report artifact                     | Command             | Reads          | Writes                              |
|-------------------------------------|---------------------|----------------|-------------------------------------|
| Fig 1. Training curves (mean±std)   | `python run.py --plot` | `logs/train_*.csv` | `figures/fig1_training_curves.png` |
| Table. Evaluation aggregate         | `python run.py --eval` | `models/*.zip`     | `results/aggregate.csv`, `aggregate.json` |
| Per-seed evaluation numbers         | `python run.py --eval` | `models/*.zip`     | `results/per_seed_summary.csv`      |
| TD3-vs-DDPG effect check            | `python run.py --eval` | (in-run)           | `results/effect_checks.json`        |

Every figure is generated **only** from committed CSV logs, so each figure
traces to raw data in this repository.

## 4. Repository structure

```
td3_reacher/
├── run.py                 # single entry point
├── requirements.txt       # pinned dependencies
├── src/
│   ├── config.py          # ALL hyperparameters, seeds, paths (single source of truth)
│   ├── env.py             # Reacher wrapper: distance/effort in info, success logic
│   ├── reward.py          # reward function with configurable component weights
│   ├── metrics.py         # per-episode metrics + cross-seed aggregation (mean±std)
│   ├── train.py           # SB3 TD3/DDPG training + CSV logging of training return
│   ├── baseline_random.py # random continuous controller (performance floor)
│   ├── evaluate.py        # held-out deterministic evaluation harness
│   └── plots.py           # training-curve figure from committed logs
├── logs/                  # raw training logs (COMMITTED after runs)
├── models/                # saved agent weights (see note below)
├── results/               # evaluation tables (COMMITTED after runs)
└── figures/               # generated figures
```

## 5. Configuration

All experiment settings live in `src/config.py`. The seeds actually run are
`ExperimentConfig.seeds`; exactly these values are documented. Hyperparameters are held
constant across seeds and across TD3/DDPG except for the three TD3-only terms
(`policy_delay`, `target_policy_noise`, `target_noise_clip`), as declared
in the report and in `Hyperparameters_and_Seeds`.

## 6. Model weights

Trained agent weights (used for the demonstration) are committed under models/ - one .zip per algorithm and seed (e.g. TD3_seed0.zip). They are small (~2–3 MB each), so they live directly in the repository; no external download is required.

## 7. Attribution

- Stable-Baselines3 (TD3, DDPG, replay buffer) — https://github.com/DLR-RM/stable-baselines3
- Gymnasium / MuJoCo Reacher-v5 — https://gymnasium.farama.org/environments/mujoco/reacher/

Use of generative AI is declared in `AI_Use_Declaration`.
