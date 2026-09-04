# TD3-2 - Robotic Arm Target Reaching

Deep RL group project (DSCD 614). A **TD3** agent is trained to drive the
fingertip of a two-jointed arm to a randomly spawned target in Gymnasium's
MuJoCo **Reacher-v5**, and is compared under a controlled protocol against a
**DDPG** baseline (the required baseline) and a random continuous controller
(a performance floor).

The algorithm implementations (TD3, DDPG, replay buffer) come from
Stable-Baselines3. Everything else - the environment wrapper, the reward
function, both baselines' harness integration, the evaluation harness,
and all logging/plotting/analysis - is the group's own work.

---

## 1. Installation

Requires **Python 3.10–3.12**. **Python 3.14 is not verified** for Gymnasium /
Stable-Baselines3 / Torch on this project; on macOS use Homebrew `python3.11`
if `python3` is 3.14.

Create a virtual environment, then install the **portable** pins (no CUDA).
After the venv is **activated**, `python` and `pip` refer to the venv.

**Linux (Ubuntu/Debian)**
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**macOS** (prefer `python3.11` from Homebrew if the default is 3.14)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` is a CPU/macOS-safe freeze. A historical Linux+NVIDIA
`pip freeze` is kept as `requirements-cuda-linux.txt` for audit; installing
that file on macOS will fail.

MuJoCo ships as a wheel; no separate MuJoCo binary or licence is required.
Verify against the **installed** package, not against memory or this README:

```bash
python verify_env.py
```

> **On-screen rollout** (`render_mode="human"`, demonstration video only)
> needs a display and OpenGL. Training and evaluation never need a display.

After a successful install on the machine that will produce **assessed** runs,
commit a lock file from that machine:

```bash
pip freeze > requirements-lock-$(python -c "import sys,platform; print(platform.system()+'-'+sys.platform)").txt
```

**Troubleshooting (Linux / macOS).**
- Default `python3` is 3.14 — create the venv with `python3.11` (this project
  was verified on 3.11.14). Do not assume 3.14 wheels exist for SB3/Torch.
- `python: command not found` — use `python3` / `python3.11` to create the venv;
  once the venv is **activated**, `python` and `pip` exist inside it.
- `.venv/bin/pip: No such file or directory` — the venv was created before
  `python3-venv` was installed. Fix it with `python -m ensurepip --upgrade`,
  or recreate: `deactivate; rm -rf .venv; python3.11 -m venv .venv; source .venv/bin/activate`.
  Deleting `.venv` loses nothing — your code, logs and results live in the
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
python run.py --smoke    # tiny check; writes under scratch/smoke/ only
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
├── run.py                      # single entry point
├── verify_env.py               # print Reacher-v5 facts from installed packages
├── requirements.txt            # portable pins (no CUDA)
├── requirements-cuda-linux.txt # historical Linux+NVIDIA freeze (audit only)
├── AI_Use_Declaration.md
├── src/
│   ├── config.py               # ALL hyperparameters, seeds, paths
│   ├── env.py                  # Reacher wrapper
│   ├── reward.py               # reward weights
│   ├── metrics.py              # per-episode metrics + cross-seed aggregation
│   ├── train.py                # SB3 TD3/DDPG training + CSV logging
│   ├── baseline_random.py      # random continuous controller
│   ├── evaluate.py             # held-out deterministic evaluation
│   └── plots.py                # figures from committed logs
├── logs/                       # raw training logs (COMMITTED)
├── models/                     # saved weights
├── results/                    # evaluation tables (COMMITTED)
└── figures/                    # generated figures
```

## 5. Configuration

All experiment settings live in `src/config.py`. The seeds actually run are
`ExperimentConfig.seeds`; exactly these values are documented. Hyperparameters are held
constant across seeds and across TD3/DDPG except for the three TD3-only terms
(`policy_delay`, `target_policy_noise`, `target_noise_clip`), as declared
in the report, in `Hyperparameters_and_Seeds`, and in the table in section 8.

## 6. Model weights

Trained agent weights (used for the demonstration) are committed under models/ - one .zip per algorithm and seed (e.g. TD3_seed0.zip). They are small (~2–3 MB each), so they live directly in the repository; no external download is required.

## 7. Attribution

- Stable-Baselines3 (TD3, DDPG, replay buffer) — https://github.com/DLR-RM/stable-baselines3
- Gymnasium / MuJoCo Reacher-v5 — https://gymnasium.farama.org/environments/mujoco/reacher/

Use of generative AI is declared in `AI_Use_Declaration.md`.

## 8. Hyperparameters vs Stable-Baselines3 2.9.0 defaults

Verified against the installed `stable_baselines3==2.9.0` constructors (re-check
with `python verify_env.py` after any upgrade). Only rows that **differ** from
the library default, plus the three TD3-only knobs, belong in the report table.
Do not present library defaults as if they were a search.

| Setting | This project (`src/config.py`) | SB3 2.9.0 TD3 default | Same? |
|---|---|---|---|
| `learning_rate` | `1e-3` | `0.001` | yes |
| `buffer_size` | `200_000` | `1_000_000` | **no** |
| `learning_starts` | `1_000` | `100` | **no** |
| `batch_size` | `256` | `256` | yes |
| `tau` | `0.005` | `0.005` | yes |
| `gamma` | `0.98` | `0.99` | **no** (justified vs 50-step episode) |
| `train_freq` / `gradient_steps` | `1` / `1` | `1` / `1` | yes |
| `net_arch` (MlpPolicy) | `[256, 256]` | `[400, 300]` | **no** |
| action noise | `NormalActionNoise(σ=0.1)` | `None` | **no** |
| `policy_delay` (TD3 only) | `2` | `2` | yes |
| `target_policy_noise` (TD3 only) | `0.2` | `0.2` | yes |
| `target_noise_clip` (TD3 only) | `0.5` | `0.5` | yes |

Held constant across seeds and across TD3/DDPG except the three TD3-only terms.

**Do not retune these after looking at evaluation success rates.** If a search
was done, document range, which seed was used, and that evaluation episodes
were not used for selection.

## 9. What `--smoke` does not prove

`python run.py --smoke` trains 2,000 steps on two seeds and writes under
`scratch/smoke/`. It must not overwrite `logs/`, `models/`, or `results/`.
Smoke output is not a headline result. Headline numbers come from the
committed 50,000-step logs and `python run.py --eval`.
