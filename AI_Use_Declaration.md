# AI Use Declaration (draft — fill before submission)

The examination requires an honest record of what a tool produced and what
the group still has to verify or rewrite. Replace every TODO with names and
dates. Do not claim original-work modules as group-authored if an assistant
wrote the function bodies.

## Tool

- Tool: Cursor (Grok 4.6)
- Dates: TODO

## What the assistant contributed

- Library configuration and install debugging (venv on Python 3.11; portable
  `requirements.txt`; historical CUDA freeze moved to
  `requirements-cuda-linux.txt`).
- Glue: `--smoke` writes under `scratch/smoke/` so assessed `logs/`, `models/`
  and `results/` are not overwritten.
- README scaffolding (install path, hyperparameter-vs-default table).
- `verify_env.py` (prints Reacher-v5 facts from the installed package).
- Language edit of comments in `src/config.py` (reward-weight default was
  incorrectly described as matching Reacher-v5).
- Review of existing pipeline behaviour (eval reproducibility; TD3 vs DDPG
  gap vs across-seed spread). The assistant did **not** choose a new success
  threshold, reward weights, or hyperparameters after seeing results.

## What must remain original group work

The following must be designed and written by the group. If the current file
bodies were produced by an assistant, rewrite them before submission:

- [ ] MDP formulation and state justification (report)
- [ ] `src/env.py` (observation, action handling, termination/truncation)
- [ ] `src/reward.py` (equation and weights)
- [ ] Baseline (`src/baseline_random.py` and DDPG harness integration)
- [ ] `src/evaluate.py`, `src/metrics.py` (seeding, metrics, aggregation)
- [ ] Logging, plotting, analysis (`src/train.py` logger, `src/plots.py`, report)

## Verification the group still has to perform

- [ ] Run `python verify_env.py` on the machine used for assessed runs and
      paste the output into an appendix.
- [ ] Confirm `success_threshold`, reward weights, and metric definitions were
      frozen **before** looking at success rates (or declare any later change
      as a new experiment, not a headline tweak).
- [ ] Confirm every figure regenerates from committed CSVs:
      `python run.py --plot`.
- [ ] Confirm evaluation uses the same wrapper, seeds, episodes, and metric
      code for TD3, DDPG, and the random controller: `python run.py --eval`.
- [ ] Each member can explain the module they own without reading it aloud.
