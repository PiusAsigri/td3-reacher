# Report outline — DSCD614 TD3-2 Robotic Arm Target Reaching

**This is not the report.** Maximum 4,000 words excluding references and
appendices. You write every section. An assistant must not draft formulation,
justification, or analysis in your voice. Use this file as a structure, a
word budget, and a list of obligations. When a paragraph is done, delete the
corresponding TODO.

Paste `python verify_env.py` output into Appendix A after you run it on the
machine that produced (or can load) the assessed results. Numbers in §5 come
from committed `results/aggregate.json` (mean ± sample std over seeds 0, 1, 2).
Re-copy them from that file; do not round in a way that hides DDPG ≈ TD3.

---

## Word budget (4,000 words)

Marks sit on formulation, protocol, evaluation, and reporting (56). Algorithm
implementation is 10. Do not spend the word count retelling SB3 internals.

| § | Section | Words | Why this share |
|---|---------|------:|----------------|
| 1 | Introduction | 350 | Scope, question, headline result in one sentence |
| 2 | Background | 450 | Only what you use: TD3 vs DDPG, Reacher, Gymnasium API |
| 3 | Problem formulation | 800 | 16 marks. MDP must be yours |
| 4 | Methodology | 850 | Protocol, seeds, baseline identity, library versions |
| 5 | Results | 550 | All five metrics with spread; no best-seed headline |
| 6 | Discussion | 500 | TD3 vs DDPG vs random; stability; what the gap cannot support |
| 7 | Limitations and deployment | 400 | Robotic manipulation, not “we had limited time” |
| 8 | Conclusion and further work | 100 | Three sentences, then one honest next experiment |
| | **Body total** | **4,000** | |
| | References | excluded | Look up every citation; do not copy guessed venues |
| | Appendices | excluded | Config dump, per-seed table, verify_env, AI use |

If you overrun, cut Background and Conclusion first. Never cut §3 or the
seed/eval paragraphs in §4.

---

## Standing bans (examiner will look for these)

- Do not report the best seed as the result. TD3 seed 2 return is better than
  seed 0; the headline is the **mean ± std over three seeds**.
- Do not write “TD3 outperforms DDPG.” On every headline metric the
  TD3−DDPG gap is smaller than the combined across-seed spread
  (`results/effect_checks.json`, `exceeds_spread: false`).
- Do not invent p-values. n = 3 seeds cannot support a significance test.
  The effect check is descriptive only; say so.
- Do not claim the control-weight 0.1 matches Gymnasium Reacher-v5. On
  gymnasium 1.3.0 the **constructor** default is `reward_control_weight=1`
  (the docstring still says 0.1). Headline runs used `w_ctrl = 0.1`.
- Do not conflate termination and truncation. Reacher **never terminates**;
  it **truncates** at 50 steps. Headline training used
  `success_termination=False`.
- Do not cite a paper unless you have opened it and copied the real
  bibliographic fields. If unsure, omit.

---

## 1. Introduction (~350 words)

**Job.** One paragraph on the task, one on the question, one on the protocol,
one on the result. The result sentence must match §5.

**Must include**

- Environment: Gymnasium MuJoCo **Reacher-v5**, two-joint arm, continuous
  torques, random target.
- Algorithm: TD3 from Stable-Baselines3 (state version in §4).
- Required comparison: DDPG (same wrapper, reward, seeds, eval code) and a
  random continuous controller as a floor.
- The scientific question is **not** “can we get a high score.” It is whether
  TD3’s extras (twin critics, delayed policy updates, target policy smoothing)
  change behaviour on this reaching task under a controlled protocol.

**Headline you must be willing to defend** (write it in your own words):

> Both learned agents reach the target on held-out episodes; the random
> controller does not. At 50,000 steps and three seeds, TD3 and DDPG are
> indistinguishable relative to across-seed variation.

**TODO**

- [ ] Why target reaching is a sensible DRL exam task (low-dimensional,
      continuous action, dense distance signal) — two sentences, not a
      robotics manifesto.
- [ ] What this paper will **not** claim (sim-to-real, contact, 7-DoF).
- [ ] Point to the public repo and that figures regenerate from committed CSVs.

---

## 2. Background (~450 words)

**Job.** Give the examiner the minimum needed to read §3–4. Do not write a
textbook chapter on RL.

**Must include (in your words, after you have opened the sources)**

- DDPG: deterministic actor, replay buffer, target networks, exploration
  noise. Cite the original paper after you look it up.
- TD3 as three changes to DDPG: clipped double Q, delayed actor updates,
  target policy smoothing. Cite Fujimoto et al. after you look it up.
- Why those three changes exist (overestimation, actor–critic lockstep,
  brittle peaks in Q). One mechanism sentence each.
- Gymnasium vs legacy `gym`: we use Gymnasium only; `terminated` vs
  `truncated`.
- Reacher-v5 observation is **not** raw `(qpos, qvel)`: cos/sin of joint
  angles, target xy, joint velocities, fingertip−target xy (10-D). Confirm
  from Appendix A, not from memory.

**Must not include**

- PPO, SAC, or MuJoCo locomotion unless you actually ran them.
- “TD3 is always better than DDPG.”

**TODO**

- [ ] Exact bibliographic entries (author, title, year, venue) from the PDF
      or a library database — not from this outline.
- [ ] One sentence on why a **random torque** baseline is required in
      addition to DDPG (floor vs isolating TD3’s extras).

---

## 3. Problem formulation (~800 words) — original group work

**Job.** This section is 16 marks. Write an MDP, not a code tour. The
examiner should be able to reimplement the MDP from this section alone.

Write the tuple \((S, A, P, R, \gamma)\) with the following decisions
**justified**. If you cannot justify a component, drop it or say it is
redundant.

### 3.1 State

Installed Reacher-v5 observation (verify in Appendix A), dimension 10:

| Index | Content |
|------:|---------|
| 0–1 | \(\cos q_0, \cos q_1\) |
| 2–3 | \(\sin q_0, \sin q_1\) |
| 4–5 | target \((x, y)\) |
| 6–7 | \(\dot q_0, \dot q_1\) |
| 8–9 | \((p_{\text{fingertip}} - p_{\text{target}})_{x,y}\) |

**You must answer, in prose:**

1. Why cosine/sine of joint angle instead of the raw angle.
2. Whether indices 8–9 are **Markov-redundant** given 0–7 (they are a
   function of kinematics + target). If you keep them, say what they buy
   the function approximator (scale, linearity) and what they cost
   (derived features the brief asked you to justify).
3. Whether you normalise. If not, why the default Box is acceptable.
4. Does the Markov property hold? Joint positions and velocities plus
   target location: yes for this rigid 2-link planar arm in MuJoCo, **if**
   you accept that the target is part of the observable state and that
   actuator dynamics are abstracted as instantaneous torque. If you think
   something is missing (e.g. previous action, contact, motor temperature),
   name it and say you are treating it as out of scope, not pretending it
   is Markov.

### 3.2 Action

\(A = [-1, 1]^2\): torques at the two hinges. Policy output is mapped by
SB3’s squashing to the Box (state that; do not invent a custom scaling
unless you implemented one).

### 3.3 Reward (equation, not prose)

Headline configuration (`src/config.py`):

\[
r_t = - w_{\mathrm{dist}} \, d_t - w_{\mathrm{ctrl}} \,\|a_t\|_2^2 + b \,\mathbf{1}[d_t < \varepsilon]
\]

with \(w_{\mathrm{dist}} = 1.0\), \(w_{\mathrm{ctrl}} = 0.1\), \(b = 0\),
\(\varepsilon = 0.05\) used only for metrics when \(b = 0\),
\(d_t = \|p_{\mathrm{fingertip}} - p_{\mathrm{target}}\|_2\) in the plane.

**You must answer:**

1. Relative scale: \(d_t\) on Reacher is typically \(\mathcal{O}(10^{-1})\)
   (workspace radius 0.2). \(\|a\|_2^2\) is at most 2. With \(w_{\mathrm{ctrl}}
   = 0.1\) the effort term is not negligible compared with distance. Was
   that intended? (You did **not** match the installed native default
   \(w_{\mathrm{ctrl}} = 1\). Own that.)
2. Why \(b = 0\): a dense distance term plus a sparse reach bonus can
   dominate or double-count. You chose not to mix them in the headline.
3. Success threshold \(\varepsilon = 0.05\) is an **evaluation** design
   choice, not a physical SI millimetre claim unless you justify it from
   the model (fingertip size / target size in `reacher.xml`). If you cannot
   justify 0.05 from the XML, say it is a pre-registered tolerance in
   simulator units and that you did not retune it after seeing success
   rates.

### 3.4 Termination vs truncation

- **Termination** \(T\): never, in the headline. Bootstrapping: no terminal
  value of 0 from a goal being reached. Q-targets always use the 50-step
  truncation handling of the algorithm (SB3 uses `done` carefully; **you**
  must say whether truncated steps bootstrap — look at SB3 2.9.0 behaviour
  and do not guess).
- **Truncation**: 50 steps (`TimeLimit`). This is a timeout, not success.
  Treating timeout as `terminated=True` would incorrectly zero the backup.

### 3.5 Discount

\(\gamma = 0.98\). Effective horizon \(1/(1-\gamma) = 50\) steps, equal to
the episode limit. **You** must say why that is better or worse than the
SB3 default 0.99 (horizon 100, longer than the episode). This is a real
choice; do not write “we used 0.99 because everyone does.”

**TODO**

- [ ] Write 3.1–3.5 as continuous prose with the equation in 3.3.
- [ ] One short paragraph: observation construction is the **library**
      vector; you did not add extra features beyond what Reacher-v5 returns.
      If that is false, describe the extra features.

---

## 4. Methodology (~850 words)

**Job.** An examiner should reproduce the table in §5 from this section plus
the repo.

### 4.1 Library and versions (verify, then paste)

You must state **library, version, class, and every hyperparameter that
differs from the default.** From the installed stack used in the README
table (re-check with `verify_env.py`):

| Item | Value |
|------|--------|
| Library | Stable-Baselines3 |
| Version | 2.9.0 (confirm) |
| Agent class | `TD3` |
| Baseline class | `DDPG` |
| Policy | `MlpPolicy` |
| Gymnasium | 1.3.0 (confirm) |
| MuJoCo | 3.12.0 (confirm) |
| Env ID | `Reacher-v5` |

Differ from SB3 2.9.0 TD3 defaults (already in README §8): `buffer_size`
200,000 vs 1,000,000; `learning_starts` 1,000 vs 100; `gamma` 0.98 vs 0.99;
`net_arch` [256, 256] vs [400, 300]; `NormalActionNoise(σ=0.1)` vs none.
TD3-only `policy_delay=2`, `target_policy_noise=0.2`, `target_noise_clip=0.5`
are library defaults — say **left at default** rather than “we tuned them.”

**No hyperparameter search is documented in this repo.** If that is true,
write “no search; evaluation episodes were not used for selection.” If
someone did search off-repo, you must declare seed, range, and that eval
episodes were held out — or you are in trouble.

### 4.2 Identical protocol (copy-paste checklist into prose)

TD3, DDPG, and random used **the same**: wrapper, reward, evaluation
episode construction, seeds, metric code, 30 episodes per seed,
`deterministic=True` for learned policies. Random has no deterministic
mode; say that the flag is accepted for interface compatibility.

### 4.3 Seeds and budget

- Seeds: **0, 1, 2** (report the values, not “three random seeds”).
- Steps: **50,000** per algorithm per seed (frozen headline).
- If compute was limited, say so here: fewer **steps**, not fewer seeds.
- Training exploration: Gaussian action noise \(\sigma=0.1\); disabled at
  eval.

### 4.4 Evaluation episodes

- 30 episodes per seed (rubric minimum).
- Eval seed \(=\) training seed \(+ 10{,}000\).
- Targets are whatever Reacher samples from that RNG stream; you did not
  freeze an explicit target list. **Say that honestly.** It is weaker than
  a listed held-out target set, but it is disjoint from training RNG.

### 4.5 Metrics (definitions must match `src/metrics.py`)

| Metric | Definition in code |
|--------|--------------------|
| Cumulative reward | Sum of wrapper \(r_t\) over the episode |
| Final distance | \(d_T\) at last step |
| Success | \(1\) iff \(d_T < 0.05\) (**final** position, not any-time) |
| Time to target | First step (1-based) with \(d_t < 0.05\); NaN if never; seed mean uses `nanmean` |
| Control effort | Mean of \(\|a_t\|_2^2\) over steps |

**Disclose the inconsistency:** success uses the **end** of the episode;
time-to-target uses the **first crossing**. An episode can contribute a
time-to-target and still be a failure. Do not silently “fix” this after
seeing the table.

Aggregation: per-seed mean over 30 episodes, then mean ± sample std
(`ddof=1`) over 3 seeds. Every number in §5 follows that.

### 4.6 Implementation attribution

SB3 supplies TD3, DDPG, and the replay buffer. The wrapper, reward weights,
random controller, eval harness, logging, and plots are group work (or must
be, by the time you submit). Point to `AI_Use_Declaration.md`.

**TODO**

- [ ] Wall-clock and hardware of the machine that produced `logs/train_*.csv`
      (you measure this; do not invent).
- [ ] Confirm SB3’s handling of truncated vs terminated bootstrapping with
      the installed version; one accurate sentence.

---

## 5. Results (~550 words)

**Job.** Tables and a figure, then what they show. No discussion of “why TD3
failed to beat DDPG” here — that is §6.

### Table 1 — Headline evaluation (copy from `results/aggregate.json`)

Mean ± std over **n = 3 seeds**. 30 deterministic episodes per seed.

| Method | Return | Success | Final dist. | Time to target | Control effort |
|--------|--------|---------|-------------|----------------|----------------|
| TD3 | −2.449 ± 0.211 | 0.989 ± 0.019 | 0.011 ± 0.003 | 11.37 ± 0.90 | 0.083 ± 0.006 |
| DDPG | −2.401 ± 0.093 | 1.000 ± 0.000 | 0.010 ± 0.002 | 10.69 ± 0.52 | 0.085 ± 0.004 |
| random | −12.815 ± 0.932 | 0.078 ± 0.038 | 0.169 ± 0.015 | 23.80 ± 1.73 | 0.660 ± 0.011 |

Episode length is 50.0 ± 0.0 for all methods (truncation).

Per-seed values: Appendix B (`results/per_seed_summary.csv`). **Do not**
promote TD3 seed 2 (−2.207) as the result.

### Figure 1

`figures/fig1_training_curves.png`, generated only from `logs/train_*.csv`
via `python run.py --plot`. Caption: training return vs steps, mean ± 1 std
across three seeds. Say that training return uses exploration noise, so it
is not comparable one-to-one with the deterministic eval table.

**TODO (results prose, not caption padding)**

- [ ] Learned agents vs random: large gap on return, success, distance,
      effort. The floor is established.
- [ ] TD3 vs DDPG: report the **direction** (DDPG slightly higher return and
      success) and that `|diff|` < combined across-seed std on return,
      distance, success, and effort.
- [ ] Time-to-target: both learned policies reach on the order of 11 steps
      *when a crossing occurs*; do not compare it as if it used the same
      denominator as success.
- [ ] Training curves: say whether they flatten, whether the std band shrinks,
      whether either algorithm is unstable (divergent seeds). Look at the
      figure; do not invent “stable convergence” if seed 2 is an outlier.

---

## 6. Discussion (~500 words)

**Job.** Interpret. This is where “TD3 did not beat DDPG” becomes a
diagnosis, not an apology.

**Arguments that are defensible from logs you already have**

1. **Task saturation.** Success ≈ 1 for DDPG and ≈ 0.99 for TD3. Once the
   fingertip is inside 0.05, twin critics cannot show up as a success-rate
   win. Look at **return** and **control effort** for remaining signal;
   they also do not separate.
2. **What three seeds can and cannot do.** Combined-spread check is not a
   t-test. You can say “we did not observe a difference larger than seed
   noise.” You cannot say “TD3 and DDPG are equal.”
3. **TD3’s extras need a problem where DDPG overestimates.** Reacher-v5 is
   a short-horizon, dense-reward, 2-D torque task. Overestimation may be
   mild. That is a **task** limitation, not proof that TD3 is pointless.
4. **Control-effort weight.** Learned effort ≈ 0.08 vs random ≈ 0.66. The
   penalty is shaping action magnitude relative to the floor. You cannot
   claim \(w_{\mathrm{ctrl}}\) was “optimal” without an ablation you did
   not run. If you did not run a weight sweep, do not pretend you did.

**Do not use** “maybe we under-trained” as the lead diagnosis: 50,000 steps
is 1,000 episodes of length 50, and eval success is already ceilinged. If
you mention under-training, tie it to the **training curve still moving**
(only if the figure shows that), not to the success rate.

**TODO**

- [ ] One paragraph: random controller isolates “learning happened”; DDPG
      isolates “TD3 extras.” Your interesting comparison is TD3 vs DDPG;
      it did not separate.
- [ ] Optional: qualitative failure mode (TD3 seed 0 success 0.967 vs 1.0
      for other learned cells). Do not overfit a story to one seed.

---

## 7. Limitations and deployment (~400 words)

**Do not** write “limited compute, limited time, future work on more seeds”
as the body of this section. Mention compute in one clause if you must
(`n=3` seeds). Spend the words on **robotic manipulation**.

**Push these points (in your wording)**

1. **Sim-to-real.** MuJoCo Reacher has instantaneous torque, no cable
   stretch, no joint friction identification, no latency. A policy that
   tracks a simulated fingertip will not transfer to a physical arm without
   domain randomisation, system ID, or residual learning — none of which
   you did.
2. **Torque limits and actuator dynamics.** Actions are clipped to
   \([-1,1]\) in simulator units, not measured N·m on a named motor. Real
   actuators have bandwidth, backlash, and thermal limits. A high-frequency
   torque chatter that is cheap in sim can damage a gearbox.
3. **Safety under a continuous learned policy.** No constraint layer, no
   control barrier, no emergency stop in the MDP. Evaluation is
   deterministic in sim; a real deployment needs torque/velocity
   saturation **outside** the policy and a monitored workspace. Reacher’s
   “never terminate on collision” is not a safety case.
4. **What reaching does not tell you.** No contact, no grasping, no
   partial observability from vision, no human in the workspace, no
   multi-step manipulation. Success on Reacher-v5 is not evidence for
   contact-rich tasks (assembly, wiping, peg-in-hole).

**Also disclose (short)**

- Success vs time-to-target definition mismatch.
- \(w_{\mathrm{ctrl}}=0.1\) ≠ installed native default.
- Eval targets sampled from a disjoint seed stream, not a frozen target
  catalogue.
- Three seeds: spread is reported; inference is weak.

**TODO**

- [ ] Write 7 as four short subsections with those headers, not a bullet
      dump. No generic “DL is data hungry.”

---

## 8. Conclusion and further work (~100 words)

Three sentences:

1. What you asked.
2. What you found (reach vs random; TD3 ≈ DDPG under this protocol).
3. One next experiment that would **test a mechanism**, not “try SAC”:
   e.g. longer horizon / sparser reward where overestimation should show,
   or a declared \(w_{\mathrm{ctrl}}\) ablation with the same seeds — not
   a hunt for a TD3 win on the same table.

---

## Suggested appendix list (uncounted)

- **A.** `python verify_env.py` full stdout (installed versions, obs/action,
  native reward weights, truncate at 50).
- **B.** `results/per_seed_summary.csv` as a table.
- **C.** Full hyperparameter dump (`python -m src.config`).
- **D.** Completed `AI_Use_Declaration.md`.
- **E.** How to regenerate Table 1 and Figure 1 (`python run.py --eval` /
  `--plot`) from committed artifacts.

---

## Who writes what (proposal — change names)

| Member | Report | Must be able to defend in the 20 min demo |
|--------|--------|-------------------------------------------|
| A | §3 formulation + §7 sim-to-real / actuators | Why this state, why \(\gamma\), term vs trunc |
| B | §4 protocol + §5 tables | Seeds, identical baseline path, metric definitions |
| C | §2 background + §6 discussion | TD3 extras, why they did not show up here |

Everyone reads §1 and §8. Nobody should present only “I made the slides.”

---

## After you draft

Send **one section at a time** for language editing. The assistant should
name vague claims and protocol slips, not rewrite the MDP for you.

First section to draft: **§3 Problem formulation**. That is where the marks
are, and it does not depend on polishing Background.
