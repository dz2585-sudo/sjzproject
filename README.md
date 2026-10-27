# 6885finalproject_dtz

# AimRL

AimRL is a small reproducible reinforcement-learning project for ELEN E6885. It investigates whether distance-based potential reward shaping accelerates tabular Q-learning for hitting a moving target, and how policies perform when the target movement probability changes. The comparison includes a rule policy, base-reward Q-learning, and potential-shaped Q-learning. No method is assumed to be superior in advance.

Day 5 is complete: the grid environment, both Q-learning methods, three-seed experiments, independent final testing, summaries, and automatic demonstrations are implemented. No GPU, CUDA, PyTorch, or TensorFlow is required. A completed report is not yet available.

## Installation and Setup Check

Use a project-specific virtual environment. The existing environment was verified on Windows 11 with Python 3.13.9, NumPy 2.5.4, and Matplotlib 3.11.2. The two dependency versions are pinned in [requirements.txt](requirements.txt). Other operating systems and Python versions have not been verified.

Run the following commands from the repository root when creating a new environment. An existing working environment can be reused.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

macOS / Linux:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/check_setup.py
./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

Activation and PowerShell execution-policy changes are unnecessary. The setup script reports the operating system, interpreter, imported dependency versions, configuration checks, and resource budget. It returns a nonzero exit status on failure and does not open a window or train a model.

## Environment and Policies

`AimEnv(config).reset(seed=42)` returns `(observation, info)`. `step(action)` returns `(observation, reward, terminated, truncated, info)`. The observation is the immutable tuple `(xc, yc, xt, yt, v)`. Actions 0–5 are left, right, up, down, stay, and fire. Each step executes the crosshair action and checks firing before updating the target. A hit terminates immediately; otherwise, target turning and movement are sampled independently, in that order.

Base rewards are −0.05 for an ordinary step, −1.05 for a miss, and +9.95 for a hit. Reaching the 50-step limit without a hit truncates the episode. A hit on the last allowed step takes termination priority. Call `reset()` explicitly after either ending condition. See [spec.md](spec.md) for the complete rules.

The rule policy fires when aligned; otherwise it corrects the horizontal coordinate before the vertical coordinate. The random policy samples uniformly from the six actions using an independent generator. Neither baseline is trained.

## Demonstrations

Run these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m aimrl.demo --policy rule --episodes 3 --seed 42 --interval-ms 250
.\.venv\Scripts\python.exe -m aimrl.demo --policy random --episodes 3 --seed 42 --headless
.\.venv\Scripts\python.exe -m aimrl.demo --policy qlearning --model outputs/day5/runs/potential_seed0/q_table.npy --episodes 3 --seed 42 --interval-ms 250
.\.venv\Scripts\python.exe -m aimrl.demo --policy qlearning --model outputs/day5/runs/base_seed0/q_table.npy --episodes 3 --seed 42 --headless --save-frame .tmp/base_preview.png
```

The demonstration runs automatically. Close the window after completion to exit. Headless mode creates no GUI and does not wait between steps. Optional `--csv` records episode diagnostics; `--save-frame` saves the final episode state. Scenario seeds increase from the specified seed, and random actions use an independent reproducible stream.

Model loading requires `run_config.json` and `metadata.json` alongside the final Q table. The grid size, action mapping, table shape, dtype, and finite values are checked. The saved environment configuration is used for the demonstration, which is greedy and does not train or apply shaping again. The display identifies Base or Potential Q-learning and the number of training steps. Headless previews have been verified; the interactive GUI has not been verified.

Before importing Matplotlib, plotting and demonstration processes respect an existing writable `MPLCONFIGDIR`; otherwise they use the project `.mplconfig` directory or a temporary directory. This resolved the earlier user-directory font-cache warning without changing system settings.

## Learning and Reward Shaping

`QLearningAgent.state_index` maps direction −1 to index 0 and +1 to index 1. `select_action` uses epsilon-greedy exploration and random tie-breaking during training. `greedy_action` uses a fixed action-order tie-break during evaluation. `update(state, action, reward, next_state, terminated, truncated)` suppresses bootstrap only for true termination; external truncation bootstraps from the actual next state.

Epsilon decreases from 1.00 to 0.05 over the first 80,000 global interaction steps and remains at 0.05 thereafter. Environment and exploration streams are derived independently using `SeedSequence(seed).spawn(2)`. Validation uses an independent environment, fixed scenarios, and a frozen Q table.

The environment always returns the base reward. Potential mode applies exactly one transformation immediately before the Q update:

```text
Phi(s) = -(|xc-xt| + |yc-yt|)
learning_reward = base_reward + scale * (gamma * Phi(next_state) - Phi(state))
```

Only true termination sets the next potential to zero. External truncation retains the actual next potential and bootstraps. Both methods begin with zero float32 Q tables. `base_return`, `learning_return`, and `shaping_return` in training logs are undiscounted episode sums; learning returns are not compared across reward modes as a performance measure.

## Formal Experiments and Reproduction

The [frozen Day 4 protocol](outputs/day4/protocol_frozen.json) specifies base/potential × training seeds 0/1/2, each trained from zero for exactly 100,000 interactions. Validation uses seeds 100000–100199 at steps 0, 10000, …, 100000. Final testing fixes the 100,000-step models and uses independent scenario seeds 200000–200499 at target movement probabilities 0.10, 0.25, and 0.50, with turning probability 0.10.

The archived run contains 600,000 training interactions, 13,200 validation episodes, 9,000 learned-policy test episodes, and 3,000 baseline test episodes. Test metrics include all episodes: success rate, mean steps (timeouts count as 50), mean misses, and mean undiscounted base return. Learned-policy error terms are sample standard deviations across three training seeds, not confidence intervals. Baselines have no training-seed standard deviation.

See the [experiment record](outputs/day5/summary/results_summary.md), [per-seed final results](outputs/day5/test/learned_per_seed.csv), [paired differences](outputs/day5/summary/paired_differences.csv), [learning curves](outputs/day5/summary/learning_curves.png), and [final-test comparison](outputs/day5/summary/test_comparison.png). The [run manifest](outputs/day5/protocol_used.json) preserves model hashes, the pre-test settings, and actual test timestamps. Final tests were not used to tune parameters or select checkpoints. Training seed 0 was used during development; new test seeds do not guarantee previously unseen states.

The historical results were produced before the English translation. Original frozen hashes and run metadata remain unchanged. The [translation provenance record](translation/README.md) and [manifest](translation/manifest.json) distinguish the recorded experimental source from later language and provenance-validation changes. The entry point accepts only exact source hashes documented in that record. Historical JSON files retain their original absolute paths as evidence; use a new output directory to reproduce the workflow on another computer.

Run these stages in order from the repository root. Training and testing reject existing raw outputs; summarization reads saved results and may regenerate derived summaries and figures.

```powershell
.\.venv\Scripts\python.exe -m aimrl.experiments --protocol outputs/day4/protocol_frozen.json --output outputs/day5_new --stage train
.\.venv\Scripts\python.exe -m aimrl.experiments --protocol outputs/day4/protocol_frozen.json --output outputs/day5_new --stage test
.\.venv\Scripts\python.exe -m aimrl.experiments --protocol outputs/day4/protocol_frozen.json --output outputs/day5_new --stage summarize
```

Do not use final-test results to adjust models, hyperparameters, training budgets, or checkpoint selection. The three training seeds support descriptive conclusions within this small fully observable environment, not claims of statistical significance or real-game generalization.

## Earlier Development Records

Day 3 stationary-target diagnostic (seed 42, 10,000 steps, epsilon decay over 8,000 steps): success increased from 0% to 13.5%, with final mean steps 43.72 and mean misses 1.96. No additional stationary-target training was performed. A [read-only replay note](outputs/day4/static_diagnostic_note.md) records five failed scenarios and the limits of that inspection.

Day 3 moving-target development run (seed 0, 100,000 steps): validation success increased from 0% to 100%. The base model and all checkpoint metrics were reproduced on Day 4. In the Day 4 single-seed comparison, base/potential success at 10,000 steps was 70%/100%, with mean steps 26.605/6.990. At 100,000 steps both achieved 100%, with mean steps 5.055/4.670. The first of two consecutive ≥90% checkpoints was 20,000/10,000 steps at 10,000-step resolution. These are development observations, not independent multi-seed findings.

The Day 3 base run saved 11 checkpoints and 2,200 validation-episode records. Its complete training episodes covered 99,995 interactions; the remaining five steps of an unfinished episode are recorded in metadata rather than represented as an artificial environment termination.

The repository includes the Day 4 protocol, regression models, and comparison records needed by the formal reproduction entry point. Earlier Day 1–3 development outputs remain in the original local project and are not part of this upload.

## File Responsibilities

- `spec.md`: environment, algorithms, protocol, metrics, limitations, and references.
- `configs/base.json`: central parameters; `requirements.txt`: pinned dependencies.
- `scripts/check_setup.py`: configuration and execution-environment checks.
- `aimrl/config.py`, `aimrl/env.py`: configuration loading and environment transitions.
- `aimrl/policies.py`, `aimrl/agent.py`: baselines and tabular Q-learning.
- `aimrl/rewards.py`, `aimrl/train.py`, `aimrl/evaluate.py`: reward transformation, training, and independent evaluation.
- `aimrl/experiments.py`, `aimrl/compare.py`: staged formal experiments and development comparisons.
- `aimrl/render.py`, `aimrl/demo.py`, `aimrl/plots.py`, `aimrl/mpl_setup.py`: visualization, demonstrations, and a writable cache.
- `aimrl/provenance.py`, `translation/`: validation and documentation of post-experiment translation changes.
- `tests/`: environment, learning, shaping, experiment, and provenance checks.
- `outputs/day5/`: final models, configurations, raw records, summaries, and figures.
- `.gitignore`: excludes environments, caches, credentials, and temporary files; `.gitattributes`: preserves exact bytes for recorded hashes.
