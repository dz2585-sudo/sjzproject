# AimRL: Grid Aiming and Continuous Tracking

AimRL is a reproducible tabular reinforcement-learning project for ELEN E6885. It compares base Q-learning, potential-based reward shaping, and transparent heuristic/random baselines in fully observed grid simulations.

## Tracking Representation Study (5M per Run)

Four methods (Absolute Base, Absolute Potential, Relative Base, Relative Potential), three training seeds (10, 11, 12), and 5,000,000 interactions per run: **12 fresh runs and 60,000,000 formal interactions**. The four disposable 100,000-interaction pilots are separate from that budget; their models never initialized formal runs.

![Tracking representation study: archived 20M Potential, new 5M Relative Base and Relative Potential, and predictive rule](results/tracking_improvement/render/seed3601000_main_comparison.gif)

**Full 301-frame GIFs (20.07 seconds each):** [Scenario 3601000](results/tracking_improvement/render/seed3601000_main_comparison.gif) · [Scenario 3601001](results/tracking_improvement/render/seed3601001_main_comparison.gif) · [Scenario 3601002](results/tracking_improvement/render/seed3601002_main_comparison.gif) · [Six-policy comparison including Random](results/tracking_improvement/render/seed3601000_six_policy_comparison.gif). New displayed models always use the predeclared training seed 10. Random is an untrained negative control; failures are retained.

Independent final test: 500 shared 300-tick scenarios per policy instance. Values are **mean ± sample SD across the three training seeds**; accuracy is pooled within each seed before averaging.

| Method | Mean L1 distance (cells) | Close fraction (%) | Far fraction (%) | Hits/window | Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Absolute Base | 16.469 ± 0.105 | 0.848 ± 0.037 | 91.502 ± 0.384 | 0.177 ± 0.012 | 1.200 ± 0.132 |
| Absolute Potential | 7.531 ± 0.064 | 25.808 ± 0.351 | 52.509 ± 0.618 | 21.873 ± 0.235 | 74.913 ± 0.447 |
| Relative Base | 1.502 ± 0.034 | 61.655 ± 4.022 | 2.521 ± 0.269 | 84.442 ± 0.130 | 99.396 ± 0.431 |
| Relative Potential | 1.440 ± 0.022 | 60.540 ± 1.951 | 2.338 ± 0.147 | 84.489 ± 0.116 | 99.493 ± 0.304 |

Relative representation substantially improved both reward modes over their matched absolute controls. **Relative Potential did not outperform Relative Base on every metric:** it had lower mean distance and far fraction, but lower close fraction (60.54% versus 61.65%); their hit rates were similar. The predeclared practical Relative-Potential-versus-Absolute-Potential target was achieved. The aspirational 75% close-tracking target was not; the distance, far-fraction, and acquisition targets were achieved. Three training seeds support descriptive comparisons, not a significance claim.

Relative offsets discard absolute wall information: states sharing one encoded observation can differ in clipping and reflection. This is approximate observation aggregation, not an exact Markov state. No rule fallback was added. The independent stress split doubles only turn/speed-toggle probabilities; it does not establish arbitrary real-world generalization.

These new **5M-interaction** controls use a 4M epsilon-decay schedule. The archived **20M-interaction, seed-0** models below use a different budget and schedule and are labeled cross-budget references. The main GIFs intentionally include archived Potential for visual continuity; the equal-budget four-way numerical comparison is the controlled evidence. The predictive rule has better final-test proximity (1.371 cells, 73.04% close), while the relative learned methods obtain more hits.

[Complete report and all per-seed values](results/tracking_improvement/results_summary.md) · [Cross-seed comparisons](results/tracking_improvement/evaluation/cross_seed_summary.json) · [Frozen protocol](results/tracking_improvement/protocol.json) · [Original execution audit](results/tracking_improvement/execution_checks.json) · [Figures](results/tracking_improvement/render/) · [Models and genuine visitation counters](results/tracking_improvement/runs/) · [Publication provenance and verification](provenance/tracking_improvement_publication.md).

All twelve final models and counters retain their lossless NPZ formats. Setup and tests are rerun for publication; the recorded research audit remains unchanged. No training or result regeneration is part of publication.

## Historical Continuous-Tracking Demonstrations (20M, Seed 0)

![Continuous tracking: base, potential, predictive rule, and random policies](results/continuous_tracking/render/seed3601000_comparison.gif)

**Full animated demonstrations:** [Scenario 3601000](results/continuous_tracking/render/seed3601000_comparison.gif) · [Scenario 3601001](results/continuous_tracking/render/seed3601001_comparison.gif) · [Scenario 3601002](results/continuous_tracking/render/seed3601002_comparison.gif).

Each GIF contains 301 frames: the initial state and all 300 simulation ticks, approximately 20 seconds at 15 FPS. The target continues after hits, moves on both axes, changes speed, and reflects at boundaries. The four panels use identical target paths. The three scenarios were selected from target-motion coverage before training, not from learned-policy performance.

## Historical Continuous-Tracking Results (20M, Seed 0)

Both learned policies started from zero and trained for exactly 20,000,000 interactions with seed 0. On 200 shared development scenarios:

| Policy | Mean distance | Close tracking (distance ≤1) | Hits per 300 ticks | Pooled shot accuracy | Mean base return |
|---|---:|---:|---:|---:|---:|
| Base Q-learning | 16.254 | 2.623% | 2.640 | 6.939% | −24.005 |
| Potential Q-learning | 1.702 | 67.662% | 70.535 | 90.610% | 683.040 |
| Predictive rule | 1.382 | 72.512% | 44.525 | 100.000% | 430.250 |
| Random | 17.161 | 0.633% | 0.050 | 0.170% | −43.880 |

Potential shaping improved the learned policy within this single-seed experiment. The predictive rule maintained closer proximity and fired more accurately; potential fired more often and obtained more hits. The rule is a velocity-aware heuristic, not a proven optimum. These descriptive results do not establish statistical significance or real-game generalization.

See the [complete tracking report](results/continuous_tracking/results_summary.md), [comparison CSV](results/continuous_tracking/evaluation/comparison.csv), [execution checks](results/continuous_tracking/execution_checks.json), [learning curves](results/continuous_tracking/render/learning_curves.png), [speed choices](results/continuous_tracking/render/speed_choices.png), and [time trace](results/continuous_tracking/render/time_trace.png).

## Installation and Checks

Verified environment: Windows 11, Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2, and Pillow 13.0.0. No GPU or deep-learning framework is required. Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

On macOS/Linux, use `python3 -m venv .venv` and `./.venv/bin/python` for the remaining commands. Those platforms have not been verified. Matplotlib uses a writable project-local cache when necessary. Interactive GUI behavior remains unverified; the archived GIFs and PNGs were exported and inspected without a GUI.

## Run a Saved Model Without Training

The two continuous-tracking models are losslessly packaged as NPZ files because each raw NPY exceeds GitHub's ordinary file-size limit. Each package contains the exact original `q_table.npy` bytes. The loader accepts the NPZ path, and historical NPY references resolve to the corresponding package. Historical model hashes are checked against the uncompressed member.

```powershell
.\.venv\Scripts\python.exe -m aimrl.tracking25.demo --model results/continuous_tracking/potential/q_table.npz --ticks 30 --seed 3601000
```

This bounded demonstration loads the saved model and configuration, records 30 greedy ticks to `.tmp/saved_tracking_demo.json`, and performs no learning updates. [Base model package](results/continuous_tracking/base/q_table.npz) · [Potential model package](results/continuous_tracking/potential/q_table.npz).

## Experiment Archives

- [Environment baselines](results/environment_baselines/): initial rule/random diagnostics and preview.
- [Q-learning development](results/q_learning_development/): stationary-target diagnostic and moving-target development run.
- [Reward-shaping development](results/reward_shaping_development/): equal-budget base/potential comparison and frozen benchmark protocol.
- [Grid5 benchmark](results/grid5_benchmark/): six formal runs, independent final testing, cross-seed summaries and figures.
- [Grid25 development](results/grid25_development/): two five-million-interaction hit-once runs and their demonstrations.
- [Continuous tracking](results/continuous_tracking/): two twenty-million-interaction continuing-tracking runs, raw records, figures and full GIFs.

Earlier hit-once environments terminate on a hit. Continuous tracking never terminates on a hit and samples fixed 300-tick windows. Their scores answer different questions. The Grid25 trial also changes gamma and the episode cap relative to Grid5, so historical differences cannot be attributed to grid size alone.

Original numerical records, trained values, and frozen protocol hashes are preserved. Historical paths inside immutable JSON/CSV evidence remain as recorded. [Archive provenance](provenance/README.md) documents the current path mapping, exact source compatibility checks, retired evidence verification through preserved Git history, and lossless model packaging. Current explanatory paths and commands use the descriptive result directories.

## Reproduction

Saved results can be inspected without training. For a new continuous-tracking run, use a fresh output directory consistently:

```powershell
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --output results/continuous_tracking_new --stage check
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --output results/continuous_tracking_new --stage train
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --output results/continuous_tracking_new --stage evaluate
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --output results/continuous_tracking_new --stage render
```

These commands are reproduction instructions; repository cleanup does not rerun training. Completed runs are protected from overwriting. Interrupted training is not resumable without a complete saved RNG/training state. Details are in [the tracking specification](docs/tracking25_v2.md), [Grid25 instructions](docs/trial25.md), and [the project specification](spec.md).

For a fresh Grid5 benchmark, run `aimrl.experiments` with `--protocol results/reward_shaping_development/protocol_frozen.json`, a new `--output results/grid5_benchmark_new`, and stages `train`, `test`, then `summarize`. Final-test results must not be used for tuning or checkpoint selection. Training seed 0 was used during development; independent scenario seeds do not guarantee unseen initial states.

Potential-based reward shaping follows Ng, Harada and Russell, *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping* (1999). Environment semantics, references, limitations and the planned ICML-format course report are described in the specification. No completed report is currently available.
