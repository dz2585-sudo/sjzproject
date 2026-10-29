# Artifact Locations, Demonstrations, and Path Migration

Every current path below is relative to the repository root. A local folder may have a different name; keep the internal relative layout intact. The [README](../README.md) displays the main numerical results, and the [research history](project_history.md) explains the design decisions.

## What moved, and why

The publication replaced chronological folder labels with names that describe an experiment. The archive index maps old references to those published paths. The current documentation revision adds navigation; it does not move these results or GIFs again.

| Historical reference | Canonical published location | Meaning |
|---|---|---|
| `outputs/day2_rule_smoke.csv` | [results/environment_baselines/rule_policy_diagnostic.csv](../results/environment_baselines/rule_policy_diagnostic.csv) | Initial rule-policy diagnostic |
| `outputs/day2_random_smoke.csv` | [results/environment_baselines/random_policy_diagnostic.csv](../results/environment_baselines/random_policy_diagnostic.csv) | Initial Random diagnostic |
| `outputs/day2_rule_preview.png` | [results/environment_baselines/rule_policy_preview.png](../results/environment_baselines/rule_policy_preview.png) | Initial environment preview |
| `outputs/day3/` | [results/q_learning_development/](../results/q_learning_development/) | First learned models and validation |
| `outputs/day4/` | [results/reward_shaping_development/](../results/reward_shaping_development/) | Shaping comparison and frozen benchmark protocol |
| `outputs/day5/` | [results/grid5_benchmark/](../results/grid5_benchmark/) | Three-seed small-grid benchmark and final test |
| `outputs/trial25/` | [results/grid25_development/](../results/grid25_development/) | Larger hit-once development trial |
| `outputs/tracking25_v2/` | [results/continuous_tracking/](../results/continuous_tracking/) | Original continuing-tracking study |
| New study, no preceding archive name | [results/tracking_improvement/](../results/tracking_improvement/) | Absolute/Relative representation comparison |

These older names are historical lookup keys, not active folders. Frozen JSON/CSV files may still contain original local paths. The [archive manifest](../provenance/archive_manifest.json) and provenance-aware loader retain those references without rewriting measured evidence. Explanatory links and commands should use the canonical paths above.

The retired unrelated explanatory directory is absent from the current tree. A few exact historical blobs are still needed to validate older manifests; full Git history supplies those bytes. That does not recreate the retired folder as public project content.

## Current representation-study results

| Purpose | Exact location |
|---|---|
| Human-readable findings, all seeds, and limitations | [results/tracking_improvement/results_summary.md](../results/tracking_improvement/results_summary.md) |
| Cross-seed means, sample SDs, paired differences, and targets | [results/tracking_improvement/evaluation/cross_seed_summary.json](../results/tracking_improvement/evaluation/cross_seed_summary.json) |
| Development checkpoint summaries and window records | [results/tracking_improvement/evaluation/development/](../results/tracking_improvement/evaluation/development/) |
| Independent final-test summaries and window records | [results/tracking_improvement/evaluation/final/](../results/tracking_improvement/evaluation/final/) |
| Independent stress-test summaries and window records | [results/tracking_improvement/evaluation/stress/](../results/tracking_improvement/evaluation/stress/) |
| Archived-policy failure diagnostics | [results/tracking_improvement/diagnostics/](../results/tracking_improvement/diagnostics/) |
| Frozen experimental protocol | [results/tracking_improvement/protocol.json](../results/tracking_improvement/protocol.json) |
| Original execution audit | [results/tracking_improvement/execution_checks.json](../results/tracking_improvement/execution_checks.json) |
| Published-checkout verification | [provenance/tracking_improvement_publication_checks.json](../provenance/tracking_improvement_publication_checks.json) |

Each final/stress policy has a named summary and window-record file in its split directory. The cross-seed summary aggregates seed-level statistics; do not treat each correlated tick as an independent training replicate. Reports explain which values are fractions and which displayed tables convert them to percentages.

## Figures and their numerical sources

| Figure | Source records |
|---|---|
| [Learning curves](../results/tracking_improvement/render/learning_curves.png) | [Development checkpoints](../results/tracking_improvement/evaluation/development/) |
| [Final comparison](../results/tracking_improvement/render/final_comparison.png) | [Final split](../results/tracking_improvement/evaluation/final/) and [cross-seed statistics](../results/tracking_improvement/evaluation/cross_seed_summary.json) |
| [Distance tails](../results/tracking_improvement/render/distance_tails.png) | [Final summaries and distance histograms](../results/tracking_improvement/evaluation/final/) |
| [Boundary/speed diagnostics](../results/tracking_improvement/render/boundary_speed_diagnostics.png) | [Final per-policy slice summaries](../results/tracking_improvement/evaluation/final/) |
| [Coverage](../results/tracking_improvement/render/coverage.png) | [Run metadata, validation, and genuine counters](../results/tracking_improvement/runs/) |
| [Speed choices](../results/tracking_improvement/render/speed_choices.png) | [Final action-frequency summaries](../results/tracking_improvement/evaluation/final/) |

## Exact GIF paths and what each panel means

The current main comparisons show archived Absolute Potential (20M, training seed 0), Relative Base (5M, seed 10), Relative Potential (5M, seed 10), and the predictive rule. Their purpose is visual continuity and practical behavior. The six-policy GIF shows all four new 5M learned methods, the rule, and Random; it is the matched-budget visual comparison. Displayed learned seeds were fixed before evaluation, not selected for a favorable clip.

| Current animation path | Trace path |
|---|---|
| [results/tracking_improvement/render/seed3601000_main_comparison.gif](../results/tracking_improvement/render/seed3601000_main_comparison.gif) | [seed3601000_main_comparison_traces.csv](../results/tracking_improvement/render/seed3601000_main_comparison_traces.csv) |
| [results/tracking_improvement/render/seed3601001_main_comparison.gif](../results/tracking_improvement/render/seed3601001_main_comparison.gif) | [seed3601001_main_comparison_traces.csv](../results/tracking_improvement/render/seed3601001_main_comparison_traces.csv) |
| [results/tracking_improvement/render/seed3601002_main_comparison.gif](../results/tracking_improvement/render/seed3601002_main_comparison.gif) | [seed3601002_main_comparison_traces.csv](../results/tracking_improvement/render/seed3601002_main_comparison_traces.csv) |
| [results/tracking_improvement/render/seed3601000_six_policy_comparison.gif](../results/tracking_improvement/render/seed3601000_six_policy_comparison.gif) | [seed3601000_six_policy_comparison_traces.csv](../results/tracking_improvement/render/seed3601000_six_policy_comparison_traces.csv) |

All four have 301 frames and 20.07 seconds of playback. Corresponding `_preview.png` files are static previews; `_decoded_000.png`, `_decoded_150.png`, and `_decoded_300.png` are saved inspection frames. They do not replace the actual GIF.

Historical animations remain at:

- [results/continuous_tracking/render/seed3601000_comparison.gif](../results/continuous_tracking/render/seed3601000_comparison.gif)
- [results/continuous_tracking/render/seed3601001_comparison.gif](../results/continuous_tracking/render/seed3601001_comparison.gif)
- [results/continuous_tracking/render/seed3601002_comparison.gif](../results/continuous_tracking/render/seed3601002_comparison.gif)
- [results/grid25_development/demo/demo_comparison.gif](../results/grid25_development/demo/demo_comparison.gif)

The first three are original absolute-state continuous-tracking comparisons. The last is the earlier hit-once trial, where successful panels correctly hold their terminal state. The retained first layout attempt in `results/grid25_development/demo_export_attempt1/` is historical evidence, not the featured corrected GIF.

Open the current main GIF from a local repository root in PowerShell:

```powershell
Start-Process ".\results\tracking_improvement\render\seed3601000_main_comparison.gif"
```

Open the six-policy comparison:

```powershell
Start-Process ".\results\tracking_improvement\render\seed3601000_six_policy_comparison.gif"
```

If a website preview shows a still image, download the actual `.gif` file and open it locally. Recorded file decoding verifies frames and timing; desktop GUI playback is a separate check.

## Models, counters, and the NPY-to-NPZ change

New formal runs follow `results/tracking_improvement/runs/<method>/seed_<seed>/`, with methods `absolute_base`, `absolute_potential`, `relative_base`, and `relative_potential`, and seeds 10, 11, and 12. The angle-bracket components are placeholders, not literal directories.

Each run contains a final `q_table.npz`, `visitation.npz`, `metadata.json`, `run_config.json`, `training_windows.csv`, and `validation.json`. For a concrete example, see the [Relative Potential seed-10 model](../results/tracking_improvement/runs/relative_potential/seed_10/q_table.npz), [counters](../results/tracking_improvement/runs/relative_potential/seed_10/visitation.npz), and [metadata](../results/tracking_improvement/runs/relative_potential/seed_10/metadata.json). All twelve formal model/counter pairs are published.

The historical continuous-tracking models are [base/q_table.npz](../results/continuous_tracking/base/q_table.npz) and [potential/q_table.npz](../results/continuous_tracking/potential/q_table.npz). Their packages store the exact original NPY member without reducing precision. Thus an old `outputs/tracking25_v2/potential/q_table.npy` reference maps to `results/continuous_tracking/potential/q_table.npz`. The archive manifest records the original member and packaged-file integrity separately.

This historical packaging differs from the new study's explicit representation/indexing metadata. Use the model loader appropriate to the experiment. Do not load a relative table as an absolute one or assume every NPZ archive has the same schema.

## Evidence and verification boundaries

The original experiment audits describe their execution time and workspace. Later publication checks describe the published checkout. Documentation changes have their own [navigation audit](../provenance/documentation_organization_checks.json). These records have different scopes and should not be substituted for one another.

Frozen protocols, recorded numerical data, model values, and original training-source hashes remain evidence. The exact README-only compatibility mapping documents a presentation update; executable training-source acceptance is unchanged. Eight disposable pilot arrays were intentionally excluded from publication and retained locally. See the [publication explanation](../provenance/tracking_improvement_publication.md) for the full distinction.
