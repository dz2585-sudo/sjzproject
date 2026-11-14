# Two-cell action resolution ablation

[Main README](../README.md#two-cell-action-resolution-ablation) · [Frozen report](../results/tracking_action_resolution/results_summary.md) · [Conclusions](../results/tracking_action_resolution/conclusions.md) · [Paper addendum](../results/tracking_action_resolution/paper_addendum.md) · [Behavior interpretation](../results/tracking_action_resolution/behavior_interpretation.md)

The sole intervention appends four one-tick two-cell cardinal moves. Original IDs and mechanics remain unchanged; all three planning layers use the selected action set. H=3, gamma=.99, full-state observation, known .05/.05 model, reward, guard, float64 and tie rule are fixed. Stress environment is .10/.10. Hybrid terminal is `max_original_A10 Q_shaped(o(s),a)+Phi(s)` with `Phi=-L1`. No new Q entries, formal training or Q updates exist.

| Split | Controller | Instances | D | C (%) | H / 300 ticks | A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Final | Plan-H3-A10 | 1 | 1.2120 | 74.6447 | 88.3700 | 100.0000 |
| Final | Plan-H3-A14 | 1 | 1.1164 | 75.5280 | 111.8980 | 100.0000 |
| Final | Hybrid-H3-A10 | 5 | 1.2023 ± 0.0034 | 75.0547 ± 0.0492 | 88.8128 ± 0.0227 | 100.0000 ± 0.0000 |
| Final | Hybrid-H3-A14 | 5 | 1.0745 ± 0.0004 | 76.2769 ± 0.0037 | 112.7332 ± 0.0041 | 100.0000 ± 0.0000 |
| Stress | Plan-H3-A10 | 1 | 1.3487 | 69.6750 | 79.6100 | 100.0000 |
| Stress | Plan-H3-A14 | 1 | 1.2240 | 71.7367 | 103.4400 | 100.0000 |
| Stress | Hybrid-H3-A10 | 5 | 1.3422 ± 0.0052 | 70.1420 ± 0.0494 | 79.9540 ± 0.0778 | 100.0000 ± 0.0000 |
| Stress | Hybrid-H3-A14 | 5 | 1.1794 ± 0.0009 | 72.3987 ± 0.0147 | 104.2820 ± 0.0135 | 100.0000 ± 0.0000 |

| Split | Terminal | ΔD | ΔC (pp) | ΔH | H change (%) | A14 C (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Final | Plan | -0.095613 | +0.8833 | +23.5280 | +26.6244 | 75.5280 |
| Final | Hybrid | -0.127820 | +1.2223 | +23.9204 | +26.9335 | 76.2769 |
| Stress | Plan | -0.124733 | +2.0617 | +23.8300 | +29.9334 | 71.7367 |
| Stress | Hybrid | -0.162763 | +2.2567 | +24.3280 | +30.4275 | 72.3987 |

Hybrid rows are means ± sample SD over five different source models (60–64); pure Plan has one instance and no source-model SD. H relative improvement uses **100×(mean H_A14 / mean H_A10 − 1)**. Final Hybrid is 26.9335050803%, Stress 30.4274958101%. The means of per-model ratios are respectively 26.9335110460% and 30.4275926911%; these are different statistics and are not used for the aggregate headline. [Raw recomputation and both definitions](../provenance/action_resolution_release/recalculated/raw_recomputed.json) make the distinction explicit. Per-model rows retain their own matched ratios.

Final Plan and all five Hybrid A14 instances meet C≥75% plus 98% hit retention. Stress does not meet 75%. Every paired D/C/H improves, but ΔC<5pp means both relative gates fail. Bootstrap resamples paired whole scenes and paired source models, with common scene draws across models; it never treats ticks or shared-scene model records as independent samples. Finite-horizon value dominance (56.25M checks, minimum 0, no violations at 1e-9) is separate from rollout outcomes. Regression fragments remain included.

Medium's largest benefit is speed-1 phase/alignment control; geometric crossing of the current target position increases, so no broad overshoot-reduction claim is supported. [Action denominators](../provenance/action_resolution_release/recalculated/action_denominators.csv) distinguish observable speed, actual speed, boundary and nominal versus clipped displacement. [All raw window records](../results/tracking_action_resolution/evaluation/) and [paired source-model effects](../results/tracking_action_resolution/paired_differences.csv) remain unchanged.

## Reproduction and publication audits

Install pinned `requirements-reproduction.txt` (Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2, Pillow 13.0.0 in this run). Start with the saved-record audits below; **they run no formal environment evaluation or training**:

```powershell
python -m aimrl.action_publication action --output .tmp/action-audit
python -O -m aimrl.action_publication action --output .tmp/action-audit-O
python -m aimrl.action_publication gifs --output .tmp/action-gif-audit
python -m aimrl.action_publication inputs --output .tmp/action-input-audit
python -m aimrl.action_publication tests --output .tmp/action-all-tests
python -O -m aimrl.action_publication tests-O --output .tmp/action-tests-O
```

The release adapter accepts only five named navigation/ignore changes after proving the unchanged archived originals and reviewed current hashes. It verifies the complete old history; it does not substitute expected model/core hashes. The only omitted study file is a non-scientific local Git-identity snapshot, replaced by a labelled redacted copy with the original hash/size mapping. The original is retained locally. Original frozen manifests and original reports are not rewritten. Their claims about unchanged files and unsubmitted state describe the study-completion moment, not this later publication.

Rebuild kernels with **Zig 0.15.2**, C11, `-O3 -fno-fast-math -ffp-contract=off -shared`, then run all final checks:

```powershell
python scripts/check_action_publication.py --compiler zig --output .tmp/action-release-checks
```

This builds tradeoff, boundary, planning and action-resolution kernels into the new output directory, and separately checks original DLL fingerprints and rebuilt numerical parity. Windows archived DLLs are included; on Unix rebuild as `.so` and use a separately recorded platform audit rather than changing old Windows hashes. Source: [action kernel](../aimrl/tracking_action_resolution/kernel.c). [Original reproduction document](../results/tracking_action_resolution/REPRODUCE.md) retains exact original training-free replication commands. Its pre-publication direct history assumptions now require the release adapter above; use new output paths. No fourteen-action training phase exists in this release.

The five checkpoint dependencies are repository-local files under `results/tracking_boundary_stability/runs/R-Decay/seed_60/checkpoints/step_5000000.npz` through seed_64, with Q/count/file digests in [model provenance](../results/tracking_action_resolution/model_provenance.json). A10 DP layers reference [historical planning caches](../results/tracking_planning/cache/); A14 layers are in [this study's cache](../results/tracking_action_resolution/cache/). No private absolute path or untracked file is a runtime dependency. Original recorded command/executable paths are historical provenance only, not instructions to access a private drive.

Formal counts: Final 6,000 windows, Stress 2,400, total 2.52M ticks. Development 2,400/720,000; Demo 12/3,600. Setup costs are 112.5M backups, 657M chance terms and 1.35B action-value lookups. Rebuilding validation arrays is charged separately and is not a training or evaluation tick. [Cost accounting](../results/tracking_action_resolution/execution_accounting.json) retains scopes. This publication adds only verification computation.

## Figures and complete four-panel demonstrations

[D/C/H](../results/tracking_action_resolution/figures/D_C_H.png) · [paired models](../results/tracking_action_resolution/figures/paired_models.png) · [action use](../results/tracking_action_resolution/figures/action_usage.png) · [high-speed events](../results/tracking_action_resolution/figures/high_speed_events.png)

[7400000 GIF](../results/tracking_action_resolution/demos/scenario_7400000/synchronized.gif) · [7400001 GIF](../results/tracking_action_resolution/demos/scenario_7400001/synchronized.gif) · [7400002 GIF](../results/tracking_action_resolution/demos/scenario_7400002/synchronized.gif)

Every GIF includes the initial state plus all 300 ticks, 301 frames and 20,070 ms. The publication rerenders from saved traces (zero environment steps) and compares every pixel/timing against the original GIF. **Actual-player inspection was not performed; no Chrome/desktop permission requested.**

Fourteen-action Q-learning is a future independent study. Current results establish benefits only within this full-state, known-model, fixed-terminal planning framework. [Release evidence and file/hash boundary](../provenance/action_resolution_release/README.md).
