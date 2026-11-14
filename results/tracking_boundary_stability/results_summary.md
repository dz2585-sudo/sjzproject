# Boundary information and update stability

This is the prespecified 2×2 study using newly trained R-Fixed as the paired control. All four conditions retain close_0p5 reward, existing target dynamics, ten actions and the published exploration schedule. Relative state already included heading and speed. Boundary grouping is the only representation change; it remains an approximate aggregation.

## Execution

Completed 20 independent training runs, seeds 60–64 paired across four conditions, exactly 100,000,000 real interactions and updates. Each model starts from zero and uses the 5M checkpoint for every main result. Formal execution wall time: 235.26 seconds; sum of per-run times: 234.99 seconds. Pilot: 4,000,000 separate interactions, seed 1060, with no Final access and no model selection. Peak formal-process working set: 262.46 MiB (process-lifetime high-water mark, not a stage delta). See resources.csv for all runs and actual per-update learning-rate histograms in each metadata.json.

R has 19,208 rows and 192,080 state-action pairs; B has 1,089,288 rows and 10,892,880 pairs (56.71× as many rows). Each B Q/count array is 43,571,520 bytes; combined 83.11 MiB. Each representation uses a 2,500-byte physical-axis index map. Valid axis keys were checked and all 3,125,000 physical states were range-checked. The compact axis product contains only valid physical feature keys.

## Held-out guarded performance

Values are means ± sample SD across five independently trained seeds. D and C use post-motion distance; H is hits per 300 ticks; A is actual executed-shot accuracy, averaged over defined seed accuracies. Zero-shot accuracy is NA. Acquisition is the first post-motion distance≤1 tick, or 300 if never acquired. Far streak means consecutive post-motion distance>5 ticks.

| Split | Condition | D | C | H | A | First close | Longest far |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Final | R-Fixed | 1.377 ± 0.013 | 68.90 ± 1.56% | 84.46 ± 0.09 | 100.00% | 8.50 | 5.53 |
| Final | R-Decay | 1.343 ± 0.010 | 73.19 ± 0.04% | 84.73 ± 0.10 | 100.00% | 8.55 | 5.59 |
| Final | B-Fixed | 2.031 ± 0.077 | 56.38 ± 1.82% | 60.84 ± 2.54 | 100.00% | 14.37 | 11.16 |
| Final | B-Decay | 1.920 ± 0.017 | 57.83 ± 0.48% | 69.79 ± 0.44 | 100.00% | 13.95 | 10.96 |
| Stress | R-Fixed | 1.568 ± 0.012 | 62.45 ± 1.25% | 73.90 ± 0.27 | 100.00% | 8.70 | 5.47 |
| Stress | R-Decay | 1.544 ± 0.020 | 66.35 ± 0.20% | 74.64 ± 0.13 | 100.00% | 8.75 | 5.54 |
| Stress | B-Fixed | 2.267 ± 0.061 | 50.28 ± 1.88% | 52.28 ± 2.18 | 100.00% | 14.22 | 11.53 |
| Stress | B-Decay | 2.181 ± 0.063 | 51.79 ± 0.35% | 59.61 ± 0.36 | 100.00% | 13.65 | 11.76 |

## Prespecified gates

Practical: D does not increase, C gains at least 5 percentage points, H retains at least 98%, A≥98%, relative to the same-split paired R-Fixed. Simultaneous substitutes H≥102%. Mean and seed gates are distinct; absolute C≥75% is separate. Counts are out of five seeds. R-Fixed cannot pass its own +5pp comparative gate by construction.

| Split | Condition | Practical mean | Seeds | Simultaneous mean | Seeds | C≥75% mean | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | R-Fixed | False | 0 | False | 0 | False | 0 |
| final | R-Decay | False | 1 | False | 0 | False | 0 |
| final | B-Fixed | False | 0 | False | 0 | False | 0 |
| final | B-Decay | False | 0 | False | 0 | False | 0 |
| stress | R-Fixed | False | 0 | False | 0 | False | 0 |
| stress | R-Decay | False | 1 | False | 0 | False | 0 |
| stress | B-Fixed | False | 0 | False | 0 | False | 0 |
| stress | B-Decay | False | 0 | False | 0 | False | 0 |

## Factor effects and seed consistency

Effects below are signed raw differences; negative D and positive C/H are improvements. SD is across paired seed differences. Interaction = (B-Decay−B-Fixed)−(R-Decay−R-Fixed). No tick or scenario is treated as an independent training replicate. No confidence intervals or significance claims are made with five training seeds.

| Split | Effect | Metric | Mean | Sample SD | Paired values, seeds 60–64 |
| --- | --- | --- | --- | --- | --- |
| final | boundary_at_fixed | D | 0.65373 | 0.08340 | [0.67252, 0.5346666666666666, 0.6057533333333334, 0.7286933333333334, 0.7270133333333333] |
| final | boundary_at_fixed | C | -0.12515 | 0.02424 | [-0.15254, -0.10138666666666662, -0.09793333333333332, -0.1397533333333334, -0.13414000000000004] |
| final | boundary_at_fixed | H | -23.61640 | 2.61286 | [-26.080000000000005, -20.548000000000002, -21.053999999999995, -25.547999999999995, -24.851999999999997] |
| final | boundary_at_decay | D | 0.57669 | 0.02019 | [0.5996133333333333, 0.5977733333333333, 0.5652133333333333, 0.5599466666666668, 0.5609266666666668] |
| final | boundary_at_decay | C | -0.15360 | 0.00503 | [-0.16082666666666667, -0.15603333333333325, -0.15305333333333326, -0.14990666666666663, -0.1482] |
| final | boundary_at_decay | H | -14.93760 | 0.39640 | [-14.242000000000004, -15.066000000000003, -15.22399999999999, -15.019999999999996, -15.135999999999996] |
| final | decay_in_relative | D | -0.03409 | 0.01782 | [-0.01865333333333319, -0.052159999999999984, -0.052146666666666786, -0.014660000000000117, -0.0328533333333334] |
| final | decay_in_relative | C | 0.04287 | 0.01555 | [0.035753333333333304, 0.03695999999999999, 0.07059333333333329, 0.03398666666666661, 0.03703333333333325] |
| final | decay_in_relative | H | 0.27440 | 0.11760 | [0.3119999999999976, 0.23799999999999955, 0.3960000000000008, 0.3359999999999985, 0.09000000000000341] |
| final | decay_in_boundary | D | -0.11113 | 0.08450 | [-0.09155999999999986, 0.01094666666666666, -0.0926866666666668, -0.18340666666666672, -0.1989399999999999] |
| final | decay_in_boundary | C | 0.01441 | 0.01847 | [0.02746666666666664, -0.01768666666666663, 0.015473333333333339, 0.023833333333333373, 0.02297333333333329] |
| final | decay_in_boundary | H | 8.95320 | 2.84996 | [12.149999999999999, 5.719999999999999, 6.226000000000006, 10.863999999999997, 9.806000000000004] |
| final | interaction | D | -0.07703 | 0.09660 | [-0.07290666666666668, 0.06310666666666664, -0.04054000000000002, -0.1687466666666666, -0.1660866666666665] |
| final | interaction | C | -0.02845 | 0.02422 | [-0.008286666666666664, -0.05464666666666662, -0.05511999999999995, -0.010153333333333237, -0.014059999999999961] |
| final | interaction | H | 8.67880 | 2.86405 | [11.838000000000001, 5.481999999999999, 5.830000000000005, 10.527999999999999, 9.716000000000001] |
| stress | boundary_at_fixed | D | 0.69867 | 0.06978 | [0.67825, 0.5922500000000002, 0.7015833333333334, 0.76535, 0.7559333333333333] |
| stress | boundary_at_fixed | C | -0.12168 | 0.02169 | [-0.14435000000000003, -0.09366666666666668, -0.10536666666666672, -0.13896666666666668, -0.12603333333333333] |
| stress | boundary_at_fixed | H | -21.61600 | 2.41538 | [-23.67, -18.4, -19.729999999999997, -23.67499999999999, -22.604999999999997] |
| stress | boundary_at_decay | D | 0.63718 | 0.07432 | [0.6228333333333336, 0.7600166666666666, 0.5845499999999999, 0.5741333333333334, 0.64435] |
| stress | boundary_at_decay | C | -0.14560 | 0.00422 | [-0.15138333333333331, -0.14718333333333333, -0.14628333333333332, -0.14044999999999996, -0.1427166666666667] |
| stress | boundary_at_decay | H | -15.03000 | 0.46255 | [-14.619999999999997, -14.475000000000001, -15.525000000000006, -15.155000000000001, -15.375] |
| stress | decay_in_relative | D | -0.02390 | 0.03121 | [-0.012550000000000061, -0.057366666666666566, -0.03266666666666662, 0.02388333333333348, -0.04081666666666672] |
| stress | decay_in_relative | C | 0.03901 | 0.01282 | [0.032366666666666655, 0.03548333333333331, 0.060966666666666613, 0.028166666666666673, 0.03808333333333336] |
| stress | decay_in_relative | H | 0.74600 | 0.23047 | [0.6899999999999977, 0.9099999999999966, 1.0550000000000068, 0.5400000000000063, 0.5349999999999966] |
| stress | decay_in_boundary | D | -0.08540 | 0.11617 | [-0.06796666666666651, 0.11039999999999983, -0.14970000000000017, -0.1673333333333331, -0.1524000000000001] |
| stress | decay_in_boundary | C | 0.01509 | 0.01871 | [0.025333333333333374, -0.018033333333333346, 0.020050000000000012, 0.026683333333333392, 0.021399999999999975] |
| stress | decay_in_boundary | H | 7.33200 | 2.20795 | [9.740000000000002, 4.834999999999994, 5.259999999999998, 9.059999999999995, 7.7649999999999935] |
| stress | interaction | D | -0.06150 | 0.13695 | [-0.05541666666666645, 0.1677666666666664, -0.11703333333333354, -0.1912166666666666, -0.11158333333333337] |
| stress | interaction | C | -0.02393 | 0.02239 | [-0.00703333333333328, -0.05351666666666666, -0.0409166666666666, -0.001483333333333281, -0.016683333333333383] |
| stress | interaction | H | 6.58600 | 2.39669 | [9.050000000000004, 3.924999999999997, 4.204999999999991, 8.519999999999989, 7.229999999999997] |

- Final R-Decay: C improved in 5/5 seeds; D decreased in 5/5; H increased in 5/5. The complete seed deltas are in paired_differences.csv.
- Final B-Fixed: C improved in 0/5 seeds; D decreased in 0/5; H increased in 0/5. The complete seed deltas are in paired_differences.csv.
- Final B-Decay: C improved in 0/5 seeds; D decreased in 0/5; H increased in 0/5. The complete seed deltas are in paired_differences.csv.
- Stress R-Decay: C improved in 5/5 seeds; D decreased in 4/5; H increased in 5/5. The complete seed deltas are in paired_differences.csv.
- Stress B-Fixed: C improved in 0/5 seeds; D decreased in 0/5; H increased in 0/5. The complete seed deltas are in paired_differences.csv.
- Stress B-Decay: C improved in 0/5 seeds; D decreased in 0/5; H increased in 0/5. The complete seed deltas are in paired_differences.csv.

## Mechanism and Q diagnostics

With no stochastic turn or toggle, interior (10,10,11,10,right,2) and wall (23,10,24,10,right,2) have identical R row 9993, but Stay yields relative (3,0,right,2) versus (−1,0,left,2). B assigns different rows. This proves boundary aliasing, not that any particular tabular boundary model must improve at the same budget.

128 paired unaligned Fire/Stay checks share physical next state and target RNG. Distance/close/potential terms coincide; Stay training reward and same-frozen-table one-step TD target exceed Fire by exactly one, within floating-point tolerance. This is an update-target identity, not a convergence assumption about a learned Q table.

Common-state diagnostics below use a fixed set sampled before pilot from Rule and uniform-random trajectories, independently of the learned policies. State weighting gives one vote per distinct physical state; event weighting uses its frequency in that fixed sample. Q values and Bellman residuals are computed without guard. Bellman residuals use the exact conditional expectation over six turn/toggle outcomes, not one noisy sampled target. They remain physical-state residuals for aggregated tables; irreducible aliasing can contribute.

| Condition | Weighting | Fire>Stay | Greedy invalid Fire | Fire residual MAE | Stay residual MAE | Fire updates | Stay updates | Fire unvisited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-Fixed | state | 39.61% | 1.35% | 65.892 | 65.420 | 1182.8 | 5925.8 | 2.6% |
| R-Fixed | event | 39.18% | 1.69% | 57.842 | 57.390 | 1609.5 | 8267.9 | 2.3% |
| R-Decay | state | 32.78% | 0.62% | 66.013 | 65.802 | 524.0 | 2466.8 | 2.6% |
| R-Decay | event | 29.88% | 0.54% | 57.901 | 57.654 | 684.3 | 3331.7 | 2.3% |
| B-Fixed | state | 31.11% | 5.46% | 20.784 | 20.393 | 729.8 | 1504.4 | 36.7% |
| B-Fixed | event | 33.44% | 7.10% | 20.584 | 20.115 | 1087.5 | 2265.1 | 32.3% |
| B-Decay | state | 24.82% | 2.76% | 22.311 | 21.983 | 288.2 | 2561.6 | 36.8% |
| B-Decay | event | 24.41% | 2.89% | 22.164 | 21.674 | 418.1 | 3920.5 | 32.6% |

Boundary/interior, speed 1/2 and pre-action-distance groups are in q_diagnostics_per_seed.csv. Per-physical-state final records are saved for each run. Full checkpoint distributions and Q changes are in runs/*/seed_*/q_diagnostics.csv. Quantiles there are explicitly distinct-state quantiles; event-weighted means and fractions use actual repeated sampling frequencies. q_own_trajectory.csv reports the same ranking questions on each policy's own visited states, including frequency≥5 states. Those differing visitation distributions are descriptive and cannot establish a representation effect by themselves. A reduced error fraction can also reflect unvisited, zero-tied pairs; count and unvisited columns must be read with the error rates.

## Dynamics and shooting

boundary_speed_slices.csv retains historical boundary (either object within three cells of an edge before action) and post-motion target-speed slices. event_curves.csv uses −5…+15 ticks around bounce, turn and toggle events. Same-type events within 30 ticks of an accepted event are excluded; other-type overlap is retained for curves. Recovery requires three consecutive distance≤1 post-motion ticks, starts at the first of those ticks, and is censored by the next random turn/toggle, 30-tick horizon or window end. recovery.csv reports recovery fractions and conditional times among recovered events; censored events must not be read as successful fast recoveries. All are descriptive policy-specific comparisons. Alignment opportunities and firing conditional on alignment are included in per_seed.csv.

All raw/guard pairs preserve the full physical trajectory, distance, close ticks and hit count. Every discarded unaligned Fire raises base return by one. Native execution was additionally checked against two independent original Python environments and the existing FiringGuard on 80 windows. Main comparisons use guard to isolate tracking; raw invalid firing remains in per_seed.csv. Guard never participates in training or bootstrap.

## Artifacts and verification

- [Paired seed figure](figures/paired_seeds.png), [learning curves](figures/learning_curves.png), [event figure](figures/boundary_dynamic_events.png), [D/C/H tradeoffs](figures/D_C_H_tradeoff.png), [Q stability](figures/q_stability.png).
- [Demo 5400000](demos/scenario_5400000.gif), [demo 5400001](demos/scenario_5400001.gif), [demo 5400002](demos/scenario_5400002.gif). Four guarded panels share each target route and use training seed 60. Each includes the initial frame plus all 300 ticks, with all 301 decoded pixel arrays and durations checked. Actual-player observations are separately recorded in demos/player_check.json when performed.
- [Protocol](protocol.json), [reproduction](REPRODUCE.md), [addendum](paper_addendum.md), [resource log](resources.csv), [evaluation checks](evaluation/checks.json).
- Tests, commands, actual exit codes and timing are in preflight.json, postflight.json and execution/. Verification reuses saved models; it does not repeat the 100M training.

The original tracked files and model hashes are preserved; historical_integrity.json records the final comparison. No commit or push was performed. Inventory hashing excludes its own manifest and explicitly mutable validation receipts to avoid circular hashes.

## Evidence-specific supplement

See [the measured interpretation and next research recommendation](conclusions.md) for factor effects, seed/scene attribution, residual caveats and the actual-player limitation.

## Validation contract repair and final acceptance

The frozen artifact verifier initially passed saved int16 initial-state arrays to the historical target-hash function, whose original evaluation contract uses a six-integer tuple. NumPy-array bytes contain two bytes per int16 field, while tuple bytes contain one byte per coordinate. The explicit `verify_artifacts.py` adapter restores the actual evaluation contract. All original validation checks still execute; no expected digest, model, historical file, protocol or frozen source hash was changed. Both ordinary Python and python -O artifact checks pass with this adapter. The original failed attempts remain in postflight.json and execution/ for review. Use the corrected commands in REPRODUCE.md.

Final test evidence: full suite 89/89, new-study optimized suite 7/7, both historical publication verifiers in ordinary and optimized Python, and exhaustive frozen-artifact checks of 220 checkpoints, 100M final updates, 28,000 evaluation windows and 903 GIF frames. Additional CSV and final common-state Q diagnostics were independently recomputed from saved models. File inventory checks are recorded separately after manifest creation. Actual-player observation remains blocked as described in demos/player_check.json; decoding and static inspection passed.
