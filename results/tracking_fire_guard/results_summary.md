# Inference-only firing guard

The previous and new original policies met the full simultaneous target for none and none, respectively. The guarded simultaneous result was `close_0p5`. The guarded retention set was `distance_1p0`, `close_0p5`, `close_2p0`, `close_5p0`. The cases with an actual hit loss inside the retention gate were `distance_1p0`. All comparisons use a matched reference on the same split and with the same variant. The full screen requires at least five percentage points more close ticks, no larger D, and accuracy at least 98%. The simultaneous hit gate is +2%. The retention gate permits a 2% loss. A passing guarded method keeps the original reward-conditioned tracking and hit count. Its eligibility changes because the guard removes misses.

## Intervention and pairing

Q-learning + firing guard queries the original greedy policy once per tick. It substitutes Stay only for Fire with nonzero pre-motion Manhattan distance. Both actions leave the crosshair fixed and consume the same target randomness. For this event the actual reward rises by exactly one because the miss penalty is removed. The transition, continuation term and proximity reward are identical for the paired action. The true environment therefore prefers Stay in this case, while the measured greedy tables preferred Fire. No Q-table or update count changed.

The 21 frozen policies were replayed on the first ten old final scenarios with independent Python environments. The gate compared physical state and target RNG state every tick, plus actual executed actions, distance, hits, misses and base rewards. The guard preserved state and distance at every paired tick. The held-out ranges 4500000–4500499 and 4600000–4600199 were checked for occupancy and frozen before evaluation. The latter uses 0.10 turn and speed-toggle probabilities. Rule is listed once per split and its actions were unaffected.

## Held-out final results

| Condition | D | C (%) | H / window | A mean, original → guard (%) | A pooled, original → guard (%) | Misses, original → guard | Base return, original → guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 1.526 ± 0.135 | 58.41 ± 4.69 | 81.22 ± 4.78 | 99.32 ± 0.86 → 100.00 ± 0.00 | 99.35 → 100.00 | 0.53 → 0.00 | 796.63 → 797.16 |
| distance_0p1 | 1.434 ± 0.013 | 60.47 ± 2.03 | 83.90 ± 0.05 | 95.43 ± 7.17 → 100.00 ± 0.00 | 95.06 → 100.00 | 4.36 → 0.00 | 819.60 → 823.97 |
| distance_0p3 | 1.484 ± 0.076 | 59.67 ± 3.05 | 80.85 ± 5.00 | 99.00 ± 0.43 → 100.00 ± 0.00 | 99.02 → 100.00 | 0.80 → 0.00 | 792.68 → 793.48 |
| distance_1p0 | 1.520 ± 0.073 | 65.73 ± 6.84 | 80.41 ± 4.92 | 95.53 ± 6.50 → 100.00 ± 0.00 | 95.11 → 100.00 | 4.14 → 0.00 | 784.98 → 789.11 |
| close_0p5 | 1.404 ± 0.021 | 65.20 ± 2.17 | 84.21 ± 0.09 | 95.59 ± 6.21 → 100.00 ± 0.00 | 95.31 → 100.00 | 4.14 → 0.00 | 823.01 → 827.15 |
| close_2p0 | 1.357 ± 0.011 | 69.71 ± 0.75 | 81.58 ± 4.88 | 90.36 ± 15.88 → 100.00 ± 0.00 | 87.94 → 100.00 | 11.19 → 0.00 | 789.58 → 800.77 |
| close_5p0 | 1.340 ± 0.053 | 73.19 ± 1.81 | 81.29 ± 4.97 | 86.20 ± 11.70 → 100.00 ± 0.00 | 84.79 → 100.00 | 14.58 → 0.00 | 783.36 → 797.95 |
| rule | 1.381 | 73.01 | 43.48 | 100.00 | 100.00 | 0.00 | 419.82 |

Values with uncertainty are seed means ± sample SD over seeds 20–22. Rule is a single policy. For each seed accuracy is total hits divided by actual shots over its scenarios. The seed mean gives every defined training seed equal weight, while the pooled ratio sums hits and shots over the three seeds before division. Zero executed shots give NA and any missing seed accuracy fails the learned-method accuracy gate. D, C and H are identical between the two variants.

| Condition | ΔD | ΔC (pp) | ΔH (%) | Simultaneous gate | Retention gate |
| --- | ---: | ---: | ---: | --- | --- |
| distance_0p1 | -0.093 | +2.06 | +3.30 | fail | fail |
| distance_0p3 | -0.043 | +1.25 | -0.45 | fail | fail |
| distance_1p0 | -0.007 | +7.32 | -0.99 | fail | pass |
| close_0p5 | -0.122 | +6.79 | +3.69 | pass | pass |
| close_2p0 | -0.169 | +11.30 | +0.44 | fail | pass |
| close_5p0 | -0.187 | +14.77 | +0.10 | fail | pass |

The objective coordinates and non-dominated mean sets are unchanged by the guard. The plotted rings overlap the corresponding original mean points exactly. See [objective trade-offs](figures/objective_tradeoffs.png) and [accuracy and misses](figures/accuracy_and_misses.png). The [threshold table](thresholds.csv) also gives the pooled-accuracy sensitivity, with the historical and new original results preserved.

## Held-out stress results

| Condition | D | C (%) | H / window | A mean, original → guard (%) | A pooled, original → guard (%) | Misses, original → guard | Base return, original → guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 1.683 ± 0.123 | 53.35 ± 4.10 | 71.23 ± 4.16 | 99.06 ± 1.00 → 100.00 ± 0.00 | 99.10 → 100.00 | 0.65 → 0.00 | 696.63 → 697.28 |
| distance_0p1 | 1.582 ± 0.008 | 55.22 ± 1.54 | 73.44 ± 0.17 | 94.66 ± 7.94 → 100.00 ± 0.00 | 94.20 → 100.00 | 4.52 → 0.00 | 714.87 → 719.38 |
| distance_0p3 | 1.603 ± 0.060 | 54.67 ± 2.90 | 70.90 ± 4.28 | 98.25 ± 0.77 → 100.00 ± 0.00 | 98.28 → 100.00 | 1.24 → 0.00 | 692.73 → 693.97 |
| distance_1p0 | 1.603 ± 0.075 | 60.35 ± 6.22 | 70.85 ± 4.06 | 95.07 ± 6.82 → 100.00 ± 0.00 | 94.64 → 100.00 | 4.01 → 0.00 | 689.51 → 693.52 |
| close_0p5 | 1.575 ± 0.021 | 59.25 ± 1.79 | 73.78 ± 0.33 | 95.22 ± 5.86 → 100.00 ± 0.00 | 94.95 → 100.00 | 3.92 → 0.00 | 718.88 → 722.80 |
| close_2p0 | 1.536 ± 0.015 | 63.28 ± 0.79 | 71.60 ± 4.14 | 89.93 ± 16.11 → 100.00 ± 0.00 | 87.43 → 100.00 | 10.29 → 0.00 | 690.76 → 701.05 |
| close_5p0 | 1.517 ± 0.054 | 66.58 ± 1.75 | 71.27 ± 4.04 | 85.72 ± 12.08 → 100.00 ± 0.00 | 84.26 → 100.00 | 13.31 → 0.00 | 684.35 → 697.67 |
| rule | 1.457 | 68.72 | 41.70 | 100.00 | 100.00 | 0.00 | 402.00 |

## Why the original policies fired while unaligned

All 82,391 filtered proposals had Q(Fire) > Q(Stay), with zero exact Fire–Stay ties. There were 115 events with a maximum tie involving another action. The final event-weighted median gap was 1.105, with median Fire and Stay visitation counts 12,364 and 40,318. On the final set, 114 events had an unvisited Fire or Stay action and 581 had fewer than 10 updates on at least one. Both actions had at least 100 updates on 58,809 events (98.64%). At pre-distance one there were 58,057 events. Finite sampling is compatible with the low-count tail. Most errors persisted in frequently updated pairs. Counts and Q gaps do not identify a single cause, such as transition aliasing, persistent exploration, or unequal visitation histories. Repeated visits to the same table row are descriptive events.

The [seed diagnostics](q_diagnostics_per_seed.csv), [distinct indexed rows](q_diagnostics_distinct_states.csv) and per-tick Q event files retain the estimates, counts, ties and repetitions. Stay has the lower action ID, so an exact tie between Fire and Stay would not select Fire under the existing greedy rule. The guard repairs shot selection on already trained policies. It leaves the timing trade-off of an aligned hit unchanged, because a valid Fire still occupies the tick while the target moves.

## Historical reconciliation

The [recomputed historical seed table](historical_per_seed.csv), [comparison](historical_comparison.csv) and [paired differences](historical_paired_differences.csv) agree with the recorded values. For example, historical final Close 5.0 accuracy is 86.70% as an equal-weight seed mean and 85.36% after pooling all shots.

The prior inner matrix timer was 115.266 s. It starts after protocol freeze and covers the 21 runs, checkpoint writes, development replays, streaming window logs and exact final reloads. The prior train phase was 120.317 s, and the outer command took 120.753 s including process overhead. Final and stress evaluation and rendering were separate. The native C11 DLL was compiled with Zig 0.15.2 with fast-math and FP contraction disabled, and called from Python using ctypes. The reset path and NumPy PCG64 streams came from Python. The count comes from 349,986 complete 300-tick windows and 21 200-tick tails, totaling 105,000,000 actual updates. The archived Python training parity was exact for 1,001 steps at zero added weight and for all seven weights in a greedy-heavy regime. This study independently matched 126,000 native/Python inference ticks with the frozen DLL, including actions, physical states and event flags. It did not repeat training or claim full-length Python training equivalence.

## Execution and outputs

The study ran zero training interactions. It verified 63,000 paired gate ticks, 21,500 held-out final windows and 8,600 stress windows, including Rule. The [execution checks](execution_checks.json) and [protocol](protocol.json) give counts, frozen hashes and commands. The [paired differences](paired_differences.csv) compare guard minus original for each seed, and [paired versus reference](paired_vs_reference.csv) compares weights within the same variant and seed. The [synchronized GIF](demo/synchronized_fire_guard.gif) is a preselected Close 5.0 seed-20 replay, decoded and verified over 301 frames.
