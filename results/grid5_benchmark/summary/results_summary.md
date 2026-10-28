# Grid5 benchmark Formal Experiment Record

Each method was trained from zero for 100,000 steps using root seeds 0, 1, and 2. Validation used scenario seeds 100000–100199; final testing used seeds 200000–200499 after the final models were fixed. Error terms are sample standard deviations across three training seeds, not confidence intervals.
Execution environment: Windows-11-10.0.26200-SP0, Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2.
Sum of elapsed times for six training runs: 13.633 seconds; training: 2026-10-27T02:27:37.975674+00:00 to 2026-10-27T02:27:52.184711+00:00; testing: 2026-10-27T02:27:55.767211+00:00 to 2026-10-27T02:27:56.932569+00:00.

## Validation (Mean ± Sample Standard Deviation)

| Method | Training steps | Success rate | Mean steps | Mean misses |
|---|---:|---:|---:|---:|
| base | 0 | 0.00% ± 0.00% | 50.000 ± 0.000 | 0.000 ± 0.000 |
| base | 10000 | 67.50% ± 7.57% | 27.260 ± 2.634 | 1.292 ± 0.024 |
| base | 30000 | 98.67% ± 0.29% | 7.932 ± 0.853 | 0.100 ± 0.040 |
| base | 100000 | 100.00% ± 0.00% | 4.972 ± 0.078 | 0.000 ± 0.000 |
| potential | 0 | 0.00% ± 0.00% | 50.000 ± 0.000 | 0.000 ± 0.000 |
| potential | 10000 | 100.00% ± 0.00% | 7.608 ± 0.731 | 0.160 ± 0.172 |
| potential | 30000 | 100.00% ± 0.00% | 5.145 ± 0.185 | 0.070 ± 0.061 |
| potential | 100000 | 100.00% ± 0.00% | 4.710 ± 0.057 | 0.000 ± 0.000 |

## Two Consecutive Validation Checkpoints at ≥90%

| Method | Training seed | First checkpoint | Confirmation checkpoint |
|---|---:|---:|---:|
| base | 0 | 20000 | 30000 |
| base | 1 | 30000 | 40000 |
| base | 2 | 20000 | 30000 |
| potential | 0 | 10000 | 20000 |
| potential | 1 | 10000 | 20000 |
| potential | 2 | 10000 | 20000 |

base reached the threshold in 3/3 runs; mean first checkpoint: 23333 steps; sample standard deviation: 5774 steps.

potential reached the threshold in 3/3 runs; mean first checkpoint: 10000 steps; sample standard deviation: 0 steps.

The checkpoint resolution is 10,000 steps; these observations do not establish an exact training-speed ratio.

## Final Test

| Method | Move probability | Success rate | Mean steps | Mean misses |
|---|---:|---:|---:|---:|
| base | 0.10 | 100.00% ± 0.00% | 5.973 ± 0.414 | 0.000 ± 0.000 |
| base | 0.25 | 100.00% ± 0.00% | 5.234 ± 0.113 | 0.000 ± 0.000 |
| base | 0.50 | 100.00% ± 0.00% | 5.207 ± 0.112 | 0.000 ± 0.000 |
| potential | 0.10 | 100.00% ± 0.00% | 5.243 ± 0.089 | 0.018 ± 0.025 |
| potential | 0.25 | 100.00% ± 0.00% | 4.885 ± 0.061 | 0.011 ± 0.012 |
| potential | 0.50 | 100.00% ± 0.00% | 5.039 ± 0.012 | 0.000 ± 0.000 |
| rule | 0.10 | 100.00% | 4.576 | 0.000 |
| rule | 0.25 | 100.00% | 4.972 | 0.000 |
| rule | 0.50 | 100.00% | 6.408 | 0.000 |
| random | 0.10 | 23.20% | 44.012 | 7.416 |
| random | 0.25 | 23.80% | 43.654 | 7.358 |
| random | 0.50 | 25.20% | 43.102 | 7.178 |

## Paired Differences

For each training seed, paired_differences.csv reports potential − base success rate and base − potential mean steps and misses. A positive difference favors potential shaping for the corresponding metric.
- Training seed 0, move probability 0.10: success-rate difference +0.000, step difference +0.340, miss difference +0.000.
- Training seed 0, move probability 0.25: success-rate difference +0.000, step difference +0.150, miss difference +0.000.
- Training seed 0, move probability 0.50: success-rate difference +0.000, step difference +0.074, miss difference +0.000.
- Training seed 1, move probability 0.10: success-rate difference +0.000, step difference +1.192, miss difference -0.046.
- Training seed 1, move probability 0.25: success-rate difference +0.000, step difference +0.426, miss difference -0.024.
- Training seed 1, move probability 0.50: success-rate difference +0.000, step difference +0.126, miss difference +0.000.
- Training seed 2, move probability 0.10: success-rate difference +0.000, step difference +0.656, miss difference -0.008.
- Training seed 2, move probability 0.25: success-rate difference +0.000, step difference +0.472, miss difference -0.008.
- Training seed 2, move probability 0.50: success-rate difference +0.000, step difference +0.306, miss difference +0.000.

Limitations: only three training seeds were used, and seed 0 was used during development. This small fully observable environment and finite test set do not establish generalization to real games. The rule policy is a baseline, not a proven optimum. Results are descriptive; no statistical-significance claim is made. New test seeds do not guarantee initial states absent from training.
Figures and CSV files were generated from actual raw records; the GUI has not been verified. Command: `python -m aimrl.experiments --protocol results/reward_shaping_development/protocol_frozen.json --output results/grid5_benchmark --stage train`; then use --stage test and --stage summarize in order. Outputs are stored in results/grid5_benchmark.
