# AimRL 25x25 Development Trial

This local single-seed comparison uses base and potential Q-learning, each trained from zero for 5,000,000 interactions with seed 0. It is neither a multi-seed formal experiment nor an independent final test.
The grid is 25x25, the cap is 100 steps, gamma is 0.99, alpha is 0.15, and shaping scale is 1.0. Epsilon decays from 1.00 to 0.05 over 4,000,000 interactions. Target move/turn probabilities are 0.25/0.10. Both the cap and gamma differ from the historical 5x5 experiment, so differences cannot be attributed to grid size alone.
Validation uses 200 development scenarios, seeds 3500000–3500199, at steps 0 and every 500,000 interactions. Final learned-policy results reuse the last validation records. Rule and random policies are independently evaluated once on the same scenarios. Random actions use SeedSequence([314159, scenario_seed]); the training entry point also writes separate baseline summaries, which are not pooled here.

| Policy | Success rate | Mean steps | Mean misses | Mean base return |
|---|---:|---:|---:|---:|
| base | 6.0% | 96.455 | 4.470 | -8.6928 |
| potential | 100.0% | 24.675 | 0.180 | 8.5862 |
| rule | 100.0% | 20.835 | 0.000 | 8.9582 |
| random | 2.0% | 99.450 | 16.510 | -21.2825 |

## Runtime and Record Checks

- base: 5,000,000 training interactions; 51,332 complete episodes; 7 unfinished-episode steps; elapsed 56.548 seconds. Reload matched the final checkpoint; the initial checkpoint was zero.
- potential: 5,000,000 training interactions; 109,264 complete episodes; 15 unfinished-episode steps; elapsed 56.154 seconds. Reload matched the final checkpoint; the initial checkpoint was zero.

Elapsed training durations include independent validation, checkpoint and final-model saving, reload verification, baseline evaluation, and log preparation; they are not pure update timings. Each run has 11 checkpoint summaries and 2,200 validation episodes, for 22 summaries and 4,400 validation episodes combined. Q tables are float32 with shape (25,25,25,25,2,6) and 18,750,000 element bytes each.
Success is an exact-cell hit within 100 steps. Mean steps includes timeouts as 100. Mean misses and base return include all episodes. Shaped learning returns are not compared with base returns as performance scores. One training seed and a finite development scenario set do not establish statistical significance or real-game generalization. A 100% scenario success rate would not imply that every shot hit. Interactive GUI behavior has not been verified.

## Reproduction Commands

Run from the project root; choose a new session path if any target run directory is nonempty.

```powershell
.\.venv\Scripts\python.exe -m aimrl.trial25 --stage check
.\.venv\Scripts\python.exe -m aimrl.train --config configs/trial25.json --seed 0 --reward-mode base --output results/grid25_development/base_seed0
.\.venv\Scripts\python.exe -m aimrl.train --config configs/trial25.json --seed 0 --reward-mode potential --output results/grid25_development/potential_seed0
.\.venv\Scripts\python.exe -m aimrl.trial25 --stage summarize
.\.venv\Scripts\python.exe -m aimrl.trial25 --stage demo
```

Raw run data are in base_seed0/ and potential_seed0/; figures and comparison.csv are in summary/. Real trajectories for demo seed 3501000 and export metadata are in demo/. No 50x50 training, historical final-test evaluation, or GitHub upload is part of this trial.

## Observed Outcomes and Demonstration

Potential shaping reached 100% success on the final 200 development scenarios, compared with 6% for base Q-learning. The rule policy also reached 100% and used fewer steps: 20.835 versus 24.675 for potential. These observations apply to this one training seed and development scenario set.

On the preselected demonstration scenario (seed 3501000), base timed out at step 100 with two misses, while potential and the rule policy each hit at step 14 with no misses. The GIF preserves these outcomes and holds the two completed panels while the base panel continues. The final GIF contains 101 frames at 1260x530; its first, middle and final decoded frames were inspected and checked against their source images with tolerance for GIF palette quantization. The first export had layout/legend defects and is retained in demo_export_attempt1; the corrected export is in demo/. No new dependency was installed.

All 40 unit tests, including six trial-specific smoke/load checks, passed. All task execution commands exited with code 0. The original 224 historical files checked by the preflight retain their recorded bytes. GUI behavior remains unverified. Detailed export checks and the corrected preliminary quantization-check tolerance are recorded in execution_checks.json.
