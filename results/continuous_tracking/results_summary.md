# Tracking25 V2: Continuing-Tracking Development Results

This is a new fully observed continuing-tracking simulation. A hit does not terminate or freeze the target. Every evaluation window contains all 300 ticks, and target motion is paired across policies. Each learned policy chooses its own slow/fast actions; no heuristic override is applied.
Both methods use training seed 0, zero float32 initialization and 20,000,000 interactions. Validation scenarios are 3600000–3600199; no independent final test or multi-seed formal experiment is used. The earlier hit-once trial measures a different task.

| Policy | Mean distance | Close fraction | Hits/window | Shots/window | Misses/window | Base return | Pooled accuracy | First acquisition (capped) | Acquired fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 16.2538 | 2.623% | 2.6400 | 38.045 | 35.405 | -24.0050 | 6.939% | 131.810 | 88.000% |
| potential | 1.7017 | 67.662% | 70.5350 | 77.845 | 7.310 | 683.0400 | 90.610% | 15.945 | 100.000% |
| rule | 1.3819 | 72.512% | 44.5250 | 44.525 | 0.000 | 430.2500 | 100.000% | 9.885 | 100.000% |
| random | 17.1609 | 0.633% | 0.0500 | 29.430 | 29.380 | -43.8800 | 0.170% | 175.745 | 68.000% |

## Speed Choice and Observed Recovery

| Policy | Slow movement share | Fast movement share | Recovery fraction | Mean capped recovery lag | Actual displacement/tick | Target nominal speed/tick | Target path/tick | Target net displacement/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.6131 | 0.3869 | 0.1802 | 17.9057 | 1.1803 | 1.5028 | 1.5028 | 1.4628 |
| potential | 0.3089 | 0.6911 | 0.9945 | 1.4850 | 1.4979 | 1.5028 | 1.5028 | 1.4628 |
| rule | 0.5972 | 0.4028 | 0.9864 | 1.3836 | 1.4977 | 1.5028 | 1.5028 | 1.4628 |
| random | 0.5011 | 0.4989 | 0.0944 | 19.0379 | 1.4386 | 1.5028 | 1.5028 | 1.4628 |

Movement shares exclude stay/fire and are null when there are no movement actions. Accuracy is pooled from hit/shot counts, not an average of per-window percentages. Distance and proximity include every post-motion tick. First acquisition uses the first post-step distance <=1; non-acquisition is capped at 300 with a separate flag.
Eligible recovery events are deduplicated ticks with a random turn, speed toggle or reflection at tick <=280. Recovery searches lags 0–20; failures receive capped lag 20 and recovered=false. Later events can overlap the window: this is observed recovery around changes, not an isolated causal reaction time.

## Execution and Mechanism Checks

- base: 20,000,000 interactions; 66,666 full windows and a 200-tick partial window; elapsed 306.144 seconds; exact final-model reload verified.
- potential: 20,000,000 interactions; 66,666 full windows and a 200-tick partial window; elapsed 311.490 seconds; exact final-model reload verified.

Timing includes training, checkpoint validation, streamed logs, final save and chunked reload verification. Evaluation is independent of training environment and exploration randomness. No eleven-table checkpoint archive is saved. Both runs begin from zero. Configurations, seed derivation, original source hashes and demo selection were recorded before training in protocol.json.
Validation records: 22 summaries, 4400 windows, 1,320,000 ticks. Baselines: 400 windows, 120,000 ticks. Target path hashes match across all checkpoints and policies for all 200 scenarios.

## Demonstrations

The first three qualifying seeds were selected before training using target motion alone: x/y span >=16 and at least 50 ticks at each speed. These coverage-selected illustrations are not the unbiased evaluation set. All four panels continue through 300 ticks after hits. A hit star marks the actual shot coordinate before target motion. Playback is about 15 FPS and does not change physics.
- Seed 3601000: x/y spans 24/24, speed-1/2 occupancy 185/115 ticks; 301 GIF frames, 20.07 seconds playback.
- Seed 3601001: x/y spans 24/24, speed-1/2 occupancy 168/132 ticks; 301 GIF frames, 20.07 seconds playback.
- Seed 3601002: x/y spans 22/24, speed-1/2 occupancy 173/127 ticks; 301 GIF frames, 20.07 seconds playback.

## Limitations and Reproduction

Mechanics are checked separately from learned skills. This single training seed and finite development set do not establish statistical significance or real-game generalization. Sustained tracking, hit counts, speed choices and misses must be interpreted together. A policy can maintain proximity without firing accurately. The predictive rule is a velocity-aware heuristic, not a proven optimum. No training budget or physics was adjusted after observing results. GUI behavior remains unverified; exported GIFs are checked separately.
Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage check
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage train
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage evaluate
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage render
```

Completed training stages are reused only after configuration, model hash and interaction accounting checks. Interrupted training is not resumable: no full RNG/training state is saved. Use a clearly named new session directory for incomplete or incompatible work. Historical experiments and Git identity settings are preserved; no GitHub push is performed.

## Skills Observed After the Fixed Training Budget

Mechanism correctness is supported by the 52 passing tests and artifact audit: every tracking window continues through hits; target paths are paired across all policies and checkpoints; truncation retains bootstrap and the actual next potential; model reloads are exact. All 285 preserved historical files and all frozen V2 source hashes still match. No archived experiment, Git identity setting, or GitHub repository was changed.

Base Q-learning learned weak sustained tracking: mean distance was 16.2538 cells, close tracking 2.623%, and pooled accuracy 6.939%, with 2.64 hits/window. Potential Q-learning substantially improved these measures within this single-seed development comparison: distance 1.7017, close tracking 67.662%, pooled accuracy 90.610%, and 70.535 hits/window. This does not establish statistical significance or real-game generalization.

The rule heuristic was better at proximity (distance 1.3819; close tracking 72.512%), first acquisition (9.885 ticks versus 15.945), and shot accuracy (100%). Potential fired more often (77.845 shots/window versus 44.525) and earned more hits and base return, but missed 7.31 shots/window. Thus the learned policy did not dominate every tracking criterion.

Both learned policies used both speeds without an override. Potential chose fast actions for 69.106% of its movement actions; its fast-action shares in distance bins 0–3, 4–11, and >=12 were 70.563%, 68.418%, and 39.834%. The observed choices do not follow a rigid far=fast rule. Complete counts, actual displacement, target speed/path/net distances, acquisition flags and recovery statistics are retained in evaluation/comparison.csv and the episode records.

Both training commands and all four workflow stages completed with exit code 0. Training durations were 306.144 seconds for base and 311.490 seconds for potential, including validation, streamed logs, final saving and reload checks. Exported clips contain 301 frames each and 20.07 seconds of playback. Start, middle and final frames from all three GIFs, the hit flash location, and all figures were inspected. Interactive GUI behavior remains unverified. No runtime or training failure required a rerun or protocol change.
