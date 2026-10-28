# Tracking25 V2: Continuing Tracking

This independent local experiment extends the simulation to a target that moves in two dimensions, changes heading and speed, and continues after every hit. It measures sustained tracking over 300-tick windows. The earlier hit-once task and its results remain unchanged.

The observation is `(xc, yc, xt, yt, heading, target_speed)`. Heading IDs are right/down/left/up (0–3); speed is 1 or 2 cells per tick. At reset, the complete four-coordinate tuple is sampled uniformly until Manhattan separation is at least 12. Crosshair actions include one-cell and three-cell horizontal/vertical moves, stay, and fire. Clipping affects crosshair displacement; target reflection is applied one unit substep at a time.

Each action executes the crosshair movement and checks any shot against the current target location, then draws `u_turn`, `u_sign`, and `u_speed` regardless of a hit. A perpendicular turn has probability 0.05 and a speed toggle probability 0.05. The target then moves. Hits award +10 above the −0.05 step cost; misses add −1. A hit neither terminates nor freezes play. `terminated` is always false; only tick 300 truncates. Even a hit at tick 300 counts and still permits target motion before the final observation is returned.

Both learned policies use zero float32 Q tables of shape `(25,25,25,25,4,2,10)`, 125,000,000 element bytes each. Speed maps 1→0 and 2→1. Q-learning uses alpha 0.15 and gamma 0.99, bootstrapping from the actual next state on every transition, including hits and external truncation. Potential shaping uses the negative Manhattan distance exactly once before updating; it does not zero the next potential on a hit or truncation. Training and evaluation both break greedy ties by the lowest action ID.

The frozen development protocol is base/potential × seed 0, with exactly 20,000,000 interactions each. Epsilon decreases from 1.00 to 0.05 over 16,000,000 interactions. Environment and exploration generators are separate children of the root SeedSequence; their derivation is recorded. Each run contains 66,666 complete 300-tick windows plus a final 200-tick partial window. Training episode summaries are streamed to CSV. Only the final full Q table is saved; validation metrics are collected at step 0 and every 2,000,000 steps on seeds 3600000–3600199.

The predictive rule fires if currently aligned; otherwise it predicts one target update from the current heading/speed without future randomness and selects the clipped movement or stay action minimizing predicted distance. Ties prefer smaller actual displacement, then lower action ID. This is a velocity-aware heuristic, not a proven optimum. Random baseline actions use a separate per-scenario `SeedSequence([314159, scenario_seed])` generator.

Metrics include post-motion distance, proximity (distance ≤1), hits, shots, misses, pooled shot accuracy, base return, capped first acquisition with an acquired flag, slow/fast movement shares, actual displacement, and target nominal/path/net distances. No shots means null accuracy. Movement shares exclude fire and stay. Recovery uses deduplicated event ticks ≤280, searches lag 0–20, and includes failed recoveries as capped lag 20. Overlapping changes mean this is observed recovery, not isolated causal reaction time. The comparison uses one training seed, so no across-seed error bands or statistical-significance claims are appropriate.

Before training, demo candidates 3601000–3601099 are scanned using target motion only. The first three with x/y spans at least 16 and at least 50 ticks at each speed are selected and recorded. Each four-panel GIF includes the initial frame and all 300 ticks, approximately 20 seconds at 15 FPS. Hit flashes use the actual pre-motion shot coordinate; panels continue after hits. Models are loaded sequentially to record traces, released, and then rendered offline one clip at a time. GUI verification is separate from file export.

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage check
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage train
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage evaluate
.\.venv\Scripts\python.exe -m aimrl.tracking25 --config configs/tracking25_v2.json --stage render
```

Outputs are under `results/continuous_tracking/`. Completed training is skipped only after configuration, model-hash, and interaction-count verification. Incomplete runs are preserved and require a new output directory; interrupted-state resume is not implemented because full RNG/training state is not saved. Render reads saved results and never trains. No final-test scenarios, GitHub publication, Git identity changes, or history rewriting are part of this experiment.

The potential-difference formula and external-time-limit bootstrap convention follow Ng, Harada and Russell, *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping* (1999), and Gymnasium, *Handling Time Limits*. They do not guarantee good finite-budget learning.
