# Local 25x25 Development Trial

This trial uses `configs/trial25.json` and the unchanged configurable environment, Q-learning agent, reward transformation, and training loop. It compares base and potential rewards with training seed 0 and exactly 5,000,000 interactions per method. It replaces the immediate 50x50 plan and remains local.

Settings: 25x25 grid, 100-step episode cap, gamma 0.99, alpha 0.15, float32 zero-initialized Q tables, and shaping scale 1.0. Epsilon decreases from 1.00 to 0.05 over 4,000,000 interactions. The target moves with probability 0.25 and turns with probability 0.10. Validation uses 200 scenarios with seeds 3500000–3500199 at step 0 and every 500,000 interactions. Unused test configuration fields are explicitly disabled.

`aimrl.trial25` adds trial-specific configuration and resource checks, model loading based on the saved 25x25 configuration, record verification, figures, and a three-panel GIF. It does not call the original formal-experiment entry point or the historical final test. The base 5x5 configuration, core implementations, historical hash records, models, and earlier experiments outputs are retained.

Run from the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m aimrl.trial25 --stage check
.\.venv\Scripts\python.exe -m aimrl.train --config configs/trial25.json --seed 0 --reward-mode base --output results/grid25_development/base_seed0
.\.venv\Scripts\python.exe -m aimrl.train --config configs/trial25.json --seed 0 --reward-mode potential --output results/grid25_development/potential_seed0
.\.venv\Scripts\python.exe -m aimrl.trial25 --stage summarize
.\.venv\Scripts\python.exe -m aimrl.trial25 --stage demo
```

These commands refuse to overwrite nonempty run or export directories. If a completed matching run already exists, check its saved configuration, metadata, and records and reuse it. Use `--output` to select a different session consistently when necessary. `--config` can specify the trial configuration path.

Outputs are under `results/grid25_development/`: each method has a full run directory, `summary/` contains the development comparison and figures, and `demo/` contains the fixed-scenario trajectories, preview, and GIF. The original training entry point writes baseline summaries using its original continuous action stream. The final four-policy comparison instead uses one explicitly recorded per-scenario random baseline stream, `SeedSequence([314159, scenario_seed])`, without pooling those baseline records.

The demonstration uses the preselected scenario seed 3501000. Panels show the final base and potential greedy policies alongside the rule baseline. A completed panel retains its actual final state while the remaining panels continue. The GIF is exported using the already installed Pillow library; no additional dependency is required. Sparse coordinate labels keep the grid readable. File export does not verify interactive GUI behavior.

Training timing includes validation, checkpoint/final-model writes, reload checks, baseline evaluation, and log preparation. There is one training seed, so no across-seed uncertainty band is appropriate. The larger grid also uses a different episode cap and gamma from the historical experiment; effects cannot be attributed to grid size alone. Low success rates, timeouts, and missed shots are retained as observed. This development trial is not an independent final test, evidence of statistical significance, or a demonstration of real-game generalization.
