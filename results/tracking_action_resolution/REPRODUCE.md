# Reproduce the action-resolution study

Run from the repository root. Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2 and Pillow 13.0.0 were used. Install `requirements-reproduction.txt`. The study reads the five R-Decay seed 60–64 checkpoints at `results/tracking_boundary_stability/runs/R-Decay/seed_60/checkpoints/step_5000000.npz` (and seeds 61–64). No copied/fabricated new-action Q values exist. A10 DP bundles are referenced in `precomputation.json`; A14 bundles live here under `cache/` with different action-set identities.

## Audit saved results without reevaluation

```powershell
python -c "from aimrl.tracking_action_resolution.core import config; from aimrl.tracking_action_resolution.analysis import analyze; analyze(config())"
python -m aimrl.tracking_action_resolution verify
python -O -m aimrl.tracking_action_resolution verify
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -p test_tracking_action_resolution.py -v
```

The analysis command reconstructs tables from raw ticks; verification compares them, reconstructs every Bellman array, checks all model/protocol/source digests and original historical files, and decodes all GIF frames. It runs no evaluation environment and makes zero Q updates. Verification writes only its new-study receipts. Do not use the analysis command to overwrite frozen evidence with different definitions.

## Rebuild and independently replicate

Use a **new empty output** and the unchanged checked-in configuration. Setup creates a historical baseline and rechecks scenario reservations. The already-used ranges intentionally reproduce this study; they are not a new independent holdout.

```powershell
$env:AIMRL_ACTION_OUTPUT = '.tmp/action-resolution-replication'
python -m aimrl.tracking_action_resolution setup --compiler <path-to-zig>
python -m aimrl.tracking_action_resolution development
python scripts/check_tracking_action_resolution.py preflight
python -m aimrl.tracking_action_resolution freeze
python -m aimrl.tracking_action_resolution evaluate
python -m aimrl.tracking_action_resolution report
python scripts/check_tracking_action_resolution.py postflight
python -m aimrl.tracking_action_resolution inventory
```

Compiler **Zig 0.15.2**; setup runs `zig cc -std=c11 -O3 -fno-fast-math -ffp-contract=off -shared aimrl/tracking_action_resolution/kernel.c -o <new-output>/native/action_resolution.dll`. Local/global compiler caches go under `.tmp/action-resolution-build`. On Unix the output suffix is `.so`; binary hashes differ by platform, so replicate to a separate protocol rather than replacing the archived DLL. The check runner defaults its historical environment kernels to `.tmp/latest-release-native/{boundary,tradeoff}.dll`; first rebuild those with `python scripts/reproduce_latest_studies.py build --compiler <path-to-zig> --output .tmp/latest-release-native` if absent. Their build/parity instructions are in `../../docs/reproduce_latest_studies.md`. Python tests execute only short correctness updates; they are not formal training runs.

`setup` computes both A10/A14 at every depth, demands exact A10 agreement with the published cache, and checks all 56,250,000 state/terminal/depth subset inequalities using the pre-fixed 1e-9 tolerance. It stores exact maxima while choosing the first ID within 1e-10; approximate tie selection does not redefine V. `validate` compares Python transitions, recursion, shaped/task conversion and native rollout/RNG behavior. `preflight` runs the complete test suite, optimized new tests, both-mode independent validation and historical publication/study checks. Final cannot run until a protocol exists. No training phase is provided.

The evaluation driver owns seeds and draws. The planner's API takes only current physical state and an immutable policy table; its precomputation receives only the known model and terminal. Stress environment probabilities are 0.10; planner remains 0.05. Each 2-cell movement advances the cursor once and the target once; there is no two-step environment emulation.

`protocol.json` freezes all experiment sources, configuration, models, caches, bootstrap, gates and behavior rules before new Final access. `historical_sha256_before.json` and `historical_integrity.json` distinguish unchanged released files from this independent addition. `output_inventory.csv` hashes the delivery and referenced dependencies; `inventory_scope.json` names its self-reference/later-receipt exclusions. Compilation intermediates are not needed; the archived DLL is accompanied by complete C source and build options. Failed setup attempts remain under `execution/`; they happened before precomputation or holdout access.

All three demo GIFs include reset plus every tick (301 frames, 20,070 ms). Full pixels and timings were checked against the renderer. Actual-player inspection was not performed; no Chrome or desktop permission was requested.

## Descriptive post-analysis supplement

After the saved-record analysis, run `python scripts/interpret_tracking_action_resolution.py` to regenerate the English behavioral interpretation from the already frozen action/speed/boundary tables. It adds no metric, scene selection, reward or intervention, and changes no frozen experiment source; its separate input/source hashes are in `interpretation_provenance.json`.

After all receipts are saved, run `python scripts/finalize_action_resolution_inventory.py` to hash the final delivery with exact root-only self-reference exclusions (including nested historical verification receipts). This metadata-only adapter leaves all frozen experiment sources and hashes unchanged.
