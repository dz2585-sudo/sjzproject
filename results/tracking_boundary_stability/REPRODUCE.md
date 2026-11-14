# Reproduction and frozen-artifact verification

Use Python with requirements-reproduction.txt installed. The recorded runtime versions are in protocol.json; this execution used the existing workspace virtual environment and Zig 0.15.2. The C kernel accepts all learning/reward/probability parameters explicitly. Compile without fast-math or floating-point contraction. Grid 25 and ten actions are validated fixed research constraints.

From the repository root, verify saved artifacts without retraining:

```powershell
python -m aimrl.tracking_boundary_stability.verify_artifacts
python -O -m aimrl.tracking_boundary_stability.verify_artifacts
python scripts/run_boundary_stability_checks.py preflight --legacy-kernel <path-to-frozen-tradeoff.dll>
```

For a fresh full replication choose an unused output directory; never overwrite the published result directory. Reuse the frozen seed reservation as an exact replication, not as a new independent experiment:

```powershell
$env:AIMRL_BOUNDARY_OUTPUT = 'results/tracking_boundary_stability_replication'
python -m aimrl.tracking_boundary_stability setup --compiler <path-to-zig.exe>
python scripts/run_boundary_stability_checks.py preflight --legacy-kernel <path-to-frozen-tradeoff.dll>
python -m aimrl.tracking_boundary_stability pilot
python -m aimrl.tracking_boundary_stability train
python -m aimrl.tracking_boundary_stability evaluate
python -m aimrl.tracking_boundary_stability report
python scripts/run_boundary_stability_checks.py preflight --legacy-kernel <path-to-frozen-tradeoff.dll>
```

Setup records historical hashes, seed-collision audit, build command and exact-backend gates. Pilot uses four 1M runs only. Train requires passed preflight and pilot, freezes protocol before formal execution and runs all 20 × 5M updates. It does not resume partially created runs or silently reduce budgets. Checkpoints at 0 and every 500k are preserved; only 5M is used for primary evaluation. Development has 200 scenarios at each checkpoint. Final/Stress are inaccessible until training_complete.json exists. Their ranges, demo seeds, event windows, recovery rule, diagnostics and thresholds are fixed in the config/protocol.

Evaluation saves all guarded full rollouts plus original actions and per-window actual-action metrics. Report recomputes all tables/figures from saved artifacts, then generates fixed demos. Complete GIF decoding is automated; actual-player inspection is a separate manual/UI observation in player_check.json. Do not claim actual playback from decoding alone. Verify catches malformed counts, changed hashes, missing frames and mismatched aggregate metrics using runtime exceptions that remain active under python -O.

Manifest excludes itself and explicitly listed validation receipts; those contain manifest-verification results and cannot be included in their own digest. Update inventory only after reviewable reporting changes, never to hide a changed frozen source/model. Historical hashes and the protocol hash must remain fixed.

## Post-training analysis and canonical verification

The original direct `... tracking_boundary_stability verify` entry point has the preserved array/tuple hash-contract defect. Use the explicit adapter above; it executes every frozen verification check and changes no expected hashes. The original postflight command log intentionally retains its failed verifier attempts. `postflight.json` therefore is an execution history, not the final acceptance verdict. See final_acceptance.json. The full 89-test suite and all historical checkers passed.

After report generation, reproduce the supplemental interpretation with:

```powershell
python -m aimrl.tracking_boundary_stability.evidence
python -m aimrl.tracking_boundary_stability.verify_artifacts
python -O -m aimrl.tracking_boundary_stability.verify_artifacts
```

For the delivered local audit, finalize_artifacts.py additionally checks recorded postflight receipts, recomputes the main CSV tables and final Q diagnostics, writes source/output inventories and preserves all failure records. A fresh replication can run the ordinary postflight runner to obtain the six passed test/history receipts, then the canonical verifier commands to repair only its two serialization-sensitive checks:

```powershell
python scripts/run_boundary_stability_checks.py postflight --legacy-kernel <path-to-frozen-tradeoff.dll>
# The two original artifact-verifier commands expose the documented type-contract defect.
python -m aimrl.tracking_boundary_stability.verify_artifacts
python -O -m aimrl.tracking_boundary_stability.verify_artifacts
python -m aimrl.tracking_boundary_stability.finalize_artifacts
python -m aimrl.tracking_boundary_stability.verify_artifacts
python -O -m aimrl.tracking_boundary_stability.verify_artifacts
```

The evidence, canonical verifier and finalizer were added after training solely for interpretation and artifact validation. Their hashes and scope are recorded separately in source_inventory.csv and post_training_analysis.json; the original frozen protocol remains byte-for-byte unchanged. The player receipt describes this local UI limitation and must be replaced with a genuine observation in a new execution, never inferred from decoding.
