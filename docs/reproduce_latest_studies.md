# Reproduce the latest AimRL studies

Use a fresh checkout of this repository. Every required model, checkpoint and planning cache is in the repository; no private drive, absolute source directory or Git LFS download is needed. Planning references the boundary study's ten original 5M models rather than copying their Q tables. Run commands from the repository root and use new output directories. These commands never overwrite the frozen result directories.

## Dependencies and native libraries

The recorded environment uses Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2 and Pillow 13.0.0. Check the exact Python patch/platform in each frozen protocol; the dependencies are pinned in the existing requirements files:

```powershell
python -m pip install -r requirements-reproduction.txt
```

Frozen Windows libraries are included at `results/tracking_boundary_stability/native/boundary.dll` and `results/tracking_planning/native/planning.dll`. Their protocol hashes are preserved. PDB/import-LIB intermediates are intentionally not published. C11 source is in the corresponding `aimrl/` module. Compiler: **Zig 0.15.2**. To rebuild both and the historical tradeoff kernel:

```powershell
python scripts/reproduce_latest_studies.py build --compiler zig --output .tmp/latest-native
```

The actual build commands, exit codes, source/library digests and compiler version are saved to the new directory's `build.json`. The underlying command for each kernel is:

```text
zig cc -std=c11 -O3 -fno-fast-math -ffp-contract=off -shared aimrl/tracking_planning/kernel.c -o .tmp/latest-native/planning.dll
```

Replace the source/output names for `boundary` and `tradeoff`. The driver sets local/global Zig caches inside the specified output directory. Unix outputs use `.so`; platform-specific library bytes need not match the archived Windows DLL. A rebuilt planning engine can reconstruct every frozen Bellman layer as a read-only numerical parity audit:

```powershell
python scripts/reproduce_latest_studies.py verify-planning --planning-library .tmp/latest-native/planning.dll --output .tmp/rebuilt-planning-check
```

The original archived library still supplies the historical fingerprint; the explicitly supplied rebuilt engine must reproduce every saved value, expectation and policy. No protocol hash is changed to accept a new binary. For the boundary kernel, set `AIMRL_BOUNDARY_KERNEL` to the rebuilt library and `AIMRL_TRADEOFF_KERNEL` to the rebuilt tradeoff library before running `python -m unittest discover -s tests -p test_tracking_boundary_stability.py -v`. These are short correctness updates, not a new formal training study.

## Recompute analysis from saved records

```powershell
python scripts/prepare_latest_publication.py --analyze-only .tmp/recomputed-window-tables
```

This sums original per-window integers, recomputes D/C/H/A and between-model sample SD, and checks each policy/model summary. Differences use unrounded values before formatting. It does not run an environment or update Q.

Run the complete original artifact checks through the explicit publication adapter in both Python modes:

```powershell
python scripts/reproduce_latest_studies.py verify-boundary --output .tmp/boundary-check
python -O scripts/reproduce_latest_studies.py verify-boundary --output .tmp/boundary-check-O
python scripts/reproduce_latest_studies.py verify-planning --output .tmp/planning-check
python -O scripts/reproduce_latest_studies.py verify-planning --output .tmp/planning-check-O
python scripts/reproduce_latest_studies.py verify-history --output .tmp/history-check
python -O scripts/reproduce_latest_studies.py verify-history --output .tmp/history-check-O
```

The adapter verifies current publication documents against their reviewed hashes and archived originals against the unchanged historical hashes. Named omitted native intermediates/failed transcripts are checked against the release contract and their public replacements. Every transferred experiment/model/protocol byte keeps its original expected digest. The original numerical, update-count, window, Bellman, action-proposal and GIF checks execute. Receipts are redirected into new output directories. The historical boundary tuple/int16 initial-state canonicalization adapter remains unchanged.

The old direct study verifiers are retained as historical source; on a publication checkout their strict all-files-present / original-README assumptions require the documented release adapter. This is an explicit release difference, not a changed experiment expectation.

## Reevaluate frozen models

These commands perform fresh complete environment evaluation and are **not needed merely to audit the publication**:

```powershell
python scripts/reproduce_latest_studies.py reevaluate --study boundary --split final --output .tmp/boundary-final-replay
python scripts/reproduce_latest_studies.py reevaluate --study boundary --split stress --output .tmp/boundary-stress-replay
python scripts/reproduce_latest_studies.py reevaluate --study planning --split final --output .tmp/planning-final-replay
python scripts/reproduce_latest_studies.py reevaluate --study planning --split stress --output .tmp/planning-stress-replay
```

Optional `--boundary-library` or `--planning-library` supplies a separately rebuilt engine. Outputs are separate; the default method list, scene ranges and budgets are not reduced. Boundary evaluates original and guarded forms; planning evaluates all 20 instances. Planning still uses the fixed 0.05 motion model under real 0.10 Stress dynamics. All frozen Q/count arrays remain read-only.

## Rebuild a full boundary training replication only when needed

No training was repeated for this publication. For an **exact replication**, intentionally reuse the published original reservation and choose a new directory:

```powershell
python scripts/reproduce_latest_studies.py train-boundary --phase setup --compiler zig --output .tmp/boundary-replication
$env:AIMRL_BOUNDARY_OUTPUT = '.tmp/boundary-replication'
python scripts/run_boundary_stability_checks.py preflight --legacy-kernel .tmp/latest-native/tradeoff.dll
python scripts/reproduce_latest_studies.py train-boundary --phase pilot --output .tmp/boundary-replication
python scripts/reproduce_latest_studies.py train-boundary --phase train --output .tmp/boundary-replication
python scripts/reproduce_latest_studies.py train-boundary --phase evaluate --output .tmp/boundary-replication
python scripts/reproduce_latest_studies.py train-boundary --phase report --output .tmp/boundary-replication
```

Setup checks the unchanged original configuration and records the original seed-audit/protocol hashes. Its explicit exact-replication adapter acknowledges that the newer planning study references the same models; it does not misreport those intentional references as a new seed collision or claim a new untouched holdout. The original algorithm, four 1M pilots, twenty 5M formal runs, checkpoint schedule and final endpoint remain unchanged. Use a new seed protocol for a genuinely independent future study.

## Definitions and checks

For planning, `Phi(s)=-L1_distance`. The previous training reward was `r_task + gamma*Phi(next)-Phi(current)`, with scale 1 and gamma 0.99. Thus **task-reward value = shaped value + Phi(s)** and Hybrid's terminal is `max_a Q_shaped(o(s),a)+Phi(s)`, not raw shaped Q. The reference and native planners are compared numerically/action-wise, and task/shaped recursions differ by only the action-independent root constant. All ten actions are retained. A continuing-task horizon is used even at tick 300.

Saved GIFs contain 301 frames and 20,070 ms each. All frame pixels/timings were programmatically decoded and verified. Actual-player inspection was not performed; this publication requests no Chrome or desktop-control permission.

[Research index](research_index.md) · [Boundary report](../results/tracking_boundary_stability/results_summary.md) · [Planning report](../results/tracking_planning/results_summary.md) · [Release contract and checks](../provenance/latest_studies_release/README.md)

## Current publication adapter and two-cell action study

The newer navigation changes are checked through `python -m aimrl.action_publication boundary --output .tmp/release-boundary` (or `planning`, `history`, `action`; also run under `python -O`). This narrow adapter verifies both archived baseline documents and reviewed current documents, then executes the historical numerical checks without changing their contracts. Receipts go to the supplied fresh directory. See [current study and exact commands](tracking_action_resolution.md#reproduction-and-publication-audits). Frozen historical commands/reports above retain their original context. Fourteen-action training has not been run.
