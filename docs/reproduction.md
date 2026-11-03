# Reproducing the tracking study from a checkout

The supported entry point is `python -m aimrl.reproduce`. It uses the published environment, agent, reward transform, and evaluation functions, with a separate session runner. It creates its own provenance and never requires the original startup snapshot or disposable pilot packages.

The historical commands in `tracking_improvement.md` document the original execution workspace. The original workflow, source bundle, audits, models, and experiment protocols remain unchanged. This entry point does not pretend a new checkout is that original workspace or weaken its audit.

## Install and check

From the cloned repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The recorded experiment environment is Python 3.13.9 on Windows. The original Linux review used Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8, and Pillow 12.3.0. On an activated Linux/macOS virtual environment, use `python` in place of `.\.venv\Scripts\python.exe` below. See the Windows acceptance section for the scope tested on the current version.

## Verify the published evidence

```powershell
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode verify --output .tmp/reproduction/publication_check
.\.venv\Scripts\python.exe -O -m aimrl.reproduce --mode verify --output .tmp/reproduction/publication_check_optimized
```

This delegates to the publication verifier. It checks the published study packages and records, the original source bundle, and the study GIFs. Both modes run the same integrity checks and require the complete repository artifacts. Verification performs no training. Choose a fresh output path for each attempt. A failed check exits nonzero and leaves the session incomplete, without a success report.

## Regenerate the main matched final-test table

```powershell
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode evaluate --methods absolute_base absolute_potential relative_base relative_potential --seeds 10 11 12 --split final --output .tmp/reproduction/matched_final
```

This evaluates the twelve published models plus Rule and Random, each on all 500 final scenarios. The historical 20M references are separate and are not part of this new runner's matched table. Target trajectories are checked for pairing. Models and counters are checksum-verified, and evaluation checks that Q values, update counters, and exploration state remain unchanged.

For a full split, nine primary metrics are compared with the saved reference summaries. A mismatch raises an error and leaves `session.json` marked incomplete. The results include:

| Output | Purpose |
|---|---|
| `session.json` | Configuration, source hashes, protocol hash, runtime, memory semantics, completion or failure |
| `<policy>_windows.csv` | Every scenario's measured outcome |
| `summary.json` | Recomputed metrics and diagnostics |
| `comparison_per_instance.csv` | Main table by method and training seed |
| `cross_seed_summary.json` | Seed means and sample SD; SD is null when only one seed is selected |
| `comparison.png` | A static comparison of the individual policy instances |
| `replay_checks.json` | Pairing, scope, metric differences, and whether reference comparison occurred |

Use `--split stress` for the frozen 200-scenario stress condition. Use `--split development --limit 3` for a small functional check; a limited replay is explicitly marked as partial and is not compared against the full-split mean. It is not a new final-test result.

A smaller full-split check of the two already replayed seed-10 models is:

```powershell
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode evaluate --methods relative_base relative_potential --seeds 10 --output .tmp/reproduction/relative_final
```

Rule and Random are included once unless `--no-baselines` is supplied.

## Run fresh training

Start with a disposable integration fixture:

```powershell
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode train --smoke --methods relative_potential --seeds 99 --output .tmp/reproduction/smoke_train
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode evaluate --model-root .tmp/reproduction/smoke_train --methods relative_potential --seeds 99 --split development --limit 3 --output .tmp/reproduction/smoke_replay
```

The fixture performs exactly 1,001 interactions: three complete 300-tick windows and a 101-tick partial window. It retains the original four-million-step epsilon schedule, so it tests execution rather than effective learning in a short budget. Its output is explicitly labeled as a smoke fixture. It creates a fresh model and counters, validates them after reload, and evaluates development scenarios only during training.

For a full fresh replication of the matched matrix:

```powershell
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode train --methods absolute_base absolute_potential relative_base relative_potential --seeds 10 11 12 --output .tmp/reproduction/fresh_matrix
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode evaluate --model-root .tmp/reproduction/fresh_matrix --methods absolute_base absolute_potential relative_base relative_potential --seeds 10 11 12 --split final --output .tmp/reproduction/fresh_matrix_final
```

The full matrix means 12 fresh runs at 5M interactions each. It is an explicit optional command, not part of installation or verification. This patch was validated with short fixtures and saved-model replays; a new full 60M run has not been performed for this update. An evaluation of a fresh model root without a saved reference summary records that no reference comparison was performed. Compare its generated table with the archived matched results separately.

Each run saves its configuration, actual source hashes, model/counter array hashes, window log, checkpoint validation, and timing. The original exploration streams, action order, tie handling, Q update, and actual-next-state bootstrap are retained. Training starts from zero; no archived Q values are transferred. The runner does not require the historical 20M models or omitted pilot binaries.

An existing destination, including a failed or partial session, is rejected. Choose a new descriptive output suffix for another attempt. Source and evidence directories are protected against output. All new outputs belong outside the archived `results/` tree.

## Diagnostic terminology and platform behavior

The historical `clipped_move_fraction` is `zero_displacement_moves / movement_actions`: it means fully blocked movement. A three-cell action reduced to one cell by a wall is not counted. New replay summaries retain the old field and add the explicit alias `fully_blocked_move_fraction`. They do not invent a total-clipping count or rewrite old records.

Memory readings are best effort: Windows reports peak working set; Linux/macOS use process-lifetime peak resident set with platform-specific unit conversion. Unsupported or failing telemetry records null and an explanation instead of failing the experiment. These are not attributed per-run memory deltas.

## What this changes and why

Publication source adaptations and the original workspace-dependent audit made the legacy stage workflow unsuitable as a fresh-checkout interface. The additive runner separates new execution records from immutable historical evidence. It uses exact documented compatibility mappings for the unchanged published learner, rejects unknown source differences, validates selected model packages, and compares full replays with recorded metrics.

Seven integration tests cover all-four-method short-run parity with the archived training loop, model corruption rejection, evaluation isolation, failure records, output protection, telemetry failure, and an external-working-directory train/replay workflow. The four parity fixtures produced identical Q arrays, count arrays, and training-window CSV bytes for the same method/seed in the two runners. Four publication regression tests run the verifier in both Python modes on an isolated checkout copy. They cover valid evidence, an incorrect recorded protocol digest, a missing required file, and a modified required file. Each failure also checks the checkout runner's exit code, incomplete session, and absence of a success report.

The current paper and actual member-contribution statement are still pending. This software update is a reproduction improvement, not a new research experiment or a completed course submission.

## Windows acceptance

The [original Linux review record](../provenance/reproduction_update_checks.json) covers the original 70-test update. The first Windows acceptance on 2026-10-31 also passed 70 tests after moving the test fixture from the sandbox's denied system temporary directory to the checkout's ignored `.tmp/` directory. The [final Windows acceptance record](../provenance/windows_acceptance.json) records the 2026-11-03 candidate separately, with the measured commands, exit codes, and final file hashes.

On 2026-11-03, setup, all 74 tests, and complete publication verification passed on Windows. The verifier passed in standard Python and with `python -O`. A separate fresh Windows virtual environment passed installation from `requirements-reproduction.txt`, dependency and setup checks, and a limited development replay. The limited replay used three development scenarios with Relative Potential seed 10, Rule, and Random. It records functional execution only.

Relative Base and Relative Potential (seed 10), Rule, and Random were replayed on all 500 final scenarios each, for 2,000 windows and 600,000 evaluation ticks. All 36 comparisons across the nine checked metrics matched the archived summaries exactly. Full twelve-model final/stress replay, formal 60M retraining, other platforms for this update, and native GUI playback remain unverified in this acceptance.
