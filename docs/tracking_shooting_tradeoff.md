# Tracking and shooting reward study

The study compares seven scalar rewards under the same target motion, relative observation, ten-action controller and Q-learning budget. See the [frozen protocol](../results/tracking_shooting_tradeoff/protocol.json), [English results](../results/tracking_shooting_tradeoff/results_summary.md), and [paper addendum](../results/tracking_shooting_tradeoff/paper_addendum.md).

## Reproduce on Windows

Use Python 3.13.9 and install `requirements-reproduction.txt` as in the [reproduction guide](reproduction.md). The verified Windows runtime used NumPy 2.5.4, Matplotlib 3.11.2, and Pillow 13.0.0. The native C11 compiler is Zig 0.15.2. The rollout uses NumPy's PCG64 callbacks, strict floating-point operations, and the existing Python resets. The frozen [kernel source](../aimrl/tracking_tradeoff/kernel.c) SHA-256 is `ec1d977c6b81a038c3599f3eabec660444d539b6cbad3b36f09c50ced2bf3c17`.

Training parity was tested against the Python environment and learner for 1,001 exact zero-weight updates and all seven weights around the greedy end of the epsilon schedule. The tests matched actions, physical states, float32 Q values, update counts and RNG states. This is a bounded training parity check. The firing-guard study separately checked 126,000 read-only inference ticks against Python with the original DLL. Its [checkout guide](tracking_fire_guard.md) explains how to repeat the read-only check with a rebuilt DLL. See the [training evidence](../results/tracking_shooting_tradeoff/execution_checks.json) and [release metadata](../provenance/tracking_studies_release.json).

From the publication checkout, with a Python interpreter and Zig 0.15.2 on PATH:

```powershell
$build = '.tmp/tradeoff-rebuild'
$output = 'results/tracking_shooting_tradeoff_replication'
New-Item -ItemType Directory $build -ErrorAction Stop
$kernel = Join-Path $build 'tradeoff.dll'
if ((zig version) -ne '0.15.2') { throw 'Use Zig 0.15.2' }
zig cc -std=c11 -O3 -fno-fast-math -ffp-contract=off -shared aimrl/tracking_tradeoff/kernel.c -o $kernel
if ($LASTEXITCODE) { throw 'Kernel build failed' }
$env:AIMRL_TRADEOFF_KERNEL = (Resolve-Path $kernel).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
python scripts/check_setup.py
if ($LASTEXITCODE) { throw 'Setup failed' }
python -m aimrl.tracking_tradeoff --phase smoke --output $output --kernel $kernel
if ($LASTEXITCODE) { throw 'Smoke failed' }
python -m unittest discover -s tests -p 'test_*.py' -v
if ($LASTEXITCODE) { throw 'Tests failed' }
$seedAudit = python scripts/check_tracking_tradeoff_seeds.py
if ($LASTEXITCODE) { throw 'Seed collision or audit failure' }
@{setup_exit_code=0; tests_exit_code=0; seed_search_exit_code=0; seed_search=($seedAudit | ConvertFrom-Json)} |
    ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $output 'preflight.json')
foreach ($phase in @('train','evaluate','report')) {
    python -m aimrl.tracking_tradeoff --phase $phase --output $output --kernel $kernel
    if ($LASTEXITCODE) { throw "$phase failed" }
}
```

The seed auditor accepts `--original` to also check the original experiment workspace when available. The training phase freezes the config, source and historical hashes before the first formal interaction. A run records 11 checkpoints, including zero and the final budget. Every development evaluation uses a separate environment. The evaluation phase requires the completed 21-run matrix and its development-only display selection.

Choose a new output directory for a new attempt. The smoke phase rejects existing directories and later phases refuse a previously attempted phase. The reports describe the seven fixed weights and compare common outcome metrics. Checkpoint packages are compressed NumPy files with Q values, update counts, condition, weights, config and indexing metadata. Outputs from separate attempts must keep their own provenance.

The commands through the tests and seed audit are sufficient for a bounded local check. The `train` phase is the optional 105-million-update replication. For an evidence-only release check, run `python scripts/verify_tracking_studies.py --output .tmp/study-verification`. The [release metadata](../provenance/tracking_studies_release.json) records the one omitted machine-local setup transcript and links its [redacted copy](../provenance/tracking_tradeoff_setup_public.txt). The frozen inventory retains the original transcript hash.
