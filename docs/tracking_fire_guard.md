# Reproduce Q-learning with a firing guard

The [firing guard study](../results/tracking_fire_guard/results_summary.md) loads the 21 final checkpoints from the [reward trade-off models](../results/tracking_shooting_tradeoff/runs). It compares the original policy with a wrapper that substitutes Stay for unaligned Fire. Both variants use the original Python environment. The guard performs zero training updates and uses the existing Q tables. See the [frozen protocol](../results/tracking_fire_guard/protocol.json) and [execution checks](../results/tracking_fire_guard/execution_checks.json).

## Build the parity-check library

Use Python 3.13.9 with `requirements-reproduction.txt` and Zig 0.15.2 on Windows. These are the same pinned dependencies and C11 flags used by the [reward-study build](tracking_shooting_tradeoff.md). From the repository root:

```powershell
$build = '.tmp/guard-kernel-build'
New-Item -ItemType Directory $build -ErrorAction Stop
$kernel = Join-Path $build 'tradeoff.dll'
if ((zig version) -ne '0.15.2') { throw 'Use Zig 0.15.2' }
zig cc -std=c11 -O3 -fno-fast-math -ffp-contract=off -shared aimrl/tracking_tradeoff/kernel.c -o $kernel
if ($LASTEXITCODE) { throw 'Kernel build failed' }
```

The [C source](../aimrl/tracking_tradeoff/kernel.c) SHA-256 is `ec1d977c6b81a038c3599f3eabec660444d539b6cbad3b36f09c50ced2bf3c17`. A fresh build can differ in binary bytes from the original DLL. The gate records both hashes and verifies the rebuilt DLL against Python on all 126,000 read-only parity ticks. Any action, state or event mismatch fails the gate. If using the exact original DLL, omit `--rebuilt-kernel` to require the historical binary hash. New holdout evaluation and GIF rendering call Python directly.

## Run the study

```powershell
$output = '.tmp/fire-guard-replication'
python -m unittest discover -s tests -p 'test_tracking_fire_guard.py' -v
if ($LASTEXITCODE) { throw 'Inference-only tests failed' }
python -m aimrl.tracking_fire_guard --phase setup --output $output
if ($LASTEXITCODE) { throw 'Historical reconciliation or seed audit failed' }
python -m aimrl.tracking_fire_guard --phase gate --output $output --prior-kernel $kernel --rebuilt-kernel
if ($LASTEXITCODE) { throw 'Paired gate or native parity failed' }
python -m aimrl.tracking_fire_guard --phase evaluate --output $output --workers 6
if ($LASTEXITCODE) { throw 'Held-out inference failed' }
python -m aimrl.tracking_fire_guard --phase report --output $output
if ($LASTEXITCODE) { throw 'Report verification failed' }
```

The published source adapter acknowledges exactly one omitted machine-local setup transcript and distinguishes intentional replay seeds from held-out reservations. The [release metadata](../provenance/tracking_studies_release.json) records the omission and exact original and checkout source hashes. The [original source archive](../provenance/tracking_fire_guard_original_sources.zip), old protocol and numerical results preserve the experiment as run. For an evidence-only check, use `python scripts/verify_tracking_studies.py --output .tmp/study-verification`.
