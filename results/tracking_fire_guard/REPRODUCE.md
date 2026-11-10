# Reproduce the inference-only firing guard study

Run these commands from the AimRL publication checkout using the existing Python environment with the repository's pinned dependencies. The prior result directory and the frozen models must be present. Choose a fresh output directory for each complete attempt.

```powershell
$output = 'results/tracking_fire_guard_replication'
$kernel = '../tradeoff-build/tradeoff.dll'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m unittest discover -s tests -p 'test_tracking_fire_guard.py' -v
if ($LASTEXITCODE) { throw 'Inference-only tests failed' }
python -m aimrl.tracking_fire_guard --phase setup --output $output
if ($LASTEXITCODE) { throw 'Historical reconciliation or seed audit failed' }
python -m aimrl.tracking_fire_guard --phase gate --output $output --prior-kernel $kernel
if ($LASTEXITCODE) { throw 'Paired gate or native parity failed' }
python -m aimrl.tracking_fire_guard --phase evaluate --output $output --workers 6
if ($LASTEXITCODE) { throw 'Held-out inference failed' }
python -m aimrl.tracking_fire_guard --phase report --output $output
if ($LASTEXITCODE) { throw 'Report verification failed' }
```

The gate uses the original native library whose SHA-256 is in the previous study protocol. The relative library path above is the retained workspace build location. Use the exact archived binary for historical parity. New holdout evaluation and the demo call the original Python environment and do not require the native library. The phase records retain commands, duration and exit code. The focused tests and all four study phases perform no training.
