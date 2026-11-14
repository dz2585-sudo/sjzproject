# Reproduce and verify the planning study

Run from repository root with requirements-reproduction.txt installed. Runtime and compiler versions are frozen in protocol.json and native/build.json. Use the existing ten step-5M models; never rerun previous training. No code in this study performs a Q update.

Verify saved records (ordinary and optimized Python), without replaying formal evaluation:

```powershell
python -m aimrl.tracking_planning verify
python -O -m aimrl.tracking_planning verify
```

Fresh replication: choose an unused directory, establish a new protection inventory for all tracked historical paths plus the complete previous uncommitted source/models/results, then run the following. The one-time inventory bootstrap is documented in workspace.json/protected_sha256_before.json; the default delivered directory must not be overwritten. The exact replication intentionally reuses this study reservation. Setup audits every protected seed declaration/CSV.

```powershell
$env:AIMRL_PLANNING_OUTPUT = 'results/tracking_planning_replication'
python -m aimrl.tracking_planning initialize
python -m aimrl.tracking_planning setup --compiler <path-to-zig.exe>
python -m aimrl.tracking_planning development
python scripts/check_tracking_planning.py preflight --legacy-kernel <path-to-published-tradeoff.dll>
python -m aimrl.tracking_planning freeze
python -m aimrl.tracking_planning evaluate
python -m aimrl.tracking_planning report
python scripts/check_tracking_planning.py postflight --legacy-kernel <path-to-published-tradeoff.dll>
python -m aimrl.tracking_planning inventory
python -m aimrl.tracking_planning verify
python -O -m aimrl.tracking_planning verify
```

Setup compiles without fast-math or floating-point contraction, precomputes all 21 horizon layers across seven terminal families, and records genuine loop counters, memory and wall time. Development completes all 20×200 windows and a full 301-frame layout test before protocol freeze. Freeze requires all preflight commands, records source/config/models/caches/methods/seed ranges and retains all methods. Final and Stress use 20×500 and 20×200 independent environment windows. C uses frozen policy tables only; it cannot supply a scenario seed or true future to any policy query.

Formal rollouts and native decision timings are retained in evaluation/*/*_rollouts.npz. Report and verify reuse these records; no formal rollout is needed for recomputation. Cached Bellman arrays are checked by rebuilding mathematical DP sweeps (planning computations, not training interactions) and comparing every saved value/policy/expectation. Existing historical checkers write only fresh receipts inside this study. The previous canonical boundary verifier is wrapped solely to redirect its two receipt writes; previous expected hashes and sources remain unchanged.

Manifest excludes its own file and explicit verification/postflight receipts to avoid circular digests. Source hashes, model hashes, DP cache hashes and protected-input hashes are always checked separately. No exclusion may be used to hide a changed frozen input. GIF decoder checks all pixels/durations and sample frame hashes. Actual-player inspection is intentionally not performed by this protocol.

## Optional measured interpretation supplement

After report generation, run `python -m aimrl.tracking_planning.post_analysis` to reproduce behavior_interpretation.md and its links. This descriptive post-evaluation module changes no primary result or frozen source. Native zero-nanosecond timing medians are resolution-limited; see its timer-resolution and cold-preparation discussion.

## Final acceptance

All 96 complete-suite tests and all seven new optimized-mode tests passed. Historical publication, reward/guard-study and previous-boundary canonical verifiers passed in ordinary and optimized Python, with old receipt writes redirected outside protected files. Both new saved-artifact verifiers passed, including exhaustive reconstruction of all 21 DP layers and comparison of 5.4 million stored policy proposals. Formal environment evaluations were not replayed. All 2,972 protected historical and previously uncommitted files remain unchanged. See final_acceptance.json and execution/. The delivery manifest is checked separately after creation; receipts are excluded from their own hashes. Actual-player inspection was intentionally not executed by this protocol.
