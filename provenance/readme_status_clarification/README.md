# README research-status clarification

This documentation-only revision is based on publication **80c25e452e38e46ee18a04e6e2fb668bb93d2367**. It changes three bullets under “Limitations and next work” and adds one notice above the historical installation/verification commands. No scientific source, protocol, model, data, frozen report, previous acceptance record or old expected hash is changed. Paper-body editing is outside this revision.

The [revision contract](revision.json) pins the old and current README SHA-256 values and the exact four text substitutions. The audit reconstructs the current README from the old Git object using only those substitutions. The original README must still match the unchanged [previous release contract](../action_resolution_release/release.json). This is a new documentation version, not a replacement of an old expected hash.

## Meaning of the three corrections

1. The early representation study missed the absolute 75% close target. Later planning **Hybrid-H3**, and action-resolution **Plan-H3-A14/Hybrid-H3-A14**, meet C≥75% with hit retention on their own Final holdouts. Their Stress results remain below 75%; the relative joint gates requiring ΔC≥5 percentage points remain unmet. Baselines are not pooled across studies or holdouts. See the frozen [representation report](../../results/tracking_improvement/results_summary.md), [planning report](../../results/tracking_planning/results_summary.md) and [action-resolution report](../../results/tracking_action_resolution/results_summary.md).
2. Representation and reward conditions each use three training seeds; boundary stability uses five. Each Hybrid condition references five distinct frozen models. Each pure Plan condition is one controller, evaluated over many scenes, rather than multiple training replicates. Between-model sample SD and scenario uncertainty are different quantities; the paired bootstrap preserves source-model and scene pairing. See the [representation configuration](../../configs/tracking_research.json), [reward configuration](../../configs/tracking_shooting_tradeoff.json), [boundary configuration](../../configs/tracking_boundary_stability.json) and [action-resolution protocol](../../results/tracking_action_resolution/protocol.json).
3. Boundary stability, short-horizon planning and two-cell action resolution are complete. Course-paper/presentation integration and an independently specified fourteen-action training study remain possible next steps. **Fourteen-action Q-learning has not run.** No claim of actual-player inspection is added.

## Audit this documentation revision

From the current repository root, using the prepared Python environment:

```powershell
python scripts/verify_readme_status.py --output .tmp/readme-status-normal.json
python -O scripts/verify_readme_status.py --output .tmp/readme-status-optimized.json
git diff --check
```

The verifier checks exact edit scope, the unchanged old contract, every other baseline Git blob, consistency with saved configuration/result/gate records, and local relative links/anchors. It runs no training, environment rollout, numerical experiment suite or historical acceptance suite. The [checks record](checks.json) lists checks actually executed for this revision. Previous 112/9/3 test counts and prior experiment acceptance logs are historical evidence, **not newly executed tests** here. Git commit/tree IDs cover this record and its receipt; neither file hashes itself.

## Run hash-bound historical acceptance at its frozen release

The older release adapters intentionally pin their publication README. They are not verification entry points for a later README edit. To use the latest frozen acceptance inputs without editing the current worktree, create a separate detached checkout (requires normal Git history, not only a source ZIP):

```powershell
git worktree add --detach .tmp/readme-status-frozen-80c 80c25e452e38e46ee18a04e6e2fb668bb93d2367
Push-Location .tmp/readme-status-frozen-80c
try {
    python -m aimrl.action_publication inputs --output .tmp/frozen-release-inputs
} finally {
    Pop-Location
}
```

Use the prepared environment with that release's pinned dependencies. The `inputs` command checks frozen hashes/dependencies/links; it is not the full numerical acceptance. Optional full historical acceptance, **not run for this documentation revision**, is `python scripts/check_action_publication.py --compiler zig --output .tmp/frozen-release-checks` from that detached checkout, using Zig 0.15.2 and the Windows build requirements documented in its [study guide](../../docs/tracking_action_resolution.md). Historical study-only direct verifiers can have additional pre-publication assumptions; use the release entry points at the pinned commit. No old contract is patched to accept the current README.
