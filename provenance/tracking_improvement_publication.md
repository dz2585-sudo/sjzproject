# Tracking representation study: publication provenance

The completed local study is published under results/tracking_improvement/. Its protocol, original execution audit, per-seed models/counters, raw training/evaluation records, summaries, figures, and four full-length GIFs retain their exact original bytes. The original execution audit describes the local execution at that time, including its then-unchanged Git identity and lack of remote publication; it is not a claim about this later publication operation.

[Publication manifest](tracking_improvement_publication.json) lists every copied file's SHA-256 and size, the frozen protocol/audit hashes, and eight intentional omissions: the four disposable pilot Q packages and four pilot visitation packages. They remain in the original local project. Their hashes and all pilot text records remain available. Missing pilot arrays are explicitly reported as omissions, never silently counted as remotely verified model binaries. All twelve formal models and twelve formal visitation packages are published without re-encoding.

[Exact training sources](tracking_improvement_training_sources.zip) contains all 20 files named in the formal protocol's source hash dictionary. Every member is verified against that original dictionary. Three already-published V2 files have earlier path/NPZ loader adaptations; their existing exact mappings in source_compatibility.json are unchanged. The new research implementation itself is copied byte-for-byte. No frozen source hash is replaced or retrospectively described as having produced the original experiment.

All prior source-compatibility entries are retained. One additional exact README-only entry documents this later publication's navigation and study summary changes; the original README hash is verified against its preserved historical Git blob. Frozen protocols and executable-source acceptance entries are unchanged.

The original stage commands in the study documentation describe its original execution workspace. In particular, the original full audit checks the original root's Git identity/HEAD, temporary startup snapshot, historical filesystem layout, and pilot arrays. Those checks are intentionally not weakened to make a different checkout appear to be that original workspace. Use the separate read-only publication verifier below to inspect this checkout. It verifies archived original sources, every published file, formal models/counters and budgets, paired record integrity, cross-seed statistics, GIF timing, and document links without retraining or overwriting the recorded research audit.

Run from the repository root with an installed project environment:

```powershell
python scripts/check_setup.py
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/verify_tracking_publication.py
```

The verifier's default output is ignored under .tmp/. Publication-time logs are saved separately from original experiment logs. The source project and all historical result directories remain intact. README changes add the new study with means/sample standard deviations, explain the boundary-information limitation and shooting/proximity tradeoff, and retain the distinct historical 20M results.

No models were retrained, budgets extended, final outcomes changed, or figures/GIFs regenerated for publication. All files fit ordinary repository storage; release assets are unnecessary. See the [report](../results/tracking_improvement/results_summary.md), [original research audit](../results/tracking_improvement/execution_checks.json), and [cross-seed summary](../results/tracking_improvement/evaluation/cross_seed_summary.json).
