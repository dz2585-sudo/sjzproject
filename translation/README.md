# English Translation and Experimental Provenance

The English translation was made after the Day 1–5 experiments and the initial GitHub upload, whose commit is `d89c7db4f90fcbee45e9437f945485352f63a642`. It translates documentation, saved explanatory summaries, and text emitted by the comparison and experiment-summary generators. It does not change environment rules, actions, rewards, learning updates, configurations, random seeds, models, numerical records, or experimental conclusions.

The original experimental code is available in that earlier Git commit. The translated files are not claimed to have produced the recorded experiments. Original SHA-256 values in the Day 4 frozen protocol, Day 5 source hashes, run manifest, and model metadata remain unchanged. The original protocol and run records also retain their historical absolute paths and timestamps.

`manifest.json` records original and translated file hashes, the preceding experimental commit, and the invariant files checked during translation. The changed source files are `aimrl/compare.py` and `aimrl/experiments.py`. Their report-writing strings were translated. The latter also imports a provenance checker and uses it for exact recorded source-hash validation. Its future-run metadata now describes the translation instead of claiming that all frozen source files are unchanged.

`aimrl/provenance.py` accepts either the original recorded bytes or an exact original-to-translated hash mapping in the translation manifest. It rejects missing files, unrecorded changes, and mappings for a different original hash. This is an explicit compatibility adaptation for reproducibility after translation; it does not replace frozen hashes or modify training or evaluation behavior. Original protocol checks otherwise remain in place. Additional tests cover this validation behavior.

All existing figures already use English labels, titles, axes, and legends. They were retained byte for byte. No report directory or filenames in the upload scope required renaming. All uploaded explanatory files were scanned for remaining Chinese text, and local Markdown links were checked in the prepared repository.

No archived experiment was retrained or tuned. The requested existing unit tests include short isolated training runs in temporary directories; they do not replace any archived model or experimental output. Setup and unit-test results, file scans, immutable-artifact checks, and source compatibility checks are recorded in `verification.json`.
