# Archive Integrity and Publication Compatibility

The published result directories describe experiments rather than implementation dates. Original local files remain in the user's original project. `archive_manifest.json` maps historical references to published artifacts, records original SHA-256 hashes, and identifies lossless model packages. It is a path/storage index, not a replacement for frozen experimental evidence.

Numerical CSV/JSON records and original frozen protocols are retained byte for byte. Their historical paths are evidence and are intentionally not rewritten. Active code, documentation and links use the new result paths. The two large continuous-tracking models are ZIP-deflated NPZ packages containing the exact original `q_table.npy` member, not reduced-precision arrays. Both member and archive hashes are recorded; the model loader verifies historical hashes against the member bytes.

`source_compatibility.json` contains only verified mappings required by archived source/history checks. Each mapping binds one historical hash to one exact published file hash and explains the adaptation. Unknown source changes, missing artifacts, or mismatched packages are rejected. No historical hash is replaced with a new hash, and the current source is not claimed to have produced earlier experiments.

Three retired explanatory records that are referenced by one historical resource manifest are verified against their exact Git blobs in the preserved prior commit. They are not copied into the published tree. This verification requires the full Git history; a source-only ZIP is insufficient for that particular historical audit. The registry records the original hash and blob identifier, and the loader recomputes SHA-256 on the retrieved bytes rather than treating a missing file as verified.

`publication_checks.json` records setup, tests, links, saved-model demonstration, GIF frame/duration checks, package integrity, original local-file preservation, and the final publication file-size scan. The publication preserves existing Git history and does not retrain or tune models.
