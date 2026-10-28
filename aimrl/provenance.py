"""Verify exact archived or explicitly documented compatible publication bytes."""
import hashlib
import json
import subprocess
from pathlib import Path

from .model_io import artifact_digest, resolve_artifact


def source_hash_matches(root, name, recorded_hash):
    root = Path(root)
    path = resolve_artifact(name, root)
    if not path.is_file():
        return False
    current = artifact_digest(path, root)
    if current == recorded_hash:
        return True
    registry = root / 'provenance/source_compatibility.json'
    if not registry.is_file():
        return False
    records = json.loads(registry.read_text(encoding='utf-8')).get('mappings', [])
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return any(r['path'] == relative and r['recorded_sha256'] == recorded_hash and r['current_sha256'] == current for r in records)


def verify_historical_hashes(root, records):
    """Verify current artifacts or exact retired evidence blobs preserved in Git history."""
    root = Path(root)
    registry = root / 'provenance/source_compatibility.json'
    retired = json.loads(registry.read_text(encoding='utf-8')).get('retired_evidence', []) if registry.is_file() else []
    current_count, historical_count = 0, 0
    for name, expected in records.items():
        if source_hash_matches(root, name, expected):
            current_count += 1
            continue
        entry = next((r for r in retired if r['historical_path'] == name and r['recorded_sha256'] == expected), None)
        if entry is None:
            raise ValueError('Unverified historical artifact: ' + name)
        result = subprocess.run(['git', 'cat-file', 'blob', entry['git_blob']], cwd=root, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            raise ValueError('Retired evidence cannot be verified; clone the full repository history: ' + name)
        historical_count += 1
    return {'verified_current_or_compatible': current_count, 'verified_retired_git_blobs': historical_count}
