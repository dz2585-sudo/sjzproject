"""Validate exact post-experiment translation hashes without changing frozen records."""
import hashlib
import json
from pathlib import Path


def source_hash_matches(root, name, recorded_hash):
    """Accept original bytes or an explicitly recorded translation of those bytes."""
    root = Path(root)
    path = root / name
    if not path.is_file():
        return False
    current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if current_hash == recorded_hash:
        return True
    manifest = root / 'translation' / 'manifest.json'
    if not manifest.is_file():
        return False
    data = json.loads(manifest.read_text(encoding='utf-8'))
    entry = data.get('translated_files', {}).get(name, {})
    return (entry.get('original_sha256') == recorded_hash
            and entry.get('translated_sha256') == current_hash)
