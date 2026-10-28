"""Resolve published artifact paths and read lossless NPY-in-NPZ model packages."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def archive_manifest(root):
    path = Path(root) / 'provenance/archive_manifest.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def resolve_artifact(reference, root=ROOT):
    root = Path(root).resolve()
    manifest = archive_manifest(root)
    value = str(reference).replace('\\', '/')
    prefix = root.as_posix().rstrip('/') + '/'
    original = manifest.get('source_root', '').replace('\\', '/').rstrip('/') + '/'
    if value.startswith(prefix): key = value[len(prefix):]
    elif original != '/' and value.startswith(original): key = value[len(original):]
    elif Path(value).is_absolute(): return Path(value)
    else: key = value
    entry = manifest.get('artifacts', {}).get(key)
    if entry:
        path = root / entry['published_path']
    else:
        for old, new in manifest.get('path_prefixes', {}).items():
            if key == old or key.startswith(old + '/'):
                key = new + key[len(old):]; break
        path = root / key
    if not path.exists() and path.suffix == '.npy' and path.with_suffix('.npz').exists():
        return path.with_suffix('.npz')
    return path


def artifact_digest(reference, root=ROOT):
    """Hash original NPY bytes inside packages, preserving historical model hashes."""
    path = resolve_artifact(reference, root)
    h = hashlib.sha256()
    if path.suffix == '.npz':
        with zipfile.ZipFile(path) as archive, archive.open('q_table.npy') as file:
            for chunk in iter(lambda: file.read(1048576), b''): h.update(chunk)
    else:
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def load_model_array(reference, root=ROOT):
    path = resolve_artifact(reference, root)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as package:
            if package.files != ['q_table']:
                raise ValueError('Expected a lossless model package containing q_table.npy')
            return package['q_table']
    return np.load(path, mmap_mode='r', allow_pickle=False)
