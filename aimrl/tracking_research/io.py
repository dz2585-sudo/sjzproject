"""Study records, immutable historical packages, and operating-system resource readings."""
import csv
import ctypes
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''): h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    rows = list(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as file: return list(csv.DictReader(file))


def peak_memory_bytes():
    """Windows process-lifetime peak working set, not an attributed per-stage delta."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [(x, ctypes.c_size_t) for x in
            ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    data = Counters(); data.cb = ctypes.sizeof(data)
    handle = ctypes.windll.kernel32.GetCurrentProcess
    handle.restype = ctypes.c_void_p
    function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not function(handle(), ctypes.byref(data), data.cb): raise ctypes.WinError()
    return int(data.PeakWorkingSetSize)


def git_state():
    result = {}
    for label, args in [('head', ['rev-parse', 'HEAD']), ('status', ['status', '--short']),
                        ('identity', ['config', '--show-origin', '--get-regexp', r'^(user\.|credential\..*username)'])]:
        p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        result[label] = {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    return result


def archive_location():
    for path in (ROOT / 'results/continuous_tracking', ROOT / '.tmp/sjz-publication/results/continuous_tracking',
                 ROOT / 'outputs/tracking25_v2'):
        if (path / 'base/metadata.json').exists(): return path
    raise FileNotFoundError('No historical continuous-tracking archive found')


def historical_agent(config, mode):
    from .agent import ResearchAgent
    archive = archive_location(); directory = archive / mode
    meta = read_json(directory / 'metadata.json')
    original_config = read_json(ROOT / 'configs/tracking25_v2.json')
    run_config = read_json(directory / 'run_config.json')
    if run_config != original_config or meta['global_steps'] != 20000000 or meta['seed'] != 0 or meta['reward_mode'] != mode:
        raise ValueError('Historical model configuration or budget mismatch')
    for key in ('environment', 'actions', 'rewards'):
        if config[key] != run_config[key]: raise ValueError('Historical physics mismatch')
    path = directory / 'q_table.npz'
    if path.exists():
        h = hashlib.sha256()
        with zipfile.ZipFile(path) as z, z.open('q_table.npy') as file:
            for block in iter(lambda: file.read(1048576), b''): h.update(block)
        with np.load(path, allow_pickle=False) as z: data = z['q_table']
    else:
        path = directory / 'q_table.npy'; h = None
        data = np.load(path, mmap_mode='r', allow_pickle=False)
    digest = h.hexdigest() if h is not None else sha(path)
    if digest != meta['model_sha256']: raise ValueError('Historical original NPY hash mismatch')
    # Verify original frozen source bytes at the original root, not translated publication bytes.
    for name, expected in meta['source_sha256'].items():
        if sha(ROOT / name) != expected: raise ValueError('Historical frozen source changed: ' + name)
    agent = ResearchAgent(config, 'absolute', count_updates=False)
    if data.shape != agent.shape or data.dtype != np.float32 or not np.isfinite(data).all():
        raise ValueError('Invalid historical model table')
    agent.q[:] = data
    return agent, {'path': str(path), 'original_npy_sha256': digest, 'package_sha256': sha(path),
                   'source_hashes_verified': True, 'training_steps': 20000000, 'training_seed': 0}
