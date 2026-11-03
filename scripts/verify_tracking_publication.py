"""Verify the published study without training, rewriting evidence, or requiring omitted pilots."""
import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from aimrl.provenance import source_hash_matches
from aimrl.tracking_research.agent import array_hash
from aimrl.tracking_research.audit import validate_summary, validate_windows
from aimrl.tracking_research.io import read_csv, read_json, sha
from aimrl.tracking_research.workflow import METHODS, pair_rows, seeds


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_digest(path, expected):
    require(sha(path) == expected, f'SHA-256 mismatch: {path.relative_to(ROOT)}')


def verify():
    manifest = read_json(ROOT / 'provenance/tracking_improvement_publication.json')
    study = ROOT / 'results/tracking_improvement'
    protocol = read_json(study / 'protocol.json')
    config = read_json(ROOT / 'configs/tracking_research.json')
    audit = read_json(study / 'execution_checks.json')
    require(audit['complete'] and protocol['formal_frozen'] and protocol['config'] == config,
            'Expected a complete audit and matching frozen study configuration')
    verify_digest(study / 'protocol.json', manifest['study_protocol_sha256'])
    verify_digest(study / 'execution_checks.json', manifest['recorded_audit_sha256'])
    exclusions = manifest['intentionally_excluded']
    for name, entry in manifest['copied_files'].items():
        verify_digest(ROOT / name, entry['sha256'])
        require((ROOT / name).stat().st_size == entry['bytes'], f'File size mismatch: {name}')
    for name, digest in audit['study_files_sha256'].items():
        relative = 'results/tracking_improvement/' + name
        if relative in exclusions:
            require(exclusions[relative]['sha256'] == digest and not (ROOT / relative).exists(),
                    f'Pilot exclusion mismatch: {relative}')
        else:
            verify_digest(study / name, digest)
    require(len(exclusions) == 8, 'Expected eight pilot exclusions')
    require(all('/pilots/' in name and name.endswith(('q_table.npz', 'visitation.npz')) for name in exclusions),
            'Unexpected pilot exclusion path')
    require(not any((study / 'pilots').rglob('*.npz')), 'Unexpected published pilot archive')
    bundle = manifest['training_source_bundle']
    verify_digest(ROOT / bundle['path'], bundle['sha256'])
    require(bundle['members_sha256'] == protocol['source_sha256'], 'Source bundle manifest differs from frozen protocol')
    source_differences = []
    with zipfile.ZipFile(ROOT / bundle['path']) as z:
        require(set(z.namelist()) == set(protocol['source_sha256']), 'Source bundle member list mismatch')
        for name, expected in protocol['source_sha256'].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == expected, f'Source bundle member SHA-256 mismatch: {name}')
            require(source_hash_matches(ROOT, name, expected), f'Unverified checkout source: {name}')
            if sha(ROOT / name) != expected: source_differences.append(name)
    # Existing compatibility mappings describe the earlier path/NPZ publication adapters.
    # No original protocol hash is rewritten and no additional hash is accepted here.
    training_pairing = {s: {} for s in (10, 11, 12)}
    development_pairing = {}
    run_count = total_updates = development_windows = development_summaries = 0
    for method in METHODS:
        for seed in (10, 11, 12):
            run = study / 'runs' / method / f'seed_{seed}'
            label = f'{method} seed {seed}'
            meta = read_json(run / 'metadata.json')
            require((meta['method'], meta['training_seed'], meta['interactions']) == (method, seed, 5000000),
                    f'Model identity or budget mismatch: {label}')
            require(meta['source_sha256'] == protocol['source_sha256'], f'Model source hash mismatch: {label}')
            require(read_json(run / 'run_config.json') == config, f'Model configuration mismatch: {label}')
            for name, digest in meta['files_sha256'].items():
                verify_digest(run / name, digest)
            with np.load(run / 'q_table.npz', allow_pickle=False) as z:
                q = z['q_table']
                require(str(z['representation']) == method.split('_')[0], f'Q table representation mismatch: {label}')
                require(str(z['indexing_version']) == meta['indexing_version'], f'Q table indexing mismatch: {label}')
                require(list(q.shape) == meta['q_shape'] and q.dtype == np.float32 and np.isfinite(q).all(),
                        f'Invalid Q table: {label}')
                require(array_hash(q) == meta['q_array_sha256'], f'Q array SHA-256 mismatch: {label}')
            del q
            with np.load(run / 'visitation.npz', allow_pickle=False) as z:
                counts = z['counts']
                require(counts.dtype == np.uint32 and list(counts.shape) == meta['q_shape'],
                        f'Invalid visitation array: {label}')
                require(int(counts.sum(dtype=np.uint64)) == 5000000, f'Visitation update count mismatch: {label}')
                require(array_hash(counts) == meta['counts_array_sha256'], f'Visitation array SHA-256 mismatch: {label}')
                require(int(np.count_nonzero(counts)) == meta['coverage']['visited_pairs'],
                        f'Visited pair count mismatch: {label}')
                require(int(np.count_nonzero(counts.sum(axis=-1, dtype=np.uint64))) == meta['coverage']['visited_rows'],
                        f'Visited row count mismatch: {label}')
            del counts
            total = full = partial = 0
            for row in read_csv(run / 'training_windows.csv'):
                n = int(row['steps'])
                total += n
                window = f'{label} at step {total}'
                require(int(row['end_step']) == total and row['terminated'] == 'False',
                        f'Invalid training window: {window}')
                if row['partial'] == 'True':
                    partial += 1
                    require(n == 200 and row['truncated'] == 'False', f'Invalid partial window: {window}')
                else:
                    full += 1
                    require(n == 300 and row['truncated'] == 'True', f'Invalid full window: {window}')
                hits, misses = int(row['hit_count']), int(row['miss_count'])
                base = float(row['base_return'])
                require(int(row['shots']) == hits + misses, f'Training shot count mismatch: {window}')
                require(math.isclose(base, 10*hits-misses-.05*n, abs_tol=1e-7), f'Base return mismatch: {window}')
                require(math.isclose(float(row['learning_return']), base+float(row['shaping_return']), abs_tol=1e-7),
                        f'Shaped return mismatch: {window}')
                digest = row['target_trajectory_sha256']
                require(training_pairing[seed].setdefault(total, digest) == digest,
                        f'Training target paths not paired: {window}')
            require((total, full, partial) == (5000000, 16666, 1), f'Training window totals mismatch: {label}')
            validation = read_json(run / 'validation.json')
            require([x['global_step'] for x in validation] == list(range(0, 5000001, 500000)),
                    f'Unexpected development checkpoints: {label}')
            for item in validation:
                folder = study / 'evaluation/development' / method / f'seed_{seed}'
                t = item['global_step']
                require(read_json(folder / f'checkpoint_{t:07d}.json') == item,
                        f'Development checkpoint summary mismatch: {label} at step {t}')
                rows = read_csv(folder / f'windows_{t:07d}.csv')
                validate_windows(rows, seeds(config, 'development')); validate_summary(item['metrics'])
                pair_rows(development_pairing, rows)
                development_windows += len(rows); development_summaries += 1
            total_updates += total; run_count += 1
    require((run_count, total_updates, development_summaries, development_windows) == (12,60000000,132,26400),
            'Formal matrix or development totals mismatch')
    splits = {}
    for split, expected in [('final',8000), ('stress',3200)]:
        folder = study / 'evaluation' / split
        recorded = read_json(folder / 'checks.json')
        for name, digest in recorded['files_sha256'].items():
            verify_digest(folder / name, digest)
        summaries = read_json(folder / 'summary.json'); pairing = {}; windows = 0
        require(len(summaries) == 16, f'Unexpected number of policies: {split}')
        for key, summary in summaries.items():
            rows = read_csv(folder / f'{key}_windows.csv')
            validate_windows(rows, seeds(config, split)); validate_summary(summary); pair_rows(pairing, rows)
            windows += len(rows)
        require(windows == expected, f'Unexpected window count: {split}')
        splits[split] = {'windows':windows, 'paired_scenarios':len(pairing)}
    # Independently recompute reported seed means and sample SDs from the saved final/stress summaries.
    comparison = read_json(study / 'evaluation/cross_seed_summary.json')
    for split in ('final', 'stress'):
        data = read_json(study / 'evaluation' / split / 'summary.json')
        for method, fields in comparison['groups'][split].items():
            for field, expected in fields.items():
                values = [data[f'{method}_seed{s}'][field] for s in (10,11,12)]
                label = f'{split}/{method}/{field}'
                require(math.isclose(float(np.mean(values)), expected['mean'], abs_tol=1e-12),
                        f'Cross-seed mean mismatch: {label}')
                require(math.isclose(float(np.std(values, ddof=1)), expected['sample_sd'], abs_tol=1e-12),
                        f'Cross-seed sample SD mismatch: {label}')
    gifs = []
    for clip in audit['clips']:
        path = ROOT / clip['path']; durations = []
        verify_digest(path, clip['sha256'])
        with Image.open(path) as gif:
            require(gif.n_frames == 301, f'Unexpected GIF frame count: {clip["path"]}')
            for i in range(gif.n_frames):
                gif.seek(i); gif.load(); durations.append(gif.info.get('duration',0))
        require(sum(durations) == 20070 and set(durations) == {60,70}, f'Unexpected GIF timing: {clip["path"]}')
        gifs.append({'path':clip['path'], 'sha256':clip['sha256'], 'frames':301, 'duration_ms':20070})
    require(len(gifs) == 4, 'Expected four study GIFs')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    first = 'results/tracking_improvement/render/seed3601000_main_comparison.gif'
    require(re.search(r'!\[[^\]]*\]\('+re.escape(first)+r'\)',readme[:4500]), 'README is missing its opening study GIF')
    require(all(clip['path'] in readme for clip in gifs), 'README is missing a study GIF link')
    links = 0
    for path in [ROOT/'README.md', ROOT/'docs/tracking_improvement.md', study/'results_summary.md', ROOT/'provenance/tracking_improvement_publication.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:','#','mailto:')): continue
            require((path.parent / target.split('#')[0]).exists(),
                    f'Broken link in {path.relative_to(ROOT)}: {target}')
            links += 1
    files = [p for p in study.rglob('*') if p.is_file()]
    for path in files:
        require(path.stat().st_size < 100*1024**2, f'File exceeds publication size limit: {path.relative_to(ROOT)}')
        require(path.suffix != '.npy', f'Unexpected uncompressed model: {path.relative_to(ROOT)}')
        if path.suffix in ('.json','.md','.csv','.txt'):
            require(not re.search(r'[\u3400-\u9fff]',path.read_text(encoding='utf-8')),
                    f'Unexpected non-English publication text: {path.relative_to(ROOT)}')
    return {'complete':True, 'copied_files_verified':len(manifest['copied_files']), 'formal_runs':run_count,
            'formal_interactions':total_updates, 'development_summaries':development_summaries,
            'development_windows':development_windows, 'development_ticks':development_windows*300,
            'independent_splits':splits, 'gifs':gifs, 'relative_links_verified':links,
            'largest_study_file_bytes':max(p.stat().st_size for p in files), 'pilot_binaries_intentionally_excluded':len(exclusions),
            'original_training_source_bundle_verified':True, 'previously_documented_source_differences':source_differences,
            'recorded_audit_unchanged':True, 'training_or_result_regeneration':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/tracking_publication_verification.json')
    args = parser.parse_args()
    path = ROOT / args.output
    try:
        if path.exists():
            raise FileExistsError(f'Choose a fresh verification output: {args.output}')
        result = json.dumps(verify(), indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(result + '\n')
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Publication verification failed: {exc}\n')
    print(result)


if __name__ == '__main__': main()
