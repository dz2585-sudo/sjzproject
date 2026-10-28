"""Verify the published study without training, rewriting evidence, or requiring omitted pilots."""
import argparse
import hashlib
import json
import math
import re
import subprocess
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


def verify():
    manifest = read_json(ROOT / 'provenance/tracking_improvement_publication.json')
    study = ROOT / 'results/tracking_improvement'
    protocol = read_json(study / 'protocol.json')
    config = read_json(ROOT / 'configs/tracking_research.json')
    audit = read_json(study / 'execution_checks.json')
    assert audit['complete'] and protocol['formal_frozen'] and protocol['config'] == config
    assert sha(study / 'protocol.json') == manifest['study_protocol_sha256']
    assert sha(study / 'execution_checks.json') == manifest['recorded_audit_sha256']
    exclusions = manifest['intentionally_excluded']
    for name, entry in manifest['copied_files'].items():
        assert sha(ROOT / name) == entry['sha256'], name
        assert (ROOT / name).stat().st_size == entry['bytes'], name
    for name, digest in audit['study_files_sha256'].items():
        relative = 'results/tracking_improvement/' + name
        if relative in exclusions:
            assert exclusions[relative]['sha256'] == digest and not (ROOT / relative).exists()
        else:
            assert sha(study / name) == digest, name
    assert len(exclusions) == 8
    assert all('/pilots/' in name and name.endswith(('q_table.npz', 'visitation.npz')) for name in exclusions)
    assert not any((study / 'pilots').rglob('*.npz'))
    bundle = manifest['training_source_bundle']
    assert sha(ROOT / bundle['path']) == bundle['sha256']
    assert bundle['members_sha256'] == protocol['source_sha256']
    source_differences = []
    with zipfile.ZipFile(ROOT / bundle['path']) as z:
        assert set(z.namelist()) == set(protocol['source_sha256'])
        for name, expected in protocol['source_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
            assert source_hash_matches(ROOT, name, expected), name
            if sha(ROOT / name) != expected: source_differences.append(name)
    # Existing compatibility mappings describe the earlier path/NPZ publication adapters.
    # No original protocol hash is rewritten and no additional hash is accepted here.
    training_pairing = {s: {} for s in (10, 11, 12)}
    development_pairing = {}
    run_count = total_updates = development_windows = development_summaries = 0
    for method in METHODS:
        for seed in (10, 11, 12):
            run = study / 'runs' / method / f'seed_{seed}'
            meta = read_json(run / 'metadata.json')
            assert (meta['method'], meta['training_seed'], meta['interactions']) == (method, seed, 5000000)
            assert meta['source_sha256'] == protocol['source_sha256']
            assert read_json(run / 'run_config.json') == config
            for name, digest in meta['files_sha256'].items(): assert sha(run / name) == digest
            with np.load(run / 'q_table.npz', allow_pickle=False) as z:
                q = z['q_table']
                assert str(z['representation']) == method.split('_')[0]
                assert str(z['indexing_version']) == meta['indexing_version']
                assert list(q.shape) == meta['q_shape'] and q.dtype == np.float32 and np.isfinite(q).all()
                assert array_hash(q) == meta['q_array_sha256']
            del q
            with np.load(run / 'visitation.npz', allow_pickle=False) as z:
                counts = z['counts']
                assert counts.dtype == np.uint32 and list(counts.shape) == meta['q_shape']
                assert int(counts.sum(dtype=np.uint64)) == 5000000
                assert array_hash(counts) == meta['counts_array_sha256']
                assert int(np.count_nonzero(counts)) == meta['coverage']['visited_pairs']
                assert int(np.count_nonzero(counts.sum(axis=-1, dtype=np.uint64))) == meta['coverage']['visited_rows']
            del counts
            total = full = partial = 0
            for row in read_csv(run / 'training_windows.csv'):
                n = int(row['steps']); total += n
                assert int(row['end_step']) == total and row['terminated'] == 'False'
                if row['partial'] == 'True':
                    partial += 1; assert n == 200 and row['truncated'] == 'False'
                else:
                    full += 1; assert n == 300 and row['truncated'] == 'True'
                assert int(row['shots']) == int(row['hit_count']) + int(row['miss_count'])
                assert math.isclose(float(row['base_return']), 10*int(row['hit_count'])-int(row['miss_count'])-.05*n, abs_tol=1e-7)
                assert math.isclose(float(row['learning_return']), float(row['base_return'])+float(row['shaping_return']), abs_tol=1e-7)
                step, digest = int(row['end_step']), row['target_trajectory_sha256']
                assert step not in training_pairing[seed] or training_pairing[seed][step] == digest
                training_pairing[seed][step] = digest
            assert (total, full, partial) == (5000000, 16666, 1)
            validation = read_json(run / 'validation.json')
            assert [x['global_step'] for x in validation] == list(range(0, 5000001, 500000))
            for item in validation:
                folder = study / 'evaluation/development' / method / f'seed_{seed}'
                t = item['global_step']
                assert read_json(folder / f'checkpoint_{t:07d}.json') == item
                rows = read_csv(folder / f'windows_{t:07d}.csv')
                validate_windows(rows, seeds(config, 'development')); validate_summary(item['metrics'])
                pair_rows(development_pairing, rows)
                development_windows += len(rows); development_summaries += 1
            total_updates += total; run_count += 1
    assert (run_count, total_updates, development_summaries, development_windows) == (12,60000000,132,26400)
    splits = {}
    for split, expected in [('final',8000), ('stress',3200)]:
        folder = study / 'evaluation' / split
        recorded = read_json(folder / 'checks.json')
        for name, digest in recorded['files_sha256'].items(): assert sha(folder / name) == digest
        summaries = read_json(folder / 'summary.json'); pairing = {}; windows = 0
        assert len(summaries) == 16
        for key, summary in summaries.items():
            rows = read_csv(folder / f'{key}_windows.csv')
            validate_windows(rows, seeds(config, split)); validate_summary(summary); pair_rows(pairing, rows)
            windows += len(rows)
        assert windows == expected
        splits[split] = {'windows':windows, 'paired_scenarios':len(pairing)}
    # Independently recompute reported seed means and sample SDs from the saved final/stress summaries.
    comparison = read_json(study / 'evaluation/cross_seed_summary.json')
    for split in ('final', 'stress'):
        data = read_json(study / 'evaluation' / split / 'summary.json')
        for method, fields in comparison['groups'][split].items():
            for field, expected in fields.items():
                values = [data[f'{method}_seed{s}'][field] for s in (10,11,12)]
                assert math.isclose(float(np.mean(values)), expected['mean'], abs_tol=1e-12)
                assert math.isclose(float(np.std(values, ddof=1)), expected['sample_sd'], abs_tol=1e-12)
    gifs = []
    for clip in audit['clips']:
        path = ROOT / clip['path']; durations = []
        assert sha(path) == clip['sha256']
        with Image.open(path) as gif:
            assert gif.n_frames == 301
            for i in range(gif.n_frames):
                gif.seek(i); gif.load(); durations.append(gif.info.get('duration',0))
        assert sum(durations) == 20070 and set(durations) == {60,70}
        gifs.append({'path':clip['path'], 'sha256':sha(path), 'frames':301, 'duration_ms':20070})
    assert len(gifs) == 4
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    first = 'results/tracking_improvement/render/seed3601000_main_comparison.gif'
    assert re.search(r'!\[[^\]]*\]\('+re.escape(first)+r'\)',readme[:4500])
    assert all(clip['path'] in readme for clip in gifs)
    links = 0
    for path in [ROOT/'README.md', ROOT/'docs/tracking_improvement.md', study/'results_summary.md', ROOT/'provenance/tracking_improvement_publication.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:','#','mailto:')): continue
            assert (path.parent / target.split('#')[0]).exists(), (str(path),target)
            links += 1
    files = [p for p in study.rglob('*') if p.is_file()]
    for path in files:
        assert path.stat().st_size < 100*1024**2
        assert path.suffix != '.npy'
        if path.suffix in ('.json','.md','.csv','.txt'):
            assert not re.search(r'[\u3400-\u9fff]',path.read_text(encoding='utf-8')),str(path)
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
    result = verify()
    path = ROOT / args.output; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
