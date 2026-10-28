"""Sequential, fixed-budget research stages with immutable formal provenance."""
import copy
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..tracking25.agent import epsilon, learning_reward
from ..tracking25.env import TrackingEnv
from ..tracking25.metrics import EpisodeMetrics
from .agent import ResearchAgent, array_hash, coverage
from .evaluate import evaluate
from .io import ROOT, archive_location, git_state, historical_agent, peak_memory_bytes, read_csv, read_json, save_csv, save_json, sha

METHODS = ('absolute_base', 'absolute_potential', 'relative_base', 'relative_potential')


def source_hashes():
    files = list((ROOT / 'aimrl/tracking_research').glob('*.py'))
    files += list((ROOT / 'aimrl/tracking25').glob('*.py'))
    files += [ROOT / 'configs/tracking_research.json', ROOT / 'tests/test_tracking_research.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def seeds(config, split):
    spec = config['study'][split]
    return range(spec['start'], spec['start'] + spec['count'])


def collision_check(config):
    collisions, inspected = [], 0
    original = read_json(ROOT / '.tmp/tracking_research_start/snapshot.json')['historical_sha256']
    ranges = {split: set(seeds(config, split)) for split in ('development', 'final', 'stress')}
    for name in original:
        path = ROOT / name
        if path.suffix.lower() not in ('.json', '.csv', '.md', '.py', '.txt'): continue
        inspected += 1
        if path.suffix.lower() == '.csv':
            found = set()
            with path.open(newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                fields = [k for k in (reader.fieldnames or []) if 'seed' in k.lower()]
                if fields:
                    for row in reader:
                        for field in fields:
                            if row[field] and row[field].isdigit(): found.add(int(row[field]))
        elif path.suffix.lower() == '.json':
            found = set()
            def inspect(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if 'seed' in key.lower():
                            if isinstance(item,int):
                                count = max([v for k,v in value.items() if isinstance(v,int) and any(s in k.lower() for s in ('episodes','count','scenarios'))] or [1]) if 'start' in key else 1
                                found.update(range(item,item+count))
                            elif isinstance(item,list): found.update(x for x in item if isinstance(x,int))
                        inspect(item)
                elif isinstance(value,list):
                    for item in value: inspect(item)
            inspect(read_json(path))
        else:
            found = {int(x) for x in re.findall(r'(?<![\w.])(?:37|38|39)\d{5}(?![\w.])', path.read_text(encoding='utf-8'))}
        for split, values in ranges.items():
            overlap = sorted(values & found)
            if overlap: collisions.append({'path': name, 'split': split, 'seeds': overlap})
    if collisions: raise RuntimeError('Historical evaluation seed collision: ' + str(collisions))
    return {'historical_text_files_inspected': inspected, 'collisions': collisions,
            'scope': 'Original source, configuration, documentation and all historical experiment text records; independently generated scenarios can still share physical states.'}


def check(config, output):
    if output.exists() and any(output.iterdir()) and not (output / 'protocol.json').exists():
        entries = list(output.iterdir())
        failed_checks_only = (len(entries)==1 and entries[0].name=='stage_execution'
                              and all(p.name.startswith('check_') and read_json(p)['exit_code']==1 for p in entries[0].iterdir()))
        if not failed_checks_only: raise FileExistsError('Nonempty study directory retained; choose and record a fresh session suffix')
    output.mkdir(parents=True, exist_ok=True)
    expected = read_json(ROOT / 'configs/tracking25_v2.json')
    for key in ('environment', 'actions', 'rewards'):
        if config[key] != expected[key]: raise ValueError('Study physics/rewards must match V2')
    if config['learning'] != {**expected['learning'], 'training_steps': 5000000, 'epsilon_decay_steps': 4000000}:
        raise ValueError('Fixed study learning settings differ')
    if config['study']['methods'] != list(METHODS) or config['study']['training_seeds'] != [10, 11, 12]:
        raise ValueError('Fixed experiment matrix differs')
    for key, value in {'pilot_seed': 99, 'pilot_steps': 100000, 'checkpoint_interval': 500000,
                       'display_training_seed': 10, 'demo_seeds': [3601000, 3601001, 3601002]}.items():
        if config['study'][key] != value: raise ValueError('Fixed study setting differs: ' + key)
    for split, count in [('development', 200), ('final', 500), ('stress', 200)]:
        if config['study'][split]['count'] != count: raise ValueError('Split length differs')
    ranges = [set(seeds(config, x)) for x in ('development', 'final', 'stress')]
    if any(ranges[i] & ranges[j] for i in range(3) for j in range(i)): raise ValueError('Evaluation splits overlap')
    free = shutil.disk_usage(output).free
    if free < 3 * 1024**3: raise RuntimeError('At least 3 GiB free disk required')
    previous = read_json(output / 'protocol.json') if (output / 'protocol.json').exists() else None
    if previous and previous.get('formal_frozen'):
        verify_protocol(config, output); print('Existing formal protocol verified.', flush=True); return
    historical = read_json(ROOT / '.tmp/tracking_research_start/snapshot.json')
    archive = {}
    for mode in ('base', 'potential'):
        agent, archive[mode] = historical_agent(config, mode); del agent
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'formal_frozen': False,
                'config': config, 'source_sha256': source_hashes(), 'starting_snapshot': historical,
                'git_at_check': git_state(), 'seed_collision_check': collision_check(config),
                'historical_models': archive, 'free_disk_bytes': free,
                'interpreter': sys.executable, 'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                'formal_interactions': 60000000, 'pilot_interaction_cap': 400000,
                'rng': 'SeedSequence(training_seed).spawn(2): environment child generates one uint64 seed; exploration uses second child. Random uses SeedSequence([314159, scenario_seed]).',
                'representations': {'absolute': {'shape': [25,25,25,25,4,2,10], 'element_bytes': 125000000, 'indexing': 'tracking_absolute_v2'},
                                    'relative': {'shape': [49,49,4,2,10], 'element_bytes': 768320, 'indexing': 'tracking_relative_offsets_v1'}},
                'boundary_limitation': 'Relative observations alias wall clipping and target reflection; approximate aggregation, not an exact Markov state. No tabular optimality or convergence claim.',
                'metrics': {'distance': 'Post-motion ticks 1..300; pooled integer histogram p95; exclude initial frame.',
                            'acquisition': 'First post-motion d<=1; otherwise cap300 and acquired=false.',
                            'boundary': 'Pre-action either object within3 cells of any wall; also target/crosshair slices with denominators.',
                            'recovery': 'V2 overlapping event windows; last20ticks censored; failures capped20, not successful recoveries.',
                            'ties': 'Exact float32 equality primary; separately absolute tolerance1e-6 near ties.',
                            'coverage': 'uint32 state-action update counts; row sums are genuine update counts. Aggregated relative and absolute physical-state coverage are not equivalent.'},
                'interpretation': {'primary': 'relative_potential versus absolute_potential at5M, three seed means',
                                   'practical': ['distance <=90% of control', 'close >=control+0.05', 'far <=control', 'hits >=90% of control', 'pooled accuracy >=0.85'],
                                   'aspirational': {'mean_distance_max': 1.5, 'close_fraction_min': .75, 'far_fraction_max': .05, 'mean_capped_acquisition_max': 15}},
                'display': {'main': ['historical_potential', 'relative_base_seed10', 'relative_potential_seed10', 'rule'],
                            'six_policy': ['absolute_base_seed10','absolute_potential_seed10','relative_base_seed10','relative_potential_seed10','rule','random'],
                            'seeds': [3601000,3601001,3601002], 'selection': 'Fixed continuity scenarios; no outcome-based selection',
                            'frames': 301, 'fps': 15, 'duration_ms': 20070}, 'gui_verified': False}
    if previous:
        save_json(output / f'preflight_superseded_{time.time_ns()}.json', previous)
    save_json(output / 'protocol.json', protocol)
    print('Preflight captured; formal source freeze occurs after successful pilots. Historical models verified.', flush=True)


def verify_protocol(config, output, formal=False):
    protocol = read_json(output / 'protocol.json')
    if protocol['config'] != config: raise ValueError('Study configuration changed')
    if formal and (not protocol['formal_frozen'] or protocol['source_sha256'] != source_hashes()):
        raise ValueError('Formal source/configuration provenance mismatch')
    return protocol


def diagnose(config, output):
    verify_protocol(config, output)
    destination = output / 'diagnostics'
    if (destination / 'checks.json').exists():
        verify_stage_files(destination); print('Verified archived diagnostics reused.', flush=True); return
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Partial diagnostics retained')
    destination.mkdir(parents=True)
    results, pairing, archive = {}, {}, {}
    for policy in ('historical_base', 'historical_potential', 'rule', 'random'):
        agent = None
        if policy.startswith('historical_'): agent, archive[policy] = historical_agent(config, policy.split('_')[1])
        summary, rows, traces = evaluate(config, policy, range(3600000,3600200), agent)
        pair_rows(pairing, rows)
        save_csv(destination / f'{policy}_windows.csv', rows)
        demo_summary, demo_rows, traces = evaluate(config, policy, config['study']['demo_seeds'], agent, True)
        pair_rows(pairing, demo_rows)
        save_csv(destination / f'{policy}_demo_traces.csv', traces)
        save_csv(destination / f'{policy}_demo_windows.csv', demo_rows)
        results[policy] = {'development': summary, 'demos': demo_summary}
        del agent
        print(f'Archived diagnosis {policy}: distance={summary["mean_distance"]:.3f}; far={summary["far_fraction"]:.3%}', flush=True)
    # Replay the original recorded development metrics rather than accepting approximate visual similarity.
    original = {r['method']: r for r in read_csv(archive_location() / 'evaluation/comparison.csv')}
    for policy, values in results.items():
        key = policy.replace('historical_', '')
        for field in ('mean_distance', 'close_fraction', 'hits_per_episode', 'pooled_shot_accuracy'):
            if not math.isclose(values['development'][field], float(original[key][field]), abs_tol=1e-10):
                raise ValueError('Archived metric replay differs: ' + policy + '/' + field)
    save_json(destination / 'summary.json', results)
    finish_stage(destination, {'paired_scenarios': len(pairing), 'archived_models': archive, 'original_scores_reproduced': True})


def pair_rows(pairing, rows):
    for row in rows:
        seed, digest = int(row['scenario_seed']), row['target_trajectory_sha256']
        if seed in pairing and pairing[seed] != digest: raise ValueError('Unpaired target trajectory')
        pairing[seed] = digest


def finish_stage(directory, extra):
    files = {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*') if p.is_file() and p.name != 'checks.json'}
    save_json(directory / 'checks.json', {**extra, 'files_sha256': files})


def verify_stage_files(directory):
    checks = read_json(directory / 'checks.json')
    for name, digest in checks['files_sha256'].items():
        if sha(directory / name) != digest: raise ValueError('Stage record changed: ' + name)
    return checks


def train_run(config, method, seed, run, output, pilot=False):
    steps = config['study']['pilot_steps'] if pilot else config['learning']['training_steps']
    if run.exists() and any(run.iterdir()):
        if (run / 'metadata.json').exists():
            meta = verify_run(config, method, seed, run, steps)
            print('Verified completed run reused:', method, seed, flush=True); return meta
        raise FileExistsError('Incomplete attempt retained; no silent restart: ' + str(run))
    run.mkdir(parents=True)
    save_json(run / 'run_config.json', config)
    source = source_hashes()
    save_json(run / 'attempt.json', {'method': method, 'seed': seed, 'budget': steps, 'source_sha256': source,
                                  'started_utc': datetime.now(timezone.utc).isoformat(), 'pilot': pilot})
    representation, reward_mode = method.split('_')
    env_sequence, exploration_sequence = np.random.SeedSequence(seed).spawn(2)
    env_seed = int(env_sequence.generate_state(1, dtype=np.uint64)[0])
    env = TrackingEnv(config); state, _ = env.reset(env_seed)
    agent = ResearchAgent(config, representation, np.random.default_rng(exploration_sequence))
    metric = EpisodeMetrics(state); learning_total = shaping_total = 0.0
    started = time.perf_counter(); checkpoints = []

    def checkpoint(t):
        # Pilots use three development scenarios for a bounded end-to-end instrumentation check.
        scenario_seeds = list(seeds(config, 'development'))[:3] if pilot else seeds(config, 'development')
        summary, records, _ = evaluate(config, method, scenario_seeds, agent)
        item = {'global_step': t, 'method': method, 'training_seed': seed, 'coverage': coverage(agent), 'metrics': summary}
        if item['coverage']['update_sum'] != t: raise RuntimeError('Update-count conservation failed')
        checkpoints.append(item)
        destination = run / 'development' if pilot else output / 'evaluation/development' / method / f'seed_{seed}'
        save_json(destination / f'checkpoint_{t:07d}.json', item)
        save_csv(destination / f'windows_{t:07d}.csv', records)
        print(f'{"Pilot" if pilot else "Formal"} {method} seed{seed} step{t:,}: distance={summary["mean_distance"]:.3f}, close={summary["close_fraction"]:.3%}, hits={summary["hits_per_episode"]:.2f}', flush=True)

    checkpoint(0)
    windows = 0
    with (run / 'training_windows.csv').open('w', newline='', encoding='utf-8') as file:
        writer = None
        for t in range(steps):
            action = agent.select(state, epsilon(t, config['learning']))
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated: raise RuntimeError('Unexpected true termination')
            learned, shaping = learning_reward(reward, state, next_state, reward_mode, agent.gamma)
            agent.update(state, action, learned, next_state)
            metric.add(action, next_state, reward, info)
            learning_total += learned; shaping_total += shaping; state = next_state
            if truncated or t + 1 == steps:
                row = {'end_step': t + 1, 'terminated': False, 'truncated': truncated, 'partial': not truncated,
                       'learning_return': learning_total, 'shaping_return': shaping_total, **metric.finish()}
                if writer is None: writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                if truncated:
                    windows += 1
                    if t + 1 < steps:
                        state, _ = env.reset(); metric = EpisodeMetrics(state); learning_total = shaping_total = 0.0
            if (not pilot and (t + 1) % config['study']['checkpoint_interval'] == 0) or (pilot and t + 1 == steps):
                file.flush(); checkpoint(t + 1)
    agent.save_package(run / 'q_table.npz')
    model_hash, count_hash = array_hash(agent.q), array_hash(agent.counts)
    with np.load(run / 'q_table.npz', allow_pickle=False) as z:
        if not np.array_equal(z['q_table'], agent.q): raise ValueError('Model reload inequality')
    with np.load(run / 'visitation.npz', allow_pickle=False) as z:
        if not np.array_equal(z['counts'], agent.counts): raise ValueError('Counter reload inequality')
    save_json(run / 'validation.json', checkpoints)
    metadata = {'method': method, 'representation': representation, 'indexing_version': agent.indexing_version,
                'reward_mode': reward_mode, 'training_seed': seed, 'interactions': steps, 'pilot': pilot,
                'complete_windows': windows, 'partial_ticks': steps % 300, 'environment_seed': env_seed,
                'elapsed_seconds': time.perf_counter() - started, 'process_lifetime_peak_working_set_bytes': peak_memory_bytes(),
                'timing_scope': 'Training, validation, update-count diagnostics, compression and exact reload checks',
                'q_shape': list(agent.shape), 'q_element_bytes': agent.q.nbytes, 'counter_bytes': agent.counts.nbytes,
                'q_array_sha256': model_hash, 'counts_array_sha256': count_hash,
                'source_sha256': source, 'reload_exact': True, 'coverage': coverage(agent),
                'files_sha256': {p.name: sha(p) for p in run.iterdir() if p.is_file()}}
    if source_hashes() != source: raise RuntimeError('Training source changed during run')
    save_json(run / 'metadata.json', metadata)
    del agent
    verify_run(config, method, seed, run, steps)
    print(f'Completed {method} seed{seed}: {steps:,} updates in {metadata["elapsed_seconds"]:.2f}s', flush=True)
    return metadata


def verify_run(config, method, seed, run, steps=5000000):
    meta = read_json(run / 'metadata.json')
    if (meta['method'], meta['training_seed'], meta['interactions']) != (method, seed, steps): raise ValueError('Run identity/budget mismatch')
    if read_json(run / 'run_config.json') != config or meta['source_sha256'] != source_hashes(): raise ValueError('Run source/config mismatch')
    for name, digest in meta['files_sha256'].items():
        if sha(run / name) != digest: raise ValueError('Run artifact integrity failure: ' + name)
    total = full = partial = 0
    with (run / 'training_windows.csv').open(newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            n = int(row['steps']); total += n
            if int(row['end_step']) != total or row['terminated'] != 'False': raise ValueError('Window/update accounting failed')
            if row['partial'] == 'True':
                partial += 1
                if row['truncated'] != 'False' or n != steps % 300: raise ValueError('Budget stop mislabelled')
            else:
                full += 1
                if n != 300 or row['truncated'] != 'True': raise ValueError('Full window mislabelled')
            if int(row['shots']) != int(row['hit_count']) + int(row['miss_count']): raise ValueError('Shot accounting failed')
            if not math.isclose(float(row['base_return']), 10*int(row['hit_count'])-int(row['miss_count'])-.05*n, abs_tol=1e-7): raise ValueError('Return identity failed')
            if not math.isclose(float(row['learning_return']), float(row['base_return'])+float(row['shaping_return']), abs_tol=1e-7): raise ValueError('Shaping identity failed')
    if (total, full, partial) != (steps, steps // 300, int(steps % 300 > 0)): raise ValueError('Run incomplete')
    with np.load(run / 'visitation.npz', allow_pickle=False) as z:
        c = z['counts']
        if int(c.sum(dtype=np.uint64)) != steps or array_hash(c) != meta['counts_array_sha256']: raise ValueError('Counter conservation failed')
    with np.load(run / 'q_table.npz', allow_pickle=False) as z:
        if array_hash(z['q_table']) != meta['q_array_sha256']: raise ValueError('Model array hash mismatch')
    validation = read_json(run / 'validation.json')
    expected = [0, steps] if meta['pilot'] else list(range(0, steps+1, 500000))
    if [v['global_step'] for v in validation] != expected: raise ValueError('Checkpoint list differs')
    return meta


def pilot(config, output):
    verify_protocol(config, output)
    results = []
    for method in METHODS:
        results.append(train_run(config, method, 99, output / 'pilots' / method, output, True))
    save_json(output / 'pilots/summary.json', {'interactions': sum(r['interactions'] for r in results), 'runs': results,
                                            'purpose': 'Instrumentation, execution and timing only; Q values are never transferred to formal runs.'})


def train(config, output):
    protocol = verify_protocol(config, output)
    if not (output / 'pilots/summary.json').exists(): raise RuntimeError('Four bounded pilots must pass first')
    for method in METHODS: verify_run(config, method, 99, output / 'pilots' / method, 100000)
    if not protocol['formal_frozen']:
        if (output / 'runs').exists(): raise FileExistsError('Unexpected pre-existing formal runs')
        save_json(output / f'preflight_before_formal_{time.time_ns()}.json', protocol)
        protocol.update({'formal_frozen': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hashes()})
        save_json(output / 'protocol.json', protocol)
    verify_protocol(config, output, True)
    for method in METHODS:
        for seed in config['study']['training_seeds']:
            train_run(config, method, seed, output / 'runs' / method / f'seed_{seed}', output)
    print('Exactly twelve formal runs completed.', flush=True)


def final_evaluation(config, output):
    verify_protocol(config, output, True)
    for method in METHODS:
        for seed in config['study']['training_seeds']:
            verify_run(config, method, seed, output / 'runs' / method / f'seed_{seed}')
    for split in ('final', 'stress'):
        destination = output / 'evaluation' / split
        if (destination / 'checks.json').exists(): verify_stage_files(destination); continue
        if destination.exists() and any(destination.iterdir()): raise FileExistsError('Partial evaluation retained: ' + split)
        destination.mkdir(parents=True)
        condition = copy.deepcopy(config)
        if split == 'stress': condition['environment'].update(turn_probability=.10, speed_toggle_probability=.10)
        summaries, pairing = {}, {}
        instances = [(method, seed) for method in METHODS for seed in (10,11,12)] + [('historical_base',None),('historical_potential',None),('rule',None),('random',None)]
        for method, seed in instances:
            key = method if seed is None else f'{method}_seed{seed}'
            agent = None
            if method in METHODS:
                agent = ResearchAgent(config, method.split('_')[0])
                agent.load_package(output / 'runs' / method / f'seed_{seed}' / 'q_table.npz', True)
            elif method.startswith('historical_'): agent, _ = historical_agent(config, method.split('_')[1])
            summary, rows, _ = evaluate(condition, method, seeds(config, split), agent)
            pair_rows(pairing, rows)
            save_csv(destination / f'{key}_windows.csv', rows)
            summaries[key] = summary
            save_json(destination / f'{key}_summary.json', summary)
            del agent
            print(f'{split} {key}: distance={summary["mean_distance"]:.3f}, close={summary["close_fraction"]:.3%}, hits={summary["hits_per_episode"]:.2f}', flush=True)
        save_json(destination / 'summary.json', summaries)
        finish_stage(destination, {'policy_instances': len(instances), 'windows': len(instances)*len(pairing),
                                   'ticks': len(instances)*len(pairing)*300, 'paired_scenarios': len(pairing), 'scenario_target_hashes': pairing})
