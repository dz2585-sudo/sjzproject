"""Freeze the matched protocol, train without final access, then evaluate."""
import csv
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import time

import numpy as np

from ..reproduce import memory_record
from ..tracking25.env import TrackingEnv
from ..tracking_research.agent import ResearchAgent, array_hash
from ..tracking_research.io import ROOT, read_json, save_csv, save_json, sha
from .core import Kernel, evaluate, load_model, reward_parts, save_model, summarize, window_metrics

CONFIG = ROOT / 'configs/tracking_shooting_tradeoff.json'
SOURCE = (
    'aimrl/tracking_tradeoff/kernel.c', 'aimrl/tracking_tradeoff/core.py', 'aimrl/tracking_tradeoff/study.py',
    'aimrl/tracking_tradeoff/__main__.py', 'aimrl/tracking_tradeoff/__init__.py',
    'aimrl/tracking25/env.py', 'aimrl/tracking25/agent.py', 'aimrl/tracking25/policies.py',
    'aimrl/tracking_research/agent.py', 'aimrl/tracking_research/io.py', 'aimrl/reproduce.py',
    'tests/test_tracking_tradeoff.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def config_checked():
    config = read_json(CONFIG)
    legacy = read_json(ROOT / 'configs/tracking_research.json')
    for key in ('environment', 'actions', 'rewards', 'learning'):
        require(config[key] == legacy[key], f'Frozen legacy setting differs: {key}')
    return config


def scenario_seeds(config, split):
    spec = config['tradeoff'][split]
    return range(spec['start'], spec['start'] + spec['count'])


def pair(rows, seen):
    for row in rows:
        key, digest = row['scenario_seed'], row['target_sha256']
        require(seen.setdefault(key, digest) == digest, f'Target pairing failed: {key}')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def freeze(config, library, output):
    tracked = git('ls-files', '-z').split('\0')
    paths = [p for p in tracked if p]
    historical = {p: sha(ROOT / p) for p in paths}
    save_json(output / 'historical_sha256_before.json', historical)
    save_json(output / 'protocol.json', {
        'config': config, 'config_sha256': sha(CONFIG), 'git_head': git('rev-parse', 'HEAD'),
        'git_remote': git('remote', 'get-url', 'origin'),
        'git_status_at_freeze': git('status', '--porcelain=v1', '--untracked-files=normal'),
        'source_sha256': {p: sha(ROOT / p) for p in SOURCE}, 'kernel_sha256': sha(library),
        'runtime': {'python': platform.python_version(), 'platform': platform.system(),
                    'architecture': platform.machine(),
                    'dependencies': {p: importlib.metadata.version(p) for p in ('numpy','matplotlib','Pillow')}},
        'historical_inventory': 'historical_sha256_before.json',
        'state': 'Relative offset, heading, speed. Lowest action ID breaks exact Q ties.',
        'training_rng': 'SeedSequence(training_seed).spawn(2). Environment child yields uint64 reset seed; action child supplies PCG64.',
        'target_draws': 'Three doubles per tick in original order. Window resets use original TrackingEnv.',
        'reward': 'base - lambda*d_post + beta*(d_post <= 1) + gamma*(-d_post) - (-d_pre)',
        'bootstrap': 'Actual next state on every update, including hit, truncation and budget stop.',
        'evaluation': 'D/C use post-motion distance. H is hits per 300 ticks. A pools hits and shots within each seed.',
        'boundary': 'Either object at most three cells from an edge before the action.',
        'selection': 'Final development checkpoint only. Per family screen H >= 0.98*H0 and A >= 0.98, then max C, min D, max H, min weight. If no candidate passes, same order without screen.',
        'demonstration': 'Fixed demo seeds, reference, selected distance, selected close, and Rule. Learned panels all use training seed 20.',
        'diagnostics': 'Final scenario 4200000, seed 20 and Rule. First movement crossing target on same axis, first stay, first three consecutive post-motion ticks at distance exactly one.',
        'pareto': 'D-H and C-H dominance assessed separately among all measured means including Rule.',
        'seed_collision_check': 'Historical seed-bearing configurations, protocols and records inspected before the freeze.',
    })


def verify_protocol(config, library, output):
    protocol = read_json(output / 'protocol.json')
    require(config == protocol['config'] and sha(CONFIG) == protocol['config_sha256'], 'Frozen configuration changed')
    require(sha(library) == protocol['kernel_sha256'], 'Frozen native library changed')
    for p, digest in protocol['source_sha256'].items():
        require(sha(ROOT / p) == digest, f'Frozen source changed: {p}')
    return protocol


def train_run(config, kernel, output, condition, seed, budget, smoke=False):
    started = time.perf_counter()
    run = output / 'runs' / condition / f'seed_{seed}'
    checkpoints = run / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=False)
    save_json(run / 'run_config.json', config)
    env_sequence, exploration_sequence = np.random.SeedSequence(seed).spawn(2)
    env = TrackingEnv(config)
    initial, _ = env.reset(int(env_sequence.generate_state(1, dtype=np.uint64)[0]))
    agent = ResearchAgent(config, 'relative', np.random.default_rng(exploration_sequence))
    weights = config['tradeoff']['conditions'][condition]
    development = range(4100000, 4100003) if smoke else scenario_seeds(config, 'development')
    interval = config['tradeoff']['checkpoint_interval']
    results = []

    def checkpoint(step):
        require(int(agent.counts.sum(dtype=np.uint64)) == step, 'Checkpoint update count mismatch')
        target = checkpoints / f'step_{step:07d}.npz'
        save_model(target, agent, config, condition, seed, step)
        rows, _ = evaluate(config, kernel, agent, development)
        save_csv(run / 'development' / f'windows_{step:07d}.csv', rows)
        results.append({'step': step, 'checkpoint_sha256': sha(target), **summarize(rows)})

    checkpoint(0)
    step = full = partial = 0
    with (run / 'training_windows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = None
        while step < budget:
            length = min(300, budget - step)
            stop = step + length
            next_checkpoint = (step // interval + 1) * interval
            if not smoke and next_checkpoint < stop:
                first = kernel.run(env, initial, next_checkpoint - step, agent, step, weights, train=True)
                checkpoint(next_checkpoint)
                state = tuple(map(int, first[-1, :6]))
                tail = kernel.run(env, state, stop - next_checkpoint, agent, next_checkpoint, weights, train=True)
                ticks = np.vstack([first, tail])
            else:
                ticks = kernel.run(env, initial, length, agent, step, weights, train=True)
            step = stop
            row = window_metrics(initial, ticks)
            base, distance, close, potential, train = reward_parts(
                -0.05 + 10.0 * ticks[:, 11] - ticks[:, 12], ticks[:, 7], ticks[:, 9], weights)
            sums = [float(np.sum(component)) for component in (base, distance, close, potential, train)]
            require(math.isclose(sum(sums[:4]), sums[4], rel_tol=0, abs_tol=1e-8), 'Window reward identity failed')
            require(math.isclose(row['base_return'], sums[0], rel_tol=0, abs_tol=1e-8), 'Base accounting mismatch')
            record = {'end_step': step, 'truncated': length == 300, 'partial': length != 300,
                      **row, 'distance_term': sums[1], 'close_term': sums[2], 'potential_term': sums[3],
                      'training_return': sums[4]}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            full += length == 300
            partial += length != 300
            if step == budget or (not smoke and step % interval == 0):
                stream.flush()
                checkpoint(step)
            if step < budget:
                initial, _ = env.reset()
    final = checkpoints / f'step_{budget:07d}.npz'
    loaded = load_model(final, config, condition, seed, budget)
    require(np.array_equal(loaded.q, agent.q) and np.array_equal(loaded.counts, agent.counts),
            'Exact final checkpoint reload failed')
    save_json(run / 'development_summary.json', results)
    metadata = {'condition': condition, 'seed': seed, 'interactions': step, 'smoke': smoke,
                'full_windows': full, 'partial_windows': partial, 'partial_ticks': budget % 300,
                'checkpoints': len(results), 'reload_exact': True,
                'q_sha256': array_hash(agent.q), 'counts_sha256': array_hash(agent.counts),
                'environment_rng_final': env.rng.bit_generator.state,
                'exploration_rng_final': agent.rng.bit_generator.state,
                'elapsed_seconds': time.perf_counter() - started, 'memory': memory_record()}
    save_json(run / 'metadata.json', metadata)
    print(f'{condition} seed {seed}: {step:,} updates, {len(results)} checkpoints, '
          f'{metadata["elapsed_seconds"]:.2f} s', flush=True)
    return metadata


def smoke_stage(config, kernel, output):
    target = output / 'smoke'
    target.mkdir(exist_ok=False)
    runs = [train_run(config, kernel, target, condition, config['tradeoff']['smoke_seed'], 1001, True)
            for condition in config['tradeoff']['conditions']]
    save_json(target / 'checks.json', {'complete': True, 'runs': runs, 'interactions': len(runs) * 1001})


def run_dir(output, condition, seed):
    return output / 'runs' / condition / f'seed_{seed}'


def select(config, output):
    design = config['tradeoff']
    ranking = {}
    for condition, weights in design['conditions'].items():
        rows = [read_json(run_dir(output, condition, s) / 'development_summary.json')[-1]
                for s in design['training_seeds']]
        ranking[condition] = {k: float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows)
                              else None for k in ('D', 'C', 'H', 'A')}
    h0 = ranking['reference']['H']
    selections = {}
    for family, weight_index in (('distance', 0), ('close', 1)):
        family_members = [c for c in ranking if c.startswith(family + '_')]
        eligible = [c for c in family_members if ranking[c]['H'] >= .98 * h0
                    and ranking[c]['A'] is not None and ranking[c]['A'] >= .98]
        chosen = min(eligible or family_members, key=lambda c: (
            -ranking[c]['C'], ranking[c]['D'], -ranking[c]['H'], design['conditions'][c][weight_index]))
        selections[family] = {'condition': chosen, 'screen_passed': bool(eligible), 'eligible': eligible}
    return {'split': 'development', 'checkpoint': 5000000, 'training_seeds': design['training_seeds'],
            'display_training_seed': design['display_training_seed'], 'mean_metrics': ranking,
            'selected': selections, 'written_before_final': True}


def train_stage(config, kernel, library, output):
    smoke = read_json(output / 'smoke/checks.json')
    require(smoke['complete'] and smoke['interactions'] == 7007, 'Seven short runs must pass first')
    gates = read_json(output / 'preflight.json')
    require(gates['setup_exit_code'] == 0 and gates['tests_exit_code'] == 0 and gates['seed_search_exit_code'] == 0,
            'Setup, tests and seed audit must pass before freezing')
    require(not (output / 'protocol.json').exists() and not (output / 'evaluation').exists(), 'Choose a new formal study directory')
    freeze(config, library, output)
    started = time.perf_counter()
    runs = [train_run(config, kernel, output, condition, seed, config['learning']['training_steps'])
            for condition in config['tradeoff']['conditions'] for seed in config['tradeoff']['training_seeds']]
    require(sum(run['interactions'] for run in runs) == 105000000, 'Formal training total differs')
    verify_protocol(config, library, output)
    save_json(output / 'selection.json', select(config, output))
    save_json(output / 'training_complete.json', {'complete': True, 'runs': runs, 'interactions': 105000000,
                                                   'elapsed_seconds': time.perf_counter() - started})


def evaluate_stage(config, kernel, library, output):
    verify_protocol(config, library, output)
    training = read_json(output / 'training_complete.json')
    selection = read_json(output / 'selection.json')
    require(training['complete'] and len(training['runs']) == 21 and training['interactions'] == 105000000
            and selection['written_before_final'], 'All training and development selection must precede evaluation')
    destination = output / 'evaluation'
    destination.mkdir(exist_ok=False)
    instances = [(r['condition'], r['seed']) for r in training['runs']] + [('rule', None)]
    summaries = []
    for split, expected in (('final', 11000), ('stress', 4400)):
        condition_config = json.loads(json.dumps(config))
        if split == 'stress':
            for key in ('turn_probability','speed_toggle_probability'):
                condition_config['environment'][key] = config['tradeoff']['stress'][key]
        seen = {}
        records = []
        for condition, seed in instances:
            key = condition if seed is None else f'{condition}_seed{seed}'
            agent = None if seed is None else load_model(
                run_dir(output, condition, seed) / 'checkpoints/step_5000000.npz', config, condition, seed, 5000000)
            rows, _ = evaluate(condition_config, kernel, agent, scenario_seeds(config, split), detailed=True)
            pair(rows, seen)
            save_csv(destination / split / f'{key}_windows.csv', rows)
            records.extend(rows)
            summaries.append({'split':split, 'condition': condition, 'seed': seed, **summarize(rows)})
        require(len(records) == expected and len(seen) == len(scenario_seeds(config, split)), 'Wrong evaluation window budget')
        save_json(destination / split / 'paired_targets.json', seen)
        print(f'{split}: {len(records):,} windows, {len(seen)} paired target trajectories', flush=True)
    save_json(destination / 'per_seed.json', summaries)
    save_csv(output / 'per_seed.csv', summaries)
