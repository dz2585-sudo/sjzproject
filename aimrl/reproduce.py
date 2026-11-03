"""Portable, additive reproduction of the published continuous-tracking study."""
import argparse
import copy
import csv
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .provenance import source_hash_matches
from .tracking25.agent import epsilon, learning_reward
from .tracking25.env import TrackingEnv
from .tracking25.metrics import EpisodeMetrics
from .tracking_research.agent import ResearchAgent, array_hash, coverage
from .tracking_research.evaluate import evaluate
from .tracking_research.io import ROOT, read_json, save_csv, save_json, sha

METHODS = ('absolute_base', 'absolute_potential', 'relative_base', 'relative_potential')
METRICS = ('mean_distance', 'close_fraction', 'far_fraction', 'hits_per_episode',
           'shots_per_episode', 'misses_per_episode', 'pooled_shot_accuracy',
           'mean_base_return', 'mean_first_acquisition_tick')
SCHEMA = 'tracking_reproduction_v1'


def memory_record():
    """Best-effort process-lifetime telemetry; platform semantics stay explicit."""
    try:
        if sys.platform == 'win32':
            from .tracking_research.io import peak_memory_bytes
            value, metric = peak_memory_bytes(), 'windows_peak_working_set'
        elif sys.platform in ('linux', 'darwin'):
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            value *= 1024 if sys.platform == 'linux' else 1
            metric = 'ru_maxrss_peak_resident_set'
        else:
            return {'available': False, 'bytes': None, 'reason': 'unsupported platform'}
        return {'available': True, 'bytes': int(value), 'metric': metric,
                'scope': 'process lifetime; not an attributed stage delta'}
    except Exception as exc:
        return {'available': False, 'bytes': None, 'reason': type(exc).__name__}


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'aimrl').rglob('*.py'))}


def checked_config():
    config = read_json(ROOT / 'configs/tracking_research.json')
    protocol = read_json(ROOT / 'results/tracking_improvement/protocol.json')
    if config != protocol['config'] or not protocol['formal_frozen']:
        raise ValueError('Expected the published, frozen study configuration')
    for name, expected in protocol['source_sha256'].items():
        if not source_hash_matches(ROOT, name, expected):
            raise ValueError('Unverified published source: ' + name)
    return config, protocol


def fresh_output(path, model_root):
    out = Path(path).resolve()
    protected = [ROOT / p for p in ('aimrl', 'tests', 'scripts', 'configs', 'docs',
                                   'results', 'outputs', 'provenance', '.git')]
    protected.append(Path(model_root).resolve())
    if out == ROOT or ROOT.is_relative_to(out):
        raise ValueError('Output cannot be the repository or its ancestor')
    if any(out == p.resolve() or out.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Output overlaps protected source or model evidence')
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_checked_model(config, method, seed, model_root):
    run = Path(model_root) / 'runs' / method / f'seed_{seed}'
    meta = read_json(run / 'metadata.json')
    if (meta['method'], meta['training_seed']) != (method, seed):
        raise ValueError('Model identity mismatch')
    if read_json(run / 'run_config.json') != config:
        raise ValueError('Model configuration mismatch')
    if meta.get('reproduction_schema') == SCHEMA:
        if meta['source_sha256'] != code_hashes():
            raise ValueError('Reproduction source changed since training')
    else:
        protocol = read_json(ROOT / 'results/tracking_improvement/protocol.json')
        if meta['source_sha256'] != protocol['source_sha256'] or meta['interactions'] != 5000000:
            raise ValueError('Published model protocol or budget mismatch')
    for name in ('q_table.npz', 'visitation.npz', 'run_config.json'):
        if sha(run / name) != meta['files_sha256'][name]:
            raise ValueError('Model package checksum mismatch: ' + name)
    agent = ResearchAgent(config, method.split('_')[0])
    agent.load_package(run / 'q_table.npz', with_counts=True)
    if (array_hash(agent.q) != meta['q_array_sha256'] or
            array_hash(agent.counts) != meta['counts_array_sha256'] or
            int(agent.counts.sum(dtype=np.uint64)) != meta['interactions']):
        raise ValueError('Model/counter array integrity mismatch')
    return agent, meta


def clarified_summary(summary):
    result = copy.deepcopy(summary)
    for item in result['slices'].values():
        item['fully_blocked_move_fraction'] = item['clipped_move_fraction']
    result['metric_note'] = ('Legacy clipped_move_fraction counts zero-displacement moves only. '
                             'fully_blocked_move_fraction is an explicit alias; partial clipping is not counted.')
    return result


def export_comparison(output, summaries):
    rows = [{'policy': k, **{f: v[f] for f in METRICS}} for k, v in summaries.items()]
    save_csv(output / 'comparison_per_instance.csv', rows)
    grouped = {}
    for method in METHODS:
        values = [v for k, v in summaries.items() if k.startswith(method + '_seed')]
        if values:
            grouped[method] = {f: {'n': len(values), 'mean': float(np.mean([v[f] for v in values])),
                                  'sample_sd': float(np.std([v[f] for v in values], ddof=1)) if len(values) > 1 else None}
                               for f in METRICS if all(v[f] is not None for v in values)}
    save_json(output / 'cross_seed_summary.json', grouped)
    from .mpl_setup import configure_mpl
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    labels = list(summaries)
    for ax, field in zip(axes, ('mean_distance', 'close_fraction', 'hits_per_episode')):
        ax.bar(range(len(labels)), [summaries[k][field] for k in labels])
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha='right', fontsize=7)
        ax.set_title(field.replace('_', ' '))
    fig.suptitle('Saved-model replay: individual policy instances; no learning updates')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def replay(config, methods, training_seeds, model_root, output, split='final', limit=None,
           baselines=True):
    condition = copy.deepcopy(config)
    if split == 'stress':
        condition['environment'].update(turn_probability=.10, speed_toggle_probability=.10)
    spec = config['study'][split]
    count = spec['count'] if limit is None else limit
    if not 1 <= count <= spec['count']:
        raise ValueError('Scenario limit is outside the selected split')
    scenarios = range(spec['start'], spec['start'] + count)
    summaries, hashes, comparisons = {}, {}, {}
    instances = [(m, s) for m in methods for s in training_seeds]
    if baselines:
        instances += [('rule', None), ('random', None)]
    reference_path = Path(model_root) / 'evaluation' / split / 'summary.json'
    reference = read_json(reference_path) if reference_path.exists() else None
    compare = reference is not None and count == spec['count']
    for method, seed in instances:
        key = method if seed is None else f'{method}_seed{seed}'
        agent = None
        if seed is not None:
            agent, _ = load_checked_model(config, method, seed, model_root)
        summary, records, _ = evaluate(condition, method, scenarios, agent)
        for row in records:
            scenario, digest = row['scenario_seed'], row['target_trajectory_sha256']
            if scenario in hashes and hashes[scenario] != digest:
                raise ValueError('Target trajectories are not paired')
            hashes[scenario] = digest
        if compare:
            if key not in reference:
                raise ValueError('Missing recorded comparison: ' + key)
            differences = {}
            for field in METRICS:
                actual, expected = summary[field], reference[key][field]
                if actual is None or expected is None:
                    if actual != expected:
                        raise ValueError('Undefined metric mismatch: ' + field)
                    differences[field] = None
                else:
                    differences[field] = actual - expected
                    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
                        raise ValueError('Recorded metric mismatch: ' + key + '/' + field)
            comparisons[key] = differences
        summaries[key] = clarified_summary(summary)
        save_csv(output / f'{key}_windows.csv', records)
        print(f'{key}: {count} windows; mean distance {summary["mean_distance"]:.6f}', flush=True)
        del agent
    save_json(output / 'summary.json', summaries)
    export_comparison(output, summaries)
    result = {'split': split, 'scenarios_per_instance': count, 'policy_instances': len(instances),
              'evaluation_interactions': count * 300 * len(instances),
              'paired_scenarios': len(hashes), 'full_split': count == spec['count'],
              'recorded_comparison_performed': compare, 'metric_differences': comparisons,
              'all_compared_metrics_match': True if compare else None}
    save_json(output / 'replay_checks.json', result)
    return result


def train_one(config, method, seed, output, smoke=False):
    """Same RNG, update, and boundary semantics; independent session provenance."""
    run = output / 'runs' / method / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=False)
    source = code_hashes()
    save_json(run / 'run_config.json', config)
    steps = 1001 if smoke else config['learning']['training_steps']
    env_sequence, action_sequence = np.random.SeedSequence(seed).spawn(2)
    env = TrackingEnv(config)
    state, _ = env.reset(int(env_sequence.generate_state(1, dtype=np.uint64)[0]))
    agent = ResearchAgent(config, method.split('_')[0], np.random.default_rng(action_sequence))
    spec = config['study']['development']
    scenarios = range(spec['start'], spec['start'] + (3 if smoke else spec['count']))
    checkpoints = []

    def checkpoint(step):
        summary, rows, _ = evaluate(config, method, scenarios, agent)
        item = {'global_step': step, 'metrics': summary, 'coverage': coverage(agent)}
        if item['coverage']['update_sum'] != step:
            raise ValueError('Training update-count mismatch')
        checkpoints.append(item)
        save_csv(run / 'development' / f'windows_{step:07d}.csv', rows)

    started = time.perf_counter()
    checkpoint(0)
    metric = EpisodeMetrics(state)
    learned_sum = shaped_sum = 0.0
    with (run / 'training_windows.csv').open('w', newline='', encoding='utf8') as stream:
        writer = None
        for t in range(steps):
            action = agent.select(state, epsilon(t, config['learning']))
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                raise ValueError('Unexpected true termination')
            learned, shaping = learning_reward(reward, state, next_state, method.split('_')[1], agent.gamma)
            agent.update(state, action, learned, next_state)
            metric.add(action, next_state, reward, info)
            learned_sum += learned
            shaped_sum += shaping
            state = next_state
            if truncated or t + 1 == steps:
                row = {'end_step': t + 1, 'terminated': False, 'truncated': truncated,
                       'partial': not truncated, 'learning_return': learned_sum,
                       'shaping_return': shaped_sum, **metric.finish()}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if truncated and t + 1 < steps:
                    state, _ = env.reset()
                    metric = EpisodeMetrics(state)
                    learned_sum = shaped_sum = 0.0
            if t + 1 == steps or (not smoke and (t + 1) % config['study']['checkpoint_interval'] == 0):
                stream.flush()
                checkpoint(t + 1)
    agent.save_package(run / 'q_table.npz')
    restored = ResearchAgent(config, method.split('_')[0])
    restored.load_package(run / 'q_table.npz', with_counts=True)
    if not np.array_equal(agent.q, restored.q) or not np.array_equal(agent.counts, restored.counts):
        raise ValueError('Exact reload verification failed')
    if code_hashes() != source:
        raise ValueError('Source changed during training')
    save_json(run / 'validation.json', checkpoints)
    meta = {'reproduction_schema': SCHEMA, 'method': method, 'training_seed': seed,
            'interactions': steps, 'smoke_fixture': smoke, 'source_sha256': source,
            'q_array_sha256': array_hash(agent.q), 'counts_array_sha256': array_hash(agent.counts),
            'coverage': coverage(agent), 'reload_exact': True, 'elapsed_seconds': time.perf_counter() - started,
            'memory': memory_record(), 'complete_windows': steps // 300, 'partial_ticks': steps % 300,
            'files_sha256': {p.name: sha(p) for p in run.iterdir() if p.is_file()}}
    save_json(run / 'metadata.json', meta)
    print(f'{method} seed {seed}: {steps} training updates; exact reload passed', flush=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('verify', 'evaluate', 'train'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'results/tracking_improvement')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['relative_potential'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[10])
    parser.add_argument('--split', choices=('development', 'final', 'stress'), default='final')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke', action='store_true', help='Train 1001 steps; retain the original epsilon schedule')
    parser.add_argument('--no-baselines', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Methods and nonnegative training seeds must be unique')
    if args.mode != 'train' and args.smoke:
        parser.error('--smoke applies only to train')
    if args.mode != 'evaluate' and args.limit is not None:
        parser.error('--limit applies only to evaluate')
    config, protocol = checked_config()
    if args.limit is not None and not 1 <= args.limit <= config['study'][args.split]['count']:
        parser.error('Scenario limit outside the selected split')
    output = fresh_output(args.output, args.model_root)
    record = {'schema': SCHEMA, 'complete': False, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'config': config, 'source_sha256': code_hashes(),
              'published_protocol_sha256': sha(ROOT / 'results/tracking_improvement/protocol.json'),
              'platform': platform.platform(), 'python': sys.version,
              'dependencies': {k: importlib.metadata.version(k) for k in ('numpy', 'matplotlib', 'Pillow')}}
    save_json(output / 'session.json', record)
    started = time.perf_counter()
    try:
        if args.mode == 'verify':
            optimization = ['-O'] * sys.flags.optimize
            subprocess.run([sys.executable, *optimization, str(ROOT / 'scripts/verify_tracking_publication.py'),
                            '--output', str(output / 'publication_verification.json')], cwd=ROOT, check=True)
        elif args.mode == 'evaluate':
            record['evaluation'] = replay(config, args.methods, args.seeds, args.model_root.resolve(),
                                          output, args.split, args.limit, not args.no_baselines)
        else:
            record['runs'] = [train_one(config, m, s, output, args.smoke) for m in args.methods for s in args.seeds]
        if code_hashes() != record['source_sha256']:
            raise ValueError('Source changed during this session')
        record['complete'] = True
    except Exception as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        record.update(elapsed_seconds=time.perf_counter() - started, memory=memory_record())
        save_json(output / 'session.json', record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
