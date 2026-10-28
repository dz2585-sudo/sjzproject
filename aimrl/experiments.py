"""Frozen Grid5 benchmark experiment: independent train, test and summarize stages."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model_io import artifact_digest, resolve_artifact
from .provenance import verify_historical_hashes

from .agent import QLearningAgent
from .evaluate import evaluate
from .policies import random_policy, rule_policy
from .provenance import source_hash_matches
from .train import train

ROOT = Path(__file__).resolve().parent.parent
METHODS = ('base', 'potential')
SEEDS = (0, 1, 2)
SPEEDS = (.1, .25, .5)
METRICS = ('success_rate', 'mean_steps', 'mean_misses', 'mean_base_return')


def digest(path):
    return artifact_digest(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(resolve_artifact(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot write empty records: {path}')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    if len(values) != 3:
        raise ValueError('Expected exactly three independent training seeds')
    return statistics.mean(values), statistics.stdev(values)


def threshold(rows):
    for a, b in zip(rows, rows[1:]):
        if float(a['success_rate']) >= .9 and float(b['success_rate']) >= .9:
            return True, int(a['global_step']), int(b['global_step'])
    return False, None, None


def require_keys(rows, fields, expected):
    keys = [tuple(str(r[k]) for k in fields) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError(f'Missing, duplicate, or unexpected keys for {fields}: got {len(keys)}, expected {len(expected)}')


def frozen_protocol(path):
    frozen = read_json(path)
    config_path = ROOT / 'configs/base.json'
    if digest(config_path) != frozen['config_sha256'] or read_json(config_path) != frozen['effective_config']:
        raise ValueError('Frozen configuration mismatch')
    mismatches = [name for name, sha in frozen['source_sha256'].items() if not source_hash_matches(ROOT, name, sha)]
    if mismatches:
        raise ValueError(f'Frozen source mismatch: {mismatches}')
    config = frozen['effective_config']
    if (frozen['methods'] != list(METHODS) or frozen['training_root_seeds'] != list(SEEDS)
            or frozen['training_steps_per_run'] != 100000 or frozen['final_test']['move_probabilities'] != list(SPEEDS)
            or config['learning']['initial_q'] != 0 or config['learning']['training_steps'] != 100000
            or config['experiments']['validation'] != {'interval_steps': 10000, 'episodes': 200, 'seed_start': 100000}
            or frozen['final_test']['seed_start'] != 200000 or frozen['final_test']['episodes_per_condition'] != 500):
        raise ValueError('Unexpected protocol settings')
    return frozen, config


def evaluation_config(config, speed):
    result = copy.deepcopy(config)
    result['environment']['target_move_probability'] = speed
    return result


def evaluate_policy(config, method, seed, speed, agent, start, count, model_path=''):
    effective = evaluation_config(config, speed)
    if method == 'rule':
        policy = rule_policy
    elif method in METHODS:
        policy = agent.greedy_action
    elif method != 'random':
        raise ValueError(f'Unknown method {method}')
    before = None if agent is None else agent.q.copy()
    rows = []
    for scenario in range(start, start + count):
        if method == 'random':
            rng = np.random.default_rng(np.random.SeedSequence([314159, scenario]))
            policy = lambda state: random_policy(rng)
        _, result = evaluate(effective, policy, scenario, 1)
        r = result[0]
        rows.append({'method': method, 'training_seed': '' if seed is None else seed, 'move_probability': speed,
                     'scenario_seed': scenario, 'success': int(r['hit']), 'steps': r['steps'], 'misses': r['misses'],
                     'base_return': r['base_return'], 'terminated': int(r['terminated']),
                     'truncated': int(r['truncated']), 'model_path': model_path})
    if agent is not None and not np.array_equal(before, agent.q):
        raise RuntimeError('Evaluation modified Q table')
    return rows


def verify_episodes(rows):
    for row in rows:
        hit, steps, misses = int(row['success']), int(row['steps']), int(row['misses'])
        term, trunc = int(row['terminated']), int(row['truncated'])
        if (hit not in (0, 1) or term != hit or trunc not in (0, 1) or term + trunc != 1
                or not 1 <= steps <= 50 or not 0 <= misses <= steps or (trunc and steps != 50)
                or not math.isclose(float(row['base_return']), 10 * hit - .05 * steps - misses, abs_tol=1e-9)):
            raise ValueError(f'Invalid episode: {row}')


def train_stage(protocol_path, output, frozen, config):
    runs = output / 'runs'
    if runs.exists() and any(runs.iterdir()):
        raise FileExistsError('Training runs already exist; use a fresh output directory')
    if (output / 'protocol_used.json').exists():
        raise FileExistsError('Protocol manifest already exists; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    runs.mkdir(exist_ok=True)
    started = now()
    for mode in METHODS:
        for seed in SEEDS:
            train(config, seed, runs / f'{mode}_seed{seed}', reward_mode=mode)
    models, regression = [], {}
    expected_steps = [str(x) for x in range(0, 100001, 10000)]
    for mode in METHODS:
        for seed in SEEDS:
            run = runs / f'{mode}_seed{seed}'
            meta, actual = read_json(run / 'metadata.json'), read_json(run / 'run_config.json')
            valid = read_csv(run / 'validation.csv')
            episodes = read_csv(run / 'validation_episodes.csv')
            agent = QLearningAgent(actual); agent.load(run / 'q_table.npy')
            initial = np.load(run / 'checkpoints/q_000000.npy', allow_pickle=False)
            last = np.load(run / 'checkpoints/q_100000.npy', allow_pickle=False)
            if (actual != config or meta['reward_mode'] != mode or meta['root_seed'] != seed or meta['global_steps'] != 100000
                    or np.any(initial) or not np.array_equal(last, agent.q) or [r['global_step'] for r in valid] != expected_steps):
                raise ValueError(f'Incomplete or inconsistent training run: {run}')
            expected = {(step, str(s)) for step in expected_steps for s in range(100000, 100200)}
            require_keys(episodes, ('global_step', 'scenario_seed'), expected)
            if meta['complete_episode_steps'] + meta['unfinished_episode']['steps'] != 100000:
                raise ValueError(f'Training step accounting failed: {run}')
            models.append({'method': mode, 'training_seed': seed, 'steps': 100000,
                           'model_path': str((run / 'q_table.npy').resolve()), 'model_sha256': digest(run / 'q_table.npy'),
                           'config_path': str((run / 'run_config.json').resolve()), 'config_sha256': digest(run / 'run_config.json'),
                           'elapsed_seconds': meta['elapsed_seconds']})
            if seed == 0:
                old = ROOT / 'results/reward_shaping_development' / f'{mode}_seed0'
                regression[mode] = {'q_equal': np.array_equal(agent.q, np.load(old / 'q_table.npy', allow_pickle=False)),
                                    'validation_equal': valid == read_csv(old / 'validation.csv')}
                if not all(regression[mode].values()):
                    raise ValueError(f'reward-shaping development seed 0 regression mismatch for {mode}')
    # A pre-test diagnostic uses distinct scenarios and does not touch saved model files.
    diagnostic = []
    for speed in SPEEDS:
        for mode in METHODS:
            run = runs / f'{mode}_seed0'
            agent = QLearningAgent(config); agent.load(run / 'q_table.npy')
            diagnostic += evaluate_policy(config, mode, 0, speed, agent, 300000, 5)
        for mode in ('rule', 'random'):
            diagnostic += evaluate_policy(config, mode, None, speed, None, 300000, 5)
    verify_episodes(diagnostic)
    save_csv(output / 'diagnostic_episodes.csv', diagnostic)
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
               for p in sorted(list((ROOT / 'aimrl').glob('*.py')) + list((ROOT / 'tests').glob('test_*.py')) + [ROOT / 'requirements.txt'])}
    save_json(output / 'source_hashes.json', sources)
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {'frozen_protocol_path': str(Path(protocol_path).resolve()), 'frozen_protocol_sha256': digest(protocol_path),
                'config_sha256': digest(ROOT / 'configs/base.json'), 'config': config, 'source_hashes': sources,
                'source_changes': 'See provenance/source_compatibility.json for post-experiment English translations and exact hash mappings. Training, environment, and reward-update rules are unchanged; this manifest describes this later reproduction run.',
                'git_revision': revision, 'git_note': 'No commit available; source hashes identify the working files.' if revision is None else 'Working files identified by source hashes.',
                'environment': {'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__, 'matplotlib': importlib.metadata.version('matplotlib')},
                'models': models, 'regression_development_seed0': regression, 'training_started_utc': started, 'training_completed_utc': now(),
                'model_selection': 'Fixed final 100000-step model; no selection by validation or test performance.',
                'test_settings': {'move_probabilities': list(SPEEDS), 'turn_probability': .1, 'scenario_seed_start': 200000,
                                  'episodes_per_condition': 500, 'random_policy_rng': 'NumPy SeedSequence([314159, scenario_seed]), independent per episode',
                                  'metrics': 'success, all-episode steps, misses and undiscounted base return'},
                'diagnostic': {'seed_start': 300000, 'seed_end': 300004, 'episode_count': len(diagnostic), 'passed': True},
                'test_started': False, 'test_completed': False, 'test_started_utc': None, 'test_completed_utc': None}
    save_json(output / 'protocol_used.json', manifest)
    print('Training complete; pre-test manifest saved; diagnostics passed.')


def test_stage(protocol_path, output, frozen, config):
    manifest_path = output / 'protocol_used.json'
    manifest = read_json(manifest_path)
    test_dir = output / 'test'
    if manifest['test_started'] or test_dir.exists():
        raise FileExistsError('Test already started or test directory exists; preserve it and use a new output directory')
    if manifest['frozen_protocol_sha256'] != digest(protocol_path):
        raise ValueError('Frozen protocol changed since training')
    if any(not source_hash_matches(ROOT, name, sha) for name, sha in manifest['source_hashes'].items()):
        raise ValueError('Source changed after training; refusing to open final test')
    for model in manifest['models']:
        if digest(model['model_path']) != model['model_sha256'] or digest(model['config_path']) != model['config_sha256']:
            raise ValueError('Model or saved configuration changed before testing')
    manifest['test_started'], manifest['test_started_utc'] = True, now()
    save_json(manifest_path, manifest)
    test_dir.mkdir()
    learned, baselines = [], []
    for model in manifest['models']:
        agent = QLearningAgent(config); agent.load(model['model_path'])
        for speed in SPEEDS:
            learned += evaluate_policy(config, model['method'], model['training_seed'], speed, agent, 200000, 500, model['model_path'])
    for method in ('rule', 'random'):
        for speed in SPEEDS:
            baselines += evaluate_policy(config, method, None, speed, None, 200000, 500)
    verify_episodes(learned); verify_episodes(baselines)
    expected = {(m, str(s), str(p), str(i)) for m in METHODS for s in SEEDS for p in SPEEDS for i in range(200000, 200500)}
    require_keys(learned, ('method', 'training_seed', 'move_probability', 'scenario_seed'), expected)
    expected_base = {(m, '', str(p), str(i)) for m in ('rule', 'random') for p in SPEEDS for i in range(200000, 200500)}
    require_keys(baselines, ('method', 'training_seed', 'move_probability', 'scenario_seed'), expected_base)
    for model in manifest['models']:
        if digest(model['model_path']) != model['model_sha256'] or digest(model['config_path']) != model['config_sha256']:
            raise RuntimeError('Model or training config changed during test')
    save_csv(test_dir / 'learned_episodes.csv', learned)
    save_csv(test_dir / 'baseline_episodes.csv', baselines)
    manifest['test_completed'], manifest['test_completed_utc'] = True, now()
    save_json(manifest_path, manifest)
    print(f'Final test complete: {len(learned)} learned and {len(baselines)} baseline episodes.')


def aggregate_episodes(rows):
    n = len(rows)
    return {'success_rate': sum(int(r['success']) for r in rows) / n,
            'mean_steps': sum(int(r['steps']) for r in rows) / n,
            'mean_misses': sum(int(r['misses']) for r in rows) / n,
            'mean_base_return': sum(float(r['base_return']) for r in rows) / n}


def summarize_stage(output):
    manifest = read_json(output / 'protocol_used.json')
    if not manifest['test_completed']:
        raise ValueError('Final test not completed')
    learned = read_csv(output / 'test/learned_episodes.csv')
    baseline = read_csv(output / 'test/baseline_episodes.csv')
    verify_episodes(learned); verify_episodes(baseline)
    expected = {(m, str(s), str(p), str(i)) for m in METHODS for s in SEEDS for p in SPEEDS for i in range(200000, 200500)}
    require_keys(learned, ('method', 'training_seed', 'move_probability', 'scenario_seed'), expected)
    expected = {(m, '', str(p), str(i)) for m in ('rule', 'random') for p in SPEEDS for i in range(200000, 200500)}
    require_keys(baseline, ('method', 'training_seed', 'move_probability', 'scenario_seed'), expected)
    per_seed, overall, baseline_summary, learning, learning_summary, thresholds, paired = [], [], [], [], [], [], []
    for mode in METHODS:
        for seed in SEEDS:
            run = output / 'runs' / f'{mode}_seed{seed}'
            vals = read_csv(run / 'validation.csv')
            require_keys(vals, ('global_step',), {(str(i),) for i in range(0, 100001, 10000)})
            vals.sort(key=lambda r: int(r['global_step']))
            learning.extend({'method': mode, 'training_seed': seed, **r} for r in vals)
            achieved, first, confirmation = threshold(vals)
            thresholds.append({'method': mode, 'training_seed': seed, 'achieved': achieved,
                               'first_checkpoint_step': first, 'confirmation_step': confirmation})
            for speed in SPEEDS:
                selected = [r for r in learned if r['method'] == mode and int(r['training_seed']) == seed and float(r['move_probability']) == speed]
                per_seed.append({'method': mode, 'training_seed': seed, 'move_probability': speed,
                                 'episodes': len(selected), **aggregate_episodes(selected)})
        for speed in SPEEDS:
            selected = [r for r in per_seed if r['method'] == mode and r['move_probability'] == speed]
            row = {'method': mode, 'move_probability': speed, 'n_train_seeds': 3, 'episodes_per_seed': 500}
            for metric in METRICS:
                row[metric + '_mean'], row[metric + '_std'] = mean_std([r[metric] for r in selected])
            overall.append(row)
        for step in range(0, 100001, 10000):
            selected = [r for r in learning if r['method'] == mode and int(r['global_step']) == step]
            row = {'method': mode, 'global_step': step, 'n_train_seeds': 3, 'episodes_per_seed': 200}
            for metric in METRICS:
                row[metric + '_mean'], row[metric + '_std'] = mean_std([float(r[metric]) for r in selected])
            learning_summary.append(row)
    for method in ('rule', 'random'):
        for speed in SPEEDS:
            selected = [r for r in baseline if r['method'] == method and float(r['move_probability']) == speed]
            baseline_summary.append({'method': method, 'move_probability': speed, 'episodes': len(selected),
                                     'n_train_seeds': 'N/A', **aggregate_episodes(selected)})
    for seed in SEEDS:
        for speed in SPEEDS:
            b = next(r for r in per_seed if r['method'] == 'base' and r['training_seed'] == seed and r['move_probability'] == speed)
            p = next(r for r in per_seed if r['method'] == 'potential' and r['training_seed'] == seed and r['move_probability'] == speed)
            paired.append({'training_seed': seed, 'move_probability': speed,
                           'success_potential_minus_base': p['success_rate'] - b['success_rate'],
                           'steps_base_minus_potential': b['mean_steps'] - p['mean_steps'],
                           'misses_base_minus_potential': b['mean_misses'] - p['mean_misses']})
    summary_dir = output / 'summary'; summary_dir.mkdir(exist_ok=True)
    save_csv(output / 'test/learned_per_seed.csv', per_seed)
    save_csv(output / 'test/learned_summary.csv', overall)
    save_csv(output / 'test/baseline_summary.csv', baseline_summary)
    save_csv(summary_dir / 'learning_per_seed.csv', learning)
    save_csv(summary_dir / 'learning_summary.csv', learning_summary)
    save_csv(summary_dir / 'thresholds.csv', thresholds)
    save_csv(summary_dir / 'paired_differences.csv', paired)
    counts = {'learned_episodes': len(learned), 'baseline_episodes': len(baseline), 'learned_per_seed': len(per_seed),
              'learned_summary': len(overall), 'baseline_summary': len(baseline_summary), 'learning_per_seed': len(learning),
              'learning_summary': len(learning_summary), 'thresholds': len(thresholds), 'paired_differences': len(paired)}
    if list(counts.values()) != [9000, 3000, 18, 6, 6, 66, 22, 6, 9]:
        raise ValueError(f'Unexpected output counts: {counts}')
    for model in manifest['models']:
        if digest(model['model_path']) != model['model_sha256'] or digest(model['config_path']) != model['config_sha256']:
            raise ValueError('Model or config changed after testing')
    checks = {'counts': counts, 'unique_keys': True, 'episode_invariants': True, 'model_and_config_hashes_unchanged': True,
              'training_steps_total': sum(m['steps'] for m in manifest['models']), 'development_seed0_regression': manifest['regression_development_seed0'],
              'threshold_resolution_steps': 10000, 'test_completed': manifest['test_completed']}
    save_json(summary_dir / 'checks.json', checks)
    plot_results(summary_dir, learning_summary, overall, baseline_summary)
    lines = ['# Grid5 benchmark Formal Experiment Record', '',
             'Each method was trained from zero for 100,000 steps using root seeds 0, 1, and 2. Validation used scenario seeds 100000–100199; final testing used seeds 200000–200499 after the final models were fixed. Error terms are sample standard deviations across three training seeds, not confidence intervals.',
             f"Execution environment: {manifest['environment']['platform']}, Python {sys.version.split()[0]}, NumPy {np.__version__}, Matplotlib {manifest['environment']['matplotlib']}.",
             f"Sum of elapsed times for six training runs: {sum(m['elapsed_seconds'] for m in manifest['models']):.3f} seconds; training: {manifest['training_started_utc']} to {manifest['training_completed_utc']}; testing: {manifest['test_started_utc']} to {manifest['test_completed_utc']}.", '',
             '## Validation (Mean ± Sample Standard Deviation)', '', '| Method | Training steps | Success rate | Mean steps | Mean misses |', '|---|---:|---:|---:|---:|']
    for r in learning_summary:
        if r['global_step'] in (0, 10000, 30000, 100000):
            lines.append(f"| {r['method']} | {r['global_step']} | {100*r['success_rate_mean']:.2f}% ± {100*r['success_rate_std']:.2f}% | {r['mean_steps_mean']:.3f} ± {r['mean_steps_std']:.3f} | {r['mean_misses_mean']:.3f} ± {r['mean_misses_std']:.3f} |")
    lines += ['', '## Two Consecutive Validation Checkpoints at ≥90%', '', '| Method | Training seed | First checkpoint | Confirmation checkpoint |', '|---|---:|---:|---:|']
    for r in thresholds:
        lines.append(f"| {r['method']} | {r['training_seed']} | {r['first_checkpoint_step'] if r['achieved'] else 'Not observed'} | {r['confirmation_step'] if r['achieved'] else 'Not observed'} |")
    for mode in METHODS:
        group = [r for r in thresholds if r['method'] == mode]
        if all(r['achieved'] for r in group):
            mean, std = mean_std([r['first_checkpoint_step'] for r in group])
            lines.append(f'\n{mode} reached the threshold in 3/3 runs; mean first checkpoint: {mean:.0f}; sample standard deviation: {std:.0f} steps.')
        else:
            lines.append(f'\n{mode} reached the threshold in {sum(r["achieved"] for r in group)}/3 runs; a full-method mean was not calculated.')
    lines += ['', 'The checkpoint resolution is 10,000 steps; these observations do not establish an exact training-speed ratio.', '',
              '## Final Test', '', '| Method | Move probability | Success rate | Mean steps | Mean misses |', '|---|---:|---:|---:|---:|']
    for r in overall:
        lines.append(f"| {r['method']} | {r['move_probability']:.2f} | {100*r['success_rate_mean']:.2f}% ± {100*r['success_rate_std']:.2f}% | {r['mean_steps_mean']:.3f} ± {r['mean_steps_std']:.3f} | {r['mean_misses_mean']:.3f} ± {r['mean_misses_std']:.3f} |")
    for r in baseline_summary:
        lines.append(f"| {r['method']} | {r['move_probability']:.2f} | {100*r['success_rate']:.2f}% | {r['mean_steps']:.3f} | {r['mean_misses']:.3f} |")
    lines += ['', '## Paired Differences', '', 'For each training seed, paired_differences.csv reports potential − base success rate and base − potential mean steps and misses. A positive difference favors potential shaping for the corresponding metric.']
    for r in paired:
        lines.append(f"- Training seed {r['training_seed']}, move probability {r['move_probability']:.2f}: success-rate difference {r['success_potential_minus_base']:+.3f}, step difference {r['steps_base_minus_potential']:+.3f}, miss difference {r['misses_base_minus_potential']:+.3f}.")
    lines += ['', 'Limitations: only three training seeds were used, and seed 0 was used during development. This small fully observable environment and finite test set do not establish generalization to real games. The rule policy is a baseline, not a proven optimum. Results are descriptive; no statistical-significance claim is made. New test seeds do not guarantee initial states absent from training.',
              'Figures and CSV files were generated from actual raw records; the GUI has not been verified. Command: `python -m aimrl.experiments --protocol results/reward_shaping_development/protocol_frozen.json --output results/grid5_benchmark --stage train`; then use --stage test and --stage summarize in order. Outputs are stored in results/grid5_benchmark.']
    (summary_dir / 'results_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Summaries and plots generated:', counts)


def plot_results(output, learning, learned, baseline):
    from .mpl_setup import configure_mpl
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for mode in METHODS:
        rows = [r for r in learning if r['method'] == mode]
        x = np.array([r['global_step'] for r in rows])
        for ax, metric, scale in ((axes[0], 'success_rate', 100), (axes[1], 'mean_steps', 1)):
            y = np.array([r[metric + '_mean'] * scale for r in rows]); sd = np.array([r[metric + '_std'] * scale for r in rows])
            ax.plot(x, y, marker='o', label=mode); ax.fill_between(x, y - sd, y + sd, alpha=.16)
    axes[0].set_ylabel('Success rate (%)'); axes[1].set_ylabel('Mean steps'); axes[1].set_xlabel('Training interaction steps')
    for ax in axes: ax.grid(alpha=.3); ax.legend()
    fig.suptitle('Validation: mean ± 1 sample SD across 3 training seeds'); fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for mode in (*METHODS, 'rule', 'random'):
        rows = [r for r in (learned if mode in METHODS else baseline) if r['method'] == mode]
        x = [r['move_probability'] for r in rows]
        for ax, metric, scale in ((axes[0], 'success_rate', 100), (axes[1], 'mean_steps', 1)):
            if mode in METHODS:
                ax.errorbar(x, [r[metric + '_mean'] * scale for r in rows], yerr=[r[metric + '_std'] * scale for r in rows], marker='o', capsize=3, label=mode)
            else:
                ax.plot(x, [r[metric] * scale for r in rows], marker='o', label=mode)
    axes[0].set_ylabel('Success rate (%)'); axes[1].set_ylabel('Mean steps'); axes[1].set_xlabel('Target move probability')
    axes[1].set_xticks(SPEEDS)
    for ax in axes: ax.grid(alpha=.3); ax.legend(ncol=2)
    fig.suptitle('Final test: learned mean ± 1 sample SD; baselines have no SD'); fig.tight_layout()
    fig.savefig(output / 'test_comparison.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'test', 'summarize'), required=True)
    args = parser.parse_args()
    frozen, config = frozen_protocol(args.protocol)
    if args.stage == 'train': train_stage(args.protocol, args.output, frozen, config)
    elif args.stage == 'test': test_stage(args.protocol, args.output, frozen, config)
    else: summarize_stage(args.output)


if __name__ == '__main__':
    main()
