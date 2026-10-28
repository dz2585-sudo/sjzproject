"""Check, summarize, and export the local single-seed 25x25 development trial."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

from .model_io import artifact_digest, resolve_artifact
from .provenance import verify_historical_hashes

from .agent import QLearningAgent
from .env import AimEnv
from .evaluate import evaluate
from .mpl_setup import configure_mpl
from .policies import random_policy, rule_policy

ROOT = Path(__file__).resolve().parent.parent
DEMO_SEED = 3501000
MODES = ('base', 'potential')


def read_json(path):
    return json.loads(resolve_artifact(path).read_text(encoding='utf-8'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return artifact_digest(path)


def validate_config(config):
    expected = {
        'environment': {'grid_size': 25, 'target_move_probability': .25, 'target_turn_probability': .1, 'max_episode_steps': 100},
        'rewards': {'step': -.05, 'hit_bonus': 10, 'miss_penalty': -1},
        'learning': {'alpha': .15, 'gamma': .99, 'initial_q': 0, 'q_dtype': 'float32', 'training_steps': 5000000,
                     'epsilon_start': 1, 'epsilon_end': .05, 'epsilon_decay_steps': 4000000, 'shaping_scale': 1},
    }
    for name, settings in expected.items():
        if config[name] != settings:
            raise ValueError(f'Trial configuration mismatch: {name}')
    if (config['actions'] != ['left', 'right', 'up', 'down', 'stay', 'fire']
            or config['experiments']['training_seeds'] != [0] or config['experiments']['reward_modes'] != list(MODES)
            or config['experiments']['validation'] != {'interval_steps': 500000, 'episodes': 200, 'seed_start': 3500000}
            or config['experiments']['test']['enabled'] is not False):
        raise ValueError('Unexpected trial actions, seeds, validation settings, or active test fields')


def historical_files():
    paths = [ROOT / 'configs/base.json', ROOT / 'README.md', ROOT / 'spec.md', ROOT / 'requirements.txt']
    paths += [p for p in (ROOT / 'aimrl').glob('*.py') if p.name != 'trial25.py']
    paths += [p for p in (ROOT / 'results').rglob('*') if p.is_file() and (p.relative_to(ROOT / 'results').parts[0] in ('q_learning_development', 'reward_shaping_development', 'grid5_benchmark', 'environment_baselines'))]
    paths += [p for p in (ROOT / 'provenance').glob('*') if p.is_file()]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def preflight(config, output):
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'preflight.json'
    if path.exists():
        raise FileExistsError(f'Preflight already exists: {path}; verify or use a new session directory')
    free = shutil.disk_usage(output).free
    if free < 600000000:
        raise RuntimeError('At least 600 MB of free space is required for checkpoints and exports')
    agent = QLearningAgent(config)
    if agent.q.shape != (25, 25, 25, 25, 2, 6) or agent.q.nbytes != 18750000 or np.any(agent.q):
        raise RuntimeError('Q table shape, dtype, size, or initialization mismatch')
    value = {'scope': 'Local single-seed development trial; no final test or GitHub upload',
             'python': sys.version, 'interpreter': sys.executable, 'platform': platform.platform(),
             'numpy': np.__version__, 'matplotlib': importlib.metadata.version('matplotlib'),
             'pillow': importlib.metadata.version('pillow'), 'added_dependencies': [],
             'disk_free_bytes': free, 'raw_q_storage_two_runs_bytes': 450000000,
             'state_combinations': 781250, 'q_values': agent.q.size, 'q_shape': list(agent.q.shape),
             'q_element_bytes': agent.q.nbytes, 'config': config,
             'historical_file_sha256': historical_files(),
             'timing_definition': 'Training metadata elapsed_seconds includes validation, checkpoint/final model writes, reload checks, baselines, and log preparation.'}
    save_json(path, value)
    print(f'Trial preflight passed: {agent.q.shape}, {agent.q.nbytes:,} element bytes; {free:,} bytes free.')


def load_agent(run, expected_config, mode):
    """Use the model's saved trial configuration rather than the default 5x5 file."""
    run = Path(run)
    config, metadata = read_json(run / 'run_config.json'), read_json(run / 'metadata.json')
    validate_config(config)
    if config != expected_config or metadata['reward_mode'] != mode or metadata['root_seed'] != 0 or metadata['global_steps'] != 5000000:
        raise ValueError(f'Model metadata or configuration mismatch in {run}')
    agent = QLearningAgent(config)
    agent.load(run / 'q_table.npy')
    return agent


def verify_episode(row, step_field='steps', hit_field='hit'):
    hit = row[hit_field] in (True, 1, 'True', '1')
    term = row['terminated'] in (True, 1, 'True', '1')
    trunc = row['truncated'] in (True, 1, 'True', '1')
    steps, misses = int(row[step_field]), int(row['misses'])
    if (hit != term or term == trunc or not 1 <= steps <= 100 or not 0 <= misses <= steps
            or (trunc and steps != 100) or not math.isclose(float(row['base_return']), 10 * hit - .05 * steps - misses, abs_tol=1e-8)):
        raise ValueError(f'Invalid episode: {row}')


def summarize(config, output):
    validate_config(config)
    summary_dir = output / 'summary'
    if summary_dir.exists() and any(summary_dir.iterdir()):
        raise FileExistsError('Summary output already exists; preserve existing results')
    summary_dir.mkdir(exist_ok=True)
    validation_rows, comparison, baselines = [], [], []
    run_checks = {}
    for mode in MODES:
        run = output / f'{mode}_seed0'
        agent = load_agent(run, config, mode)
        metadata = read_json(run / 'metadata.json')
        rows, records = read_csv(run / 'validation.csv'), read_csv(run / 'validation_episodes.csv')
        expected_steps = list(range(0, 5000001, 500000))
        if [int(r['global_step']) for r in rows] != expected_steps:
            raise ValueError('Missing, duplicate, or out-of-order validation checkpoints')
        keys = [(int(r['global_step']), int(r['scenario_seed'])) for r in records]
        expected = {(t, s) for t in expected_steps for s in range(3500000, 3500200)}
        if len(keys) != 2200 or len(set(keys)) != 2200 or set(keys) != expected:
            raise ValueError('Missing or duplicate validation scenarios')
        for r in records:
            verify_episode(r)
        for r in rows:
            group = [x for x in records if x['global_step'] == r['global_step']]
            values = {'success_rate': sum(x['hit'] == 'True' for x in group) / 200,
                      'mean_steps': sum(int(x['steps']) for x in group) / 200,
                      'mean_misses': sum(int(x['misses']) for x in group) / 200,
                      'mean_base_return': sum(float(x['base_return']) for x in group) / 200}
            if not all(math.isclose(values[k], float(r[k]), abs_tol=1e-9) for k in values):
                raise ValueError('Validation summary disagrees with episode records')
        episodes = read_csv(run / 'episodes.csv')
        total_steps = 0
        for r in episodes:
            verify_episode(r, 'length')
            total_steps += int(r['length'])
            if int(r['end_step']) != total_steps:
                raise ValueError('Training episode step accounting is inconsistent')
            if not math.isclose(float(r['base_return']) + float(r['shaping_return']), float(r['learning_return']), abs_tol=1e-8):
                raise ValueError('Learning/base/shaping return identity failed')
            if mode == 'base' and float(r['shaping_return']) != 0:
                raise ValueError('Base run unexpectedly includes shaping')
        if total_steps != metadata['complete_episode_steps'] or total_steps + metadata['unfinished_episode']['steps'] != 5000000:
            raise ValueError('Training budget does not equal 5,000,000 interactions')
        checkpoints = list((run / 'checkpoints').glob('*.npy'))
        if len(checkpoints) != 11:
            raise ValueError('Expected eleven checkpoint files')
        initial = np.load(run / 'checkpoints/q_000000.npy', mmap_mode='r', allow_pickle=False)
        final = np.load(run / 'checkpoints/q_5000000.npy', mmap_mode='r', allow_pickle=False)
        if np.any(initial) or not np.array_equal(final, agent.q):
            raise ValueError('Initial/final checkpoint invariant failed')
        # Check each checkpoint serially without retaining full additional arrays.
        for step in expected_steps:
            q = np.load(run / 'checkpoints' / f'q_{step:06d}.npy', mmap_mode='r', allow_pickle=False)
            if q.shape != agent.q.shape or q.dtype != np.float32 or not np.isfinite(q).all():
                raise ValueError('Checkpoint array is incompatible or contains non-finite values')
            del q
        run_checks[mode] = {'global_steps': metadata['global_steps'], 'complete_episodes': len(episodes),
                            'complete_episode_steps': total_steps, 'unfinished_episode': metadata['unfinished_episode'],
                            'elapsed_seconds': metadata['elapsed_seconds'], 'q_shape': list(agent.q.shape),
                            'q_dtype': str(agent.q.dtype), 'q_nbytes': agent.q.nbytes, 'final_model_sha256': sha(run / 'q_table.npy'),
                            'validation_summary_rows': len(rows), 'validation_episode_rows': len(records),
                            'reload_and_final_checkpoint_equal': True, 'zero_initial_checkpoint': True}
        validation_rows.extend({'method': mode, **r} for r in rows)
        comparison.append({'method': mode, 'training_seed': 0, 'scenario_seed_start': 3500000, 'episodes': 200,
                           **{k: float(rows[-1][k]) for k in ('success_rate', 'mean_steps', 'mean_misses', 'mean_base_return')}})
        del initial, final, agent, episodes, records
    _, rule_records = evaluate(config, rule_policy, 3500000, 200)
    random_records = []
    for seed in range(3500000, 3500200):
        rng = np.random.default_rng(np.random.SeedSequence([314159, seed]))
        _, rows = evaluate(config, lambda state: random_policy(rng), seed, 1)
        random_records.extend(rows)
    for mode, records in (('rule', rule_records), ('random', random_records)):
        for r in records:
            verify_episode(r)
        baselines.extend({'method': mode, **r} for r in records)
        comparison.append({'method': mode, 'training_seed': '', 'scenario_seed_start': 3500000, 'episodes': 200,
                           'success_rate': sum(r['hit'] for r in records) / 200,
                           'mean_steps': sum(r['steps'] for r in records) / 200,
                           'mean_misses': sum(r['misses'] for r in records) / 200,
                           'mean_base_return': sum(r['base_return'] for r in records) / 200})
    save_csv(summary_dir / 'validation.csv', validation_rows)
    save_csv(summary_dir / 'comparison.csv', comparison)
    save_csv(summary_dir / 'baseline_episodes.csv', baselines)
    old = read_json(output / 'preflight.json')['historical_file_sha256']
    verify_historical_hashes(ROOT, old)
    checks = {'runs': run_checks, 'validation_summary_rows': len(validation_rows), 'validation_episode_rows': 4400,
              'baseline_episodes': len(baselines), 'baseline_random_rng': 'SeedSequence([314159, scenario_seed]) per episode',
              'baseline_note': 'The comparison uses one new paired baseline evaluation. Baseline summaries written by train.py use a different fixed action stream and are not pooled.',
              'historical_files_unchanged': len(old), 'total_training_interactions': 10000000, 'final_test_used': False}
    save_json(summary_dir / 'checks.json', checks)
    plot_results(summary_dir, validation_rows, comparison)
    write_summary(output, comparison, run_checks)
    print(json.dumps({'comparison': comparison, 'run_checks': run_checks}, indent=2))


def plot_results(output, rows, comparison):
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for mode in MODES:
        selected = [r for r in rows if r['method'] == mode]
        x = [int(r['global_step']) / 1000000 for r in selected]
        for ax, metric, scale in zip(axes, ('success_rate', 'mean_steps', 'mean_misses'), (100, 1, 1)):
            ax.plot(x, [float(r[metric]) * scale for r in selected], marker='o', label=mode.title())
    for ax, label in zip(axes, ('Success within 100 steps (%)', 'Mean steps (all episodes)', 'Mean misses')):
        ax.set_ylabel(label); ax.grid(alpha=.3); ax.legend()
    axes[-1].set_xlabel('Training environment interactions (millions)')
    fig.suptitle('25x25 development validation | training seed 0 | 200 scenarios')
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, scale, label in zip(axes.flat, ('success_rate', 'mean_steps', 'mean_misses', 'mean_base_return'),
                                      (100, 1, 1, 1), ('Success within 100 steps (%)', 'Mean steps (all episodes)', 'Mean misses', 'Mean undiscounted base return')):
        bars = ax.bar([r['method'].title() for r in comparison], [float(r[metric]) * scale for r in comparison],
                      color=['#377eb8', '#ff7f00', '#4daf4a', '#999999'])
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set_ylabel(label); ax.margins(y=.15); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Final development comparison | 200 shared scenarios | no across-seed error bars')
    fig.tight_layout(); fig.savefig(output / 'policy_comparison.png', dpi=150); plt.close(fig)


def write_summary(output, comparison, checks):
    lines = ['# AimRL 25x25 Development Trial', '',
             'This local single-seed comparison uses base and potential Q-learning, each trained from zero for 5,000,000 interactions with seed 0. It is neither a multi-seed formal experiment nor an independent final test.',
             'The grid is 25x25, the cap is 100 steps, gamma is 0.99, alpha is 0.15, and shaping scale is 1.0. Epsilon decays from 1.00 to 0.05 over 4,000,000 interactions. Target move/turn probabilities are 0.25/0.10. Both the cap and gamma differ from the historical 5x5 experiment, so differences cannot be attributed to grid size alone.',
             'Validation uses 200 development scenarios, seeds 3500000â€“3500199, at steps 0 and every 500,000 interactions. Final learned-policy results reuse the last validation records. Rule and random policies are independently evaluated once on the same scenarios. Random actions use SeedSequence([314159, scenario_seed]); the training entry point also writes separate baseline summaries, which are not pooled here.', '',
             '| Policy | Success rate | Mean steps | Mean misses | Mean base return |', '|---|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"| {r['method']} | {100*r['success_rate']:.1f}% | {r['mean_steps']:.3f} | {r['mean_misses']:.3f} | {r['mean_base_return']:.4f} |")
    lines += ['', '## Runtime and Record Checks', '']
    for mode, r in checks.items():
        lines.append(f"- {mode}: {r['global_steps']:,} training interactions; {r['complete_episodes']:,} complete episodes; {r['unfinished_episode']['steps']} unfinished-episode steps; elapsed {r['elapsed_seconds']:.3f} seconds. Reload matched the final checkpoint; the initial checkpoint was zero.")
    lines += ['', 'Elapsed training durations include independent validation, checkpoint and final-model saving, reload verification, baseline evaluation, and log preparation; they are not pure update timings. Each run has 11 checkpoint summaries and 2,200 validation episodes, for 22 summaries and 4,400 validation episodes combined. Q tables are float32 with shape (25,25,25,25,2,6) and 18,750,000 element bytes each.',
              'Success is an exact-cell hit within 100 steps. Mean steps includes timeouts as 100. Mean misses and base return include all episodes. Shaped learning returns are not compared with base returns as performance scores. One training seed and a finite development scenario set do not establish statistical significance or real-game generalization. A 100% scenario success rate would not imply that every shot hit. Interactive GUI behavior has not been verified.', '',
              '## Reproduction Commands', '', 'Run from the project root; choose a new session path if any target run directory is nonempty.', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe -m aimrl.trial25 --stage check',
              '.\\.venv\\Scripts\\python.exe -m aimrl.train --config configs/trial25.json --seed 0 --reward-mode base --output results/grid25_development/base_seed0',
              '.\\.venv\\Scripts\\python.exe -m aimrl.train --config configs/trial25.json --seed 0 --reward-mode potential --output results/grid25_development/potential_seed0',
              '.\\.venv\\Scripts\\python.exe -m aimrl.trial25 --stage summarize',
              '.\\.venv\\Scripts\\python.exe -m aimrl.trial25 --stage demo', '```', '',
              'Raw run data are in base_seed0/ and potential_seed0/; figures and comparison.csv are in summary/. Real trajectories for demo seed 3501000 and export metadata are in demo/. No 50x50 training, historical final-test evaluation, or GitHub upload is part of this trial.']
    (output / 'summary/results_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def trajectory(config, policy, name, seed=DEMO_SEED):
    """Record actual transitions, retaining the terminal or truncated state."""
    env = AimEnv(config)
    state, _ = env.reset(seed=seed)
    def row(step, action, reward, total, misses, terminated, truncated):
        return {'policy': name, 'scenario_seed': seed, 'step_count': step,
                'xc': state[0], 'yc': state[1], 'xt': state[2], 'yt': state[3], 'v': state[4],
                'action': 'reset' if action is None else config['actions'][action],
                'base_reward': reward, 'base_return': total, 'misses': misses,
                'terminated': terminated, 'truncated': truncated,
                'status': 'Hit' if terminated else 'Timeout' if truncated else 'Ready' if step == 0 else 'In progress'}
    rows = [row(0, None, 0.0, 0.0, 0, False, False)]
    misses, total = 0, 0.0
    while True:
        action = policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        total += reward; misses += int(info['miss'])
        rows.append(row(info['step_count'], action, reward, total, misses, terminated, truncated))
        if terminated or truncated:
            break
    verify_episode(rows[-1], 'step_count', 'terminated')
    return rows


def draw_demo_panel(ax, row):
    """Draw a recorded state without advancing an environment or random stream."""
    ax.clear()
    ax.set_xlim(0, 25); ax.set_ylim(25, 0); ax.set_aspect('equal')
    coordinates = [0, 5, 10, 15, 20, 24]
    ax.set_xticks([x + .5 for x in coordinates], labels=coordinates)
    ax.set_yticks([y + .5 for y in coordinates], labels=coordinates)
    ax.set_xticks(range(26), minor=True); ax.set_yticks(range(26), minor=True)
    ax.grid(which='minor', color='#d5dce5', linewidth=.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.scatter(row['xt'] + .5, row['yt'] + .5, s=110, facecolors='none', edgecolors='#d94f39', linewidths=2, label='Target', zorder=4)
    ax.scatter(row['xc'] + .5, row['yc'] + .5, s=70, marker='x', color='#1766b1', linewidths=2, label='Crosshair', zorder=5)
    ax.set_xlabel('x (right increases)'); ax.set_ylabel('y (down increases)')
    ax.set_title(f"{row['policy']}\nStep {row['step_count']}/100 | Misses {row['misses']} | {row['status']}\n"
                 f"Action: {row['action']} | Base return: {row['base_return']:+.2f}", fontsize=10)


def export_demo(config, output):
    validate_config(config)
    destination = output / 'demo'
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Demo output already exists; preserve existing exports')
    destination.mkdir(exist_ok=True)
    paths, checks = {}, {}
    for mode in MODES:
        run = output / f'{mode}_seed0'
        before = sha(run / 'q_table.npy')
        agent = load_agent(run, config, mode)
        memory_hash = hashlib.sha256(memoryview(agent.q)).hexdigest()
        paths[mode] = trajectory(config, agent.greedy_action, f'{mode.title()} Q-learning')
        if memory_hash != hashlib.sha256(memoryview(agent.q)).hexdigest() or sha(run / 'q_table.npy') != before:
            raise RuntimeError('Demo modified the learned model')
        checks[mode] = {'model_sha256': before, 'q_unchanged': True, 'steps': paths[mode][-1]['step_count'],
                        'status': paths[mode][-1]['status'], 'misses': paths[mode][-1]['misses']}
        del agent
    paths['rule'] = trajectory(config, rule_policy, 'Rule policy')
    initial = [(rows[0]['xc'], rows[0]['yc'], rows[0]['xt'], rows[0]['yt'], rows[0]['v']) for rows in paths.values()]
    if len(set(initial)) != 1:
        raise RuntimeError('Demo policies did not start from the same scenario')
    save_csv(destination / 'trajectories.csv', [r for rows in paths.values() for r in rows])
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.3), dpi=100)
    fig.subplots_adjust(left=.05, right=.985, bottom=.18, top=.77, wspace=.28)
    fig.suptitle('25x25 development demo | scenario seed 3501000 | learned policies after 5,000,000 interactions', fontsize=12, y=.98)
    frame_count = max(len(rows) for rows in paths.values())
    frames = []
    try:
        for index in range(frame_count):
            for ax, rows in zip(axes, paths.values()):
                draw_demo_panel(ax, rows[min(index, len(rows) - 1)])
            if index == 0:
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .02))
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frame = Image.fromarray(pixels).copy()
            frames.append(frame)
            if index == 0:
                frame.save(destination / 'demo_start.png')
            if index == frame_count // 2:
                frame.save(destination / 'demo_middle.png')
            if index == frame_count - 1:
                frame.save(destination / 'demo_preview.png')
        frames[0].save(destination / 'demo_comparison.gif', save_all=True, append_images=frames[1:],
                       duration=[120] * (len(frames) - 1) + [1500], loop=0, optimize=False, disposal=2)
    finally:
        plt.close(fig)
        for frame in frames:
            frame.close()
    with Image.open(destination / 'demo_comparison.gif') as exported:
        if exported.n_frames != frame_count or exported.size != (1260, 530):
            raise RuntimeError('GIF export frame count or dimensions are inconsistent')
        for index, name in ((0, 'gif_start.png'), (frame_count // 2, 'gif_middle.png'), (frame_count - 1, 'gif_final.png')):
            exported.seek(index)
            exported.convert('RGB').save(destination / name)
    save_json(destination / 'metadata.json', {'scenario_seed': DEMO_SEED, 'frames': frame_count,
              'frame_interval_ms': 120, 'last_frame_hold_ms': 1500, 'writer': 'Pillow ' + importlib.metadata.version('pillow'),
              'added_dependencies': [], 'dimensions': [1260, 530], 'finished_panel_behavior': 'Hold the actual final observed state',
              'policies': checks, 'rule_final': paths['rule'][-1], 'gui_verified': False})
    print(f'Exported {frame_count} real-trajectory GIF frames; model arrays and files unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/trial25.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/grid25_development')
    parser.add_argument('--stage', choices=('check', 'summarize', 'demo'), required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.stage == 'check': preflight(config, args.output)
    elif args.stage == 'summarize': summarize(config, args.output)
    else: export_demo(config, args.output)


if __name__ == '__main__':
    main()
