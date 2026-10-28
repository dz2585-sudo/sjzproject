"""Staged tracking experiment with streaming logs and protected completed outputs."""
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..model_io import artifact_digest, resolve_artifact
from ..provenance import source_hash_matches, verify_historical_hashes

from .agent import TrackingAgent, epsilon, exact_reload_check, learning_reward
from .env import TrackingEnv
from .evaluate import evaluate
from .metrics import EpisodeMetrics

ROOT = Path(__file__).resolve().parents[2]


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
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return artifact_digest(path)


def source_hashes():
    files = list((ROOT / 'aimrl/tracking25').glob('*.py')) + [ROOT / 'configs/tracking25_v2.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def historical_hashes():
    paths = [p for p in (ROOT / 'results').rglob('*') if p.is_file() and p.relative_to(ROOT / 'results').parts[0] != 'continuous_tracking']
    paths += list((ROOT / 'configs').glob('*.json'))
    paths = [p for p in paths if p.name != 'tracking25_v2.json']
    paths += [p for p in (ROOT / 'aimrl').glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def check(config, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        verify_protocol(config, output)
        print('Existing protocol verified; no preflight overwritten.')
        return
    if config['environment'] != {'grid_size': 25, 'minimum_reset_distance': 12, 'turn_probability': .05, 'speed_toggle_probability': .05, 'episode_ticks': 300}:
        raise ValueError('Unexpected V2 environment settings')
    if config['learning'] != {'alpha': .15, 'gamma': .99, 'initial_q': 0, 'dtype': 'float32', 'training_steps': 20000000, 'epsilon_start': 1.0, 'epsilon_end': .05, 'epsilon_decay_steps': 16000000, 'shaping_scale': 1.0}:
        raise ValueError('Unexpected V2 learning settings')
    if config['experiments'] != {'methods': ['base', 'potential'], 'training_seed': 0, 'validation_interval': 2000000, 'validation_seed_start': 3600000, 'validation_episodes': 200}:
        raise ValueError('Unexpected development evaluation settings')
    if shutil.disk_usage(output).free < 1000000000:
        raise RuntimeError('At least 1 GB free space is required for final models, logs and exports')
    agent = TrackingAgent(config)
    if agent.q.shape != (25, 25, 25, 25, 4, 2, 10) or agent.q.nbytes != 125000000 or np.any(agent.q):
        raise RuntimeError('Tracking Q table initialization or resource budget mismatch')
    del agent
    selected = []
    env = TrackingEnv(config)
    for seed in range(3601000, 3601100):
        state, _ = env.reset(seed=seed)
        xs, ys, speeds = [state[2]], [state[3]], []
        for _ in range(300):
            state, _, _, _, _ = env.step(4)
            xs.append(state[2]); ys.append(state[3]); speeds.append(state[5])
        row = {'scenario_seed': seed, 'x_span': max(xs) - min(xs), 'y_span': max(ys) - min(ys),
               'speed_1_ticks': speeds.count(1), 'speed_2_ticks': speeds.count(2)}
        if row['x_span'] >= 16 and row['y_span'] >= 16 and min(row['speed_1_ticks'], row['speed_2_ticks']) >= 50:
            selected.append(row)
            if len(selected) == 3: break
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'source_sha256': source_hashes(),
                'historical_sha256': historical_hashes(), 'selected_demos': selected,
                'demo_selection': 'First three qualifying seeds in 3601000â€“3601099: target x/y spans >=16; >=50 ticks at each speed; stay policy only. Initial target position included in spans.',
                'training_rng': 'SeedSequence(0).spawn(2): first child generate_state(1, uint64) seeds environment; second child drives action exploration.',
                'random_baseline_rng': 'Independent per scenario SeedSequence([314159, scenario_seed])',
                'platform': platform.platform(), 'python': sys.version, 'interpreter': sys.executable,
                'numpy': np.__version__, 'matplotlib': importlib.metadata.version('matplotlib'), 'pillow': importlib.metadata.version('pillow'),
                'q_shape': [25, 25, 25, 25, 4, 2, 10], 'q_element_bytes': 125000000,
                'disk_free_bytes': shutil.disk_usage(output).free, 'gui_verified': False}
    save_json(output / 'protocol.json', protocol)
    print('Protocol frozen before training; selected demos:', selected, flush=True)


def verify_protocol(config, output):
    protocol = read_json(output / 'protocol.json')
    if config != protocol['config'] or any(not source_hash_matches(ROOT, name, expected) for name, expected in protocol['source_sha256'].items()):
        raise RuntimeError('Source or configuration differs from the recorded V2 protocol')
    return protocol


def train_run(config, mode, run, validations=True):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        if (run / 'metadata.json').exists():
            meta = verify_run(config, mode, run)
            print(f'Reusing verified completed {mode} run: {run}', flush=True)
            return meta
        raise FileExistsError(f'Incomplete run retained at {run}; use a new session, not interrupted-state resume')
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / 'run_config.json', config)
    learning, exp = config['learning'], config['experiments']
    env_seq, explore_seq = np.random.SeedSequence(exp['training_seed']).spawn(2)
    env_seed = int(env_seq.generate_state(1, dtype=np.uint64)[0])
    env = TrackingEnv(config); state, _ = env.reset(seed=env_seed)
    agent = TrackingAgent(config, np.random.default_rng(explore_seq))
    metrics = EpisodeMetrics(state)
    started = time.perf_counter()
    validation_rows, episode_rows = [], []
    complete_windows, complete_ticks, learning_total, shaping_total = 0, 0, 0.0, 0.0

    def checkpoint(t):
        summary, rows, _ = evaluate(config, mode, agent)
        validation_rows.append({'global_step': t, **summary})
        episode_rows.extend({'global_step': t, **r} for r in rows)
        print(f'{mode} step {t:,}: distance={summary["mean_distance"]:.3f}, close={summary["close_fraction"]:.3%}, hits/window={summary["hits_per_episode"]:.3f}, accuracy={summary["pooled_shot_accuracy"]}', flush=True)

    if validations: checkpoint(0)
    with (run / 'training_episodes.csv').open('w', encoding='utf-8', newline='') as file:
        writer = None
        for t in range(learning['training_steps']):
            action = agent.select(state, epsilon(t, learning))
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated: raise RuntimeError('Unexpected true termination in continuing tracking')
            learned, shaping = learning_reward(reward, state, next_state, mode, learning['gamma'])
            agent.update(state, action, learned, next_state)
            metrics.add(action, next_state, reward, info)
            learning_total += learned; shaping_total += shaping
            state = next_state
            if truncated:
                row = {'end_step': t + 1, 'terminated': False, 'truncated': True, 'partial': False,
                       'learning_return': learning_total, 'shaping_return': shaping_total, **metrics.finish()}
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                complete_windows += 1; complete_ticks += env.elapsed
                state, _ = env.reset()
                metrics = EpisodeMetrics(state)
                learning_total, shaping_total = 0.0, 0.0
            if validations and (t + 1) % exp['validation_interval'] == 0:
                file.flush(); checkpoint(t + 1)
        partial = None
        if env.elapsed:
            partial = {'end_step': learning['training_steps'], 'terminated': False, 'truncated': False, 'partial': True,
                       'learning_return': learning_total, 'shaping_return': shaping_total, **metrics.finish()}
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(partial)); writer.writeheader()
            writer.writerow(partial)
    agent.save(run / 'q_table.npy')
    if not exact_reload_check(agent, run / 'q_table.npy'):
        raise RuntimeError('Exact final-model reload check failed')
    if validations:
        save_csv(run / 'validation.csv', validation_rows)
        save_csv(run / 'validation_episodes.csv', episode_rows)
    meta = {'schema': 'tracking25_v2', 'reward_mode': mode, 'seed': exp['training_seed'], 'global_steps': learning['training_steps'],
            'complete_windows': complete_windows, 'complete_window_ticks': complete_ticks, 'partial_window': partial,
            'environment_initialization_seed': env_seed, 'rng_derivation': 'SeedSequence(root).spawn(2): environment and exploration',
            'elapsed_seconds': time.perf_counter() - started, 'timing_scope': 'Training, checkpoint validation, streamed logs, final save and chunked exact reload verification',
            'q_shape': list(agent.q.shape), 'q_dtype': str(agent.q.dtype), 'q_nbytes': agent.q.nbytes,
            'model_sha256': sha(run / 'q_table.npy'), 'source_sha256': source_hashes(), 'reload_exact': True,
            'validation_isolated': True, 'greedy_tie_break': 'lowest action ID in both training and evaluation',
            'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}
    save_json(run / 'metadata.json', meta)
    del agent
    print(f'{mode} completed {meta["global_steps"]:,} interactions in {meta["elapsed_seconds"]:.3f}s; exact reload verified.', flush=True)
    return meta


def verify_run(config, mode, run):
    meta = read_json(run / 'metadata.json')
    if read_json(run / 'run_config.json') != config or meta['reward_mode'] != mode or meta['global_steps'] != config['learning']['training_steps']:
        raise ValueError('Completed run configuration or interaction count mismatch')
    if sha(run / 'q_table.npy') != meta['model_sha256']:
        raise ValueError('Saved model hash mismatch')
    total, windows, partials = 0, 0, 0
    with (run / 'training_episodes.csv').open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            steps = int(row['steps']); total += steps
            if int(row['end_step']) != total or row['terminated'] != 'False': raise ValueError('Training step accounting error')
            is_partial = row['partial'] == 'True'
            if is_partial:
                partials += 1
                if row['truncated'] != 'False': raise ValueError('Partial budget window falsely truncated')
            else:
                windows += 1
                if steps != config['environment']['episode_ticks'] or row['truncated'] != 'True': raise ValueError('Incomplete training window')
            identity = 10 * int(row['hit_count']) - .05 * steps - int(row['miss_count'])
            if not math.isclose(float(row['base_return']), identity, abs_tol=1e-7): raise ValueError('Training base-return identity failed')
            if not math.isclose(float(row['learning_return']), float(row['base_return']) + float(row['shaping_return']), abs_tol=1e-7): raise ValueError('Training reward accounting failed')
    if total != config['learning']['training_steps'] or windows != meta['complete_windows'] or partials != int(meta['partial_window'] is not None):
        raise ValueError('Training budget or window count mismatch')
    return meta


def train_stage(config, output):
    protocol = verify_protocol(config, output)
    for mode in ('base', 'potential'):
        train_run(config, mode, output / mode)
    verify_historical_hashes(ROOT, protocol['historical_sha256'])


def evaluate_stage(config, output):
    protocol = verify_protocol(config, output)
    destination = output / 'evaluation'
    if (destination / 'checks.json').exists():
        print('Verified evaluation artifacts already present; no scenarios reevaluated.')
        return
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Partial evaluation retained; use new output directory')
    destination.mkdir(exist_ok=True)
    comparisons, final_rows, all_validations, all_validation_episodes, demo_traces = [], [], [], [], []
    pairing, run_metadata = {}, {}
    for mode in ('base', 'potential'):
        run = output / mode
        run_metadata[mode] = verify_run(config, mode, run)
        rows, records = read_csv(run / 'validation.csv'), read_csv(run / 'validation_episodes.csv')
        steps = list(range(0, 20000001, 2000000))
        if [int(r['global_step']) for r in rows] != steps or len(records) != 2200: raise ValueError('Validation counts are incomplete')
        keys = {(int(r['global_step']), int(r['scenario_seed'])) for r in records}
        if keys != {(t, s) for t in steps for s in range(3600000, 3600200)}: raise ValueError('Validation scenario keys are incomplete')
        for r in records:
            if int(r['steps']) != 300: raise ValueError('Validation stopped before 300 ticks')
            key = int(r['scenario_seed'])
            if key in pairing and pairing[key] != r['target_trajectory_sha256']: raise ValueError('Target trajectories differ across methods or checkpoints')
            pairing[key] = r['target_trajectory_sha256']
            if not math.isclose(float(r['base_return']), 10 * float(r['hit_count']) - .05 * 300 - float(r['miss_count']), abs_tol=1e-7): raise ValueError('Validation base-return identity failed')
        comparisons.append({'method': mode, **rows[-1]})
        final_rows.extend({'method': mode, **r} for r in records if int(r['global_step']) == 20000000)
        all_validations.extend({'method': mode, **r} for r in rows)
        all_validation_episodes.extend({'method': mode, **r} for r in records)
        agent = TrackingAgent(config); agent.load(run / 'q_table.npy')
        _, _, traces = evaluate(config, mode, agent, [r['scenario_seed'] for r in protocol['selected_demos']], keep_trace=True)
        demo_traces.extend(traces)
        del agent
    baseline_rows = []
    for mode in ('rule', 'random'):
        summary, rows, _ = evaluate(config, mode)
        comparisons.append({'method': mode, 'global_step': '', **summary})
        baseline_rows.extend({'method': mode, **r} for r in rows)
        for r in rows:
            if pairing[r['scenario_seed']] != r['target_trajectory_sha256']: raise ValueError('Baseline target trajectories are not paired')
        _, _, traces = evaluate(config, mode, seeds=[r['scenario_seed'] for r in protocol['selected_demos']], keep_trace=True)
        demo_traces.extend(traces)
    training_pairs = 0
    with (output / 'base/training_episodes.csv').open(newline='') as a, (output / 'potential/training_episodes.csv').open(newline='') as b:
        for left, right in zip(csv.DictReader(a), csv.DictReader(b), strict=True):
            if left['target_trajectory_sha256'] != right['target_trajectory_sha256'] or left['end_step'] != right['end_step']:
                raise ValueError('Training target trajectories or window boundaries differ across methods')
            training_pairs += 1
    # Demonstrations must share target paths too, including their initial observations.
    for selection in protocol['selected_demos']:
        seed = selection['scenario_seed']
        tracks = [[(r['xt'], r['yt'], r['heading'], r['target_speed']) for r in demo_traces if r['method'] == mode and r['scenario_seed'] == seed] for mode in ('base', 'potential', 'rule', 'random')]
        if not all(len(t) == 301 and t == tracks[0] for t in tracks): raise ValueError('Demo targets differ by policy')
    save_csv(destination / 'comparison.csv', comparisons)
    save_csv(destination / 'final_episodes.csv', final_rows)
    save_csv(destination / 'baseline_episodes.csv', baseline_rows)
    save_csv(destination / 'validation.csv', all_validations)
    save_csv(destination / 'validation_episodes.csv', all_validation_episodes)
    save_csv(destination / 'demo_traces.csv', demo_traces)
    checks = {'training_interactions': 40000000, 'validation_summaries': len(all_validations), 'validation_episode_records': len(all_validation_episodes),
              'validation_interactions': len(all_validation_episodes) * 300, 'baseline_episode_records': len(baseline_rows), 'baseline_interactions': len(baseline_rows) * 300,
              'evaluation_target_hashes_paired': len(pairing), 'demo_target_paths_paired': True, 'run_metadata': run_metadata,
              'training_window_target_hashes_paired': training_pairs,
              'historical_files_preserved': bool(verify_historical_hashes(ROOT, protocol['historical_sha256'])), 'gui_verified': False}
    save_json(destination / 'checks.json', checks)
    print('Evaluation and sequential demo recording complete; target paths paired across all policies.', flush=True)
