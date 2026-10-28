"""Isolated greedy evaluation; target trajectory hashes support direct pairing checks."""
import hashlib

import numpy as np

from .env import TrackingEnv
from .metrics import EpisodeMetrics, aggregate
from .policies import rule_policy


def evaluate(config, method, agent=None, seeds=None, keep_trace=False):
    settings = config['experiments']
    if seeds is None:
        seeds = range(settings['validation_seed_start'], settings['validation_seed_start'] + settings['validation_episodes'])
    before = hashlib.sha256(memoryview(agent.q)).hexdigest() if agent is not None else None
    env = TrackingEnv(config)
    rows, traces = [], []
    for seed in seeds:
        state, _ = env.reset(seed=int(seed))
        metrics = EpisodeMetrics(state)
        rng = np.random.default_rng(np.random.SeedSequence([314159, int(seed)])) if method == 'random' else None
        if keep_trace:
            traces.append(trace_row(method, seed, 0, state, None, None, 0, 0))
        while True:
            if method == 'rule': action = rule_policy(state, env.size)
            elif method == 'random': action = int(rng.integers(10))
            else: action = agent.greedy(state)
            state, reward, terminated, truncated, info = env.step(action)
            metrics.add(action, state, reward, info)
            if keep_trace:
                traces.append(trace_row(method, seed, info['elapsed_ticks'], state, action, info, metrics.shots, metrics.hits))
            if terminated: raise RuntimeError('Tracking environment unexpectedly terminated')
            if truncated: break
        rows.append({'scenario_seed': int(seed), **metrics.finish()})
    if agent is not None and hashlib.sha256(memoryview(agent.q)).hexdigest() != before:
        raise RuntimeError('Validation changed Q values')
    return aggregate(rows), rows, traces


def trace_row(method, seed, tick, state, action, info, shots, hits):
    return {'method': method, 'scenario_seed': int(seed), 'tick': tick, 'xc': state[0], 'yc': state[1],
            'xt': state[2], 'yt': state[3], 'heading': state[4], 'target_speed': state[5],
            'action': -1 if action is None else action, 'movement_mode': 'reset' if info is None else info['movement_mode'],
            'distance': abs(state[0] - state[2]) + abs(state[1] - state[3]), 'shots': shots, 'hits': hits,
            'hit': False if info is None else info['hit'], 'shot_x': '' if info is None or info['shot'] is None else info['shot'][0],
            'shot_y': '' if info is None or info['shot'] is None else info['shot'][1],
            'displacement': 0 if info is None else info['actual_displacement'], 'target_path_distance': 0 if info is None else info['target_path_distance'],
            'random_turn': False if info is None else info['random_turn'], 'speed_toggle': False if info is None else info['speed_toggle'],
            'reflections': 0 if info is None else info['reflections']}
