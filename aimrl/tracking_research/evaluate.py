"""Read-only paired evaluations with exact histograms and denominator-bearing diagnostics."""
import json
import math

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking25.evaluate import trace_row
from ..tracking25.metrics import EpisodeMetrics, aggregate
from ..tracking25.policies import rule_policy

FIELDS = ('ticks', 'distance_sum', 'close_ticks', 'far_ticks', 'movements', 'slow', 'fast',
          'zero_displacement_moves', 'local_pursuit_moves', 'shots', 'unaligned_shots',
          'all_zero_q_rows', 'exact_tied_maxima', 'near_tied_maxima', 'zero_visit_rows',
          'unvisited_greedy_actions', 'learned_ticks', 'counted_ticks')


def distance_p95(histogram):
    n = sum(histogram)
    if not n: return None
    threshold = math.ceil(.95 * n)
    cumulative = 0
    for i, count in enumerate(histogram):
        cumulative += count
        if cumulative >= threshold: return i


def longest_far(distances):
    longest = current = 0
    for distance in distances:
        current = current + 1 if distance > 5 else 0
        longest = max(longest, current)
    return longest


def contexts(state):
    def near(x, y): return min(x, y, 24 - x, 24 - y) <= 3
    cross, target = near(*state[:2]), near(*state[2:4])
    d = abs(state[0] - state[2]) + abs(state[1] - state[3])
    return ('all', 'boundary_near' if cross or target else 'interior',
            'target_near' if target else 'target_interior', 'crosshair_near' if cross else 'crosshair_interior',
            f'speed_{state[5]}', 'distance_0_3' if d <= 3 else 'distance_4_11' if d <= 11 else 'distance_12_plus')


def slice_summary(values, hist):
    result = dict(zip(FIELDS, values)); ticks = result['ticks']
    result.update({'distance_histogram': hist, 'distance_p95': distance_p95(hist)})
    for out, field, denom in [('mean_distance', 'distance_sum', ticks), ('close_fraction', 'close_ticks', ticks),
            ('far_fraction', 'far_ticks', ticks), ('clipped_move_fraction', 'zero_displacement_moves', result['movements']),
            ('local_pursuit_fraction', 'local_pursuit_moves', result['movements']),
            ('fast_movement_fraction', 'fast', result['movements']),
            ('all_zero_q_fraction', 'all_zero_q_rows', result['learned_ticks']),
            ('exact_tie_fraction', 'exact_tied_maxima', result['learned_ticks']),
            ('numerical_near_tie_fraction', 'near_tied_maxima', result['learned_ticks']),
            ('zero_visit_row_fraction', 'zero_visit_rows', result['counted_ticks']),
            ('unvisited_greedy_action_fraction', 'unvisited_greedy_actions', result['counted_ticks'])]:
        result[out] = result[field] / denom if denom else None
    return result


def evaluate(config, policy, seeds, agent=None, keep_trace=False):
    before = agent.snapshot() if agent is not None else None
    env = TrackingEnv(config)
    rows, traces, slices = [], [], {}
    for seed in seeds:
        seed = int(seed); state, _ = env.reset(seed)
        metrics = EpisodeMetrics(state)
        rng = np.random.default_rng(np.random.SeedSequence([314159, seed]))
        histogram, action_counts = [0] * 49, [0] * 10
        totals = [0] * len(FIELDS)
        if keep_trace:
            initial = trace_row(policy, seed, 0, state, None, None, 0, 0)
            initial.update({'pre_xc': state[0], 'pre_yc': state[1], 'pre_xt': state[2], 'pre_yt': state[3]})
            traces.append(initial)
        for tick in range(1, 301):
            encoded = agent.index(state) if agent is not None else None
            if agent is not None:
                q = agent.q[encoded]; action = int(np.argmax(q)); maximum = q[action]
                zero_q = bool(np.all(q == 0)); tied = np.count_nonzero(q == maximum) > 1
                near_tied = np.count_nonzero(np.abs(q - maximum) <= 1e-6) > 1
                count_row = agent.counts[encoded] if agent.counts is not None else None
                zero_visit = bool(not np.any(count_row)) if count_row is not None else False
                greedy_unvisited = bool(count_row[action] == 0) if count_row is not None else False
            else:
                action = rule_policy(state, env.size) if policy == 'rule' else int(rng.integers(10))
                zero_q = tied = near_tied = zero_visit = greedy_unvisited = False
                count_row = None
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated != (tick == 300): raise RuntimeError('Unexpected environment lifecycle')
            metrics.add(action, state, reward, info)
            d = info['distance']; histogram[d] += 1; action_counts[action] += 1
            move = action <= 3 or action >= 6
            local = abs(state[0] - previous[2]) + abs(state[1] - previous[3]) < info['pre_distance']
            values = [1, d, int(d <= 1), int(d > 5), int(move), int(action <= 3), int(action >= 6),
                      int(move and info['actual_displacement'] == 0), int(move and local), int(action == 5),
                      int(action == 5 and info['pre_distance'] != 0), int(zero_q), int(tied), int(near_tied),
                      int(zero_visit), int(greedy_unvisited), int(agent is not None), int(count_row is not None)]
            for i, value in enumerate(values): totals[i] += value
            for key in contexts(previous):
                if key not in slices: slices[key] = ([0] * len(FIELDS), [0] * 49)
                counters, hist = slices[key]; hist[d] += 1
                for i, value in enumerate(values): counters[i] += value
            if keep_trace:
                row = trace_row(policy, seed, tick, state, action, info, metrics.shots, metrics.hits)
                row.update({'pre_xc': previous[0], 'pre_yc': previous[1], 'pre_xt': previous[2], 'pre_yt': previous[3]})
                traces.append(row)
        row = {'scenario_seed': seed, **metrics.finish(), 'far_ticks': totals[3],
               'longest_far_streak': longest_far(metrics.distance), 'distance_p95': distance_p95(histogram),
               'distance_histogram': json.dumps(histogram), 'action_counts': json.dumps(action_counts)}
        for key, value in zip(FIELDS[7:], totals[7:]): row[key] = value
        rows.append(row)
    summary = aggregate(rows)
    hist = [sum(json.loads(r['distance_histogram'])[i] for r in rows) for i in range(49)]
    streaks = [r['longest_far_streak'] for r in rows]
    summary.update({'far_fraction': sum(r['far_ticks'] for r in rows) / summary['ticks'],
                    'distance_p95': distance_p95(hist), 'distance_histogram': hist,
                    'mean_longest_far_streak': float(np.mean(streaks)),
                    'window_p95_longest_far_streak': int(np.quantile(streaks, .95, method='inverted_cdf')),
                    'action_counts': [sum(json.loads(r['action_counts'])[i] for r in rows) for i in range(10)],
                    'slices': {key: slice_summary(*value) for key, value in slices.items()},
                    'near_tie_absolute_tolerance': 1e-6, 'initial_frame_excluded': True})
    if agent is not None and agent.snapshot() != before:
        raise RuntimeError('Evaluation changed Q, counts, or exploration RNG')
    return summary, rows, traces
