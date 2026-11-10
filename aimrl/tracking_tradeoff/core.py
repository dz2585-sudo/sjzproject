"""Native rollout adapter and common window accounting."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking_research.agent import INDEXING_VERSION, ResearchAgent, array_hash

STATE = ('xc', 'yc', 'xt', 'yt', 'heading', 'speed')
COLUMNS = (*STATE, 'action', 'd_pre', 'd_before_target_motion', 'd_post', 'target_net_displacement',
           'hit', 'miss', 'reflections', 'turn', 'toggle')
MODES = ('slow', 'fast', 'stay', 'fire')
MEASURE_KEYS = ('ticks', 'distance_sum', 'close_ticks', 'hits', 'shots', 'misses')
MODE_ID = np.array([0, 0, 0, 0, 2, 3, 1, 1, 1, 1])


def reward_parts(base, d_pre, d_post, weights, gamma=0.99):
    distance = -weights[0] * d_post
    close = weights[1] * (d_post <= 1)
    potential = gamma * -d_post + d_pre
    return base, distance, close, potential, ((base + distance) + close) + potential


class Kernel:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        ptr = ctypes.c_void_p
        self.call = self.library.rollout
        self.call.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ptr, ptr, ptr, *[ctypes.c_double] * 4, ptr]
        self.call.restype = None

    def run(self, env, state, n, agent=None, offset=0, weights=(0, 0), train=False):
        initial = np.asarray(state, dtype=np.int32)
        draws = env.rng.random((n, 3))
        out = np.empty((n, len(COLUMNS)), dtype=np.int16)
        mode = 1 if train else 2 if agent is None else 0
        rng = agent.rng.bit_generator.ctypes if train else None
        self.call(agent.q.ctypes.data if agent else None, agent.counts.ctypes.data if train else None,
                  initial.ctypes.data, draws.ctypes.data, n, offset, mode, rng.state if rng else None,
                  ctypes.cast(rng.next_double, ctypes.c_void_p) if rng else None,
                  ctypes.cast(rng.next_uint32, ctypes.c_void_p) if rng else None,
                  *weights, env.turn_probability, env.toggle_probability, out.ctypes.data)
        return out


def target_hash(initial, ticks):
    h = hashlib.sha256(bytes(initial[2:]))
    h.update(ticks[:, 2:6].astype(np.uint8).tobytes())
    return h.hexdigest()


def measures(distance, hit, shots):
    hits, fired = int(hit.sum()), int(shots.sum())
    return (len(distance), int(distance.sum()), int(np.count_nonzero(distance <= 1)),
            hits, fired, fired - hits)


def window_metrics(initial, ticks, detailed=False, boundary_cells=3):
    action, pre, post, net, hit = ticks[:, [6, 7, 9, 10, 11]].T
    shots = action == 5
    row = dict(zip(MEASURE_KEYS, measures(post, hit, shots)))
    close_indices = np.flatnonzero(post <= 1)
    streak_edges = np.flatnonzero(np.diff(np.r_[False, post > 5, False]))
    row.update(base_return=-0.05 * len(ticks) + 10 * row['hits'] - row['misses'],
               first_close_tick=int(close_indices[0]) + 1 if close_indices.size else 300,
               acquired=int(bool(close_indices.size)),
               longest_far_streak=int(np.max(streak_edges[1::2] - streak_edges[::2], initial=0)),
               alignment_opportunities=int(np.count_nonzero(pre == 0)),
               aligned_shots=int(np.count_nonzero(shots & (pre == 0))),
               unaligned_shots=int(np.count_nonzero(shots & (pre != 0))),
               hit_nonclose=int(np.count_nonzero((hit != 0) & (net > 1))),
               target_sha256=target_hash(initial, ticks))
    if np.any(post[hit != 0] != net[hit != 0]) or row['close_ticks'] + row['hit_nonclose'] > len(ticks):
        raise ValueError('Hit distance / disjoint close accounting failed')
    if row['hits'] != row['aligned_shots'] or row['misses'] != row['unaligned_shots']:
        raise ValueError('Pre-motion shot accounting failed')
    if detailed:
        action_modes = MODE_ID[action]
        row['hit_post_histogram'] = json.dumps(np.bincount(post[hit != 0], minlength=3).tolist())
        row['pre_distance_action_counts'] = json.dumps(
            np.bincount(np.minimum(pre, 3) * 4 + action_modes, minlength=16).reshape(4, 4).tolist())
        pre_positions = np.vstack([initial[:4], ticks[:-1, :4]])
        boundary = np.any(np.minimum(pre_positions, 24 - pre_positions) <= boundary_cells, axis=1)
        groups = {'speed_1': ticks[:, 5] == 1, 'speed_2': ticks[:, 5] == 2,
                  'boundary': boundary, 'interior': ~boundary}
        row['slice_counts'] = json.dumps({name: measures(post[m], hit[m], shots[m]) for name, m in groups.items()})
    return row


def summarize(rows):
    def total(key):
        return sum(row[key] for row in rows)

    ticks, shots, hits = (total(key) for key in ('ticks', 'shots', 'hits'))
    opportunities = total('alignment_opportunities')
    return dict(D=total('distance_sum') / ticks, C=total('close_ticks') / ticks, H=hits / len(rows),
                A=hits / shots if shots else None, shots=shots / len(rows),
                misses=total('misses') / len(rows), base_return=total('base_return') / len(rows),
                longest_far_streak=total('longest_far_streak') / len(rows),
                first_close_tick=total('first_close_tick') / len(rows),
                acquired_fraction=total('acquired') / len(rows),
                alignment_opportunity_fraction=opportunities / ticks,
                fire_given_alignment=total('aligned_shots') / opportunities if opportunities else None,
                hit_nonclose_fraction=total('hit_nonclose') / hits if hits else None,
                ticks=ticks, windows=len(rows), total_shots=shots, total_hits=hits,
                total_misses=total('misses'), total_unaligned_shots=total('unaligned_shots'),
                total_aligned_shots=total('aligned_shots'), total_alignment_opportunities=opportunities,
                total_close_ticks=total('close_ticks'), total_hit_nonclose=total('hit_nonclose'))


def evaluate(config, kernel, agent, seeds, detailed=False, keep_traces=False):
    before = agent.snapshot() if agent is not None else None
    env = TrackingEnv(config)
    rows, traces = [], {}
    for seed in seeds:
        initial, _ = env.reset(seed)
        ticks = kernel.run(env, initial, 300, agent)
        rows.append({'scenario_seed': seed,
                     **window_metrics(initial, ticks, detailed, config['tradeoff']['boundary_cells'])})
        if keep_traces:
            traces[seed] = (initial, ticks)
    if agent is not None and before != agent.snapshot():
        raise ValueError('Evaluation changed Q, counts or training RNG')
    return rows, traces


def save_model(path, agent, config, condition, seed, step):
    metadata = {'condition': condition, 'weights': config['tradeoff']['conditions'][condition],
                'seed': seed, 'step': step, 'config': config, 'representation': 'relative',
                'indexing_version': INDEXING_VERSION, 'q_sha256': array_hash(agent.q),
                'counts_sha256': array_hash(agent.counts)}
    np.savez_compressed(path, q=agent.q, counts=agent.counts, metadata=json.dumps(metadata, sort_keys=True))


def load_model(path, config, condition, seed, step):
    agent = ResearchAgent(config, 'relative')
    with np.load(path, allow_pickle=False) as package:
        m = json.loads(str(package['metadata']))
        identity = (m['condition'], m['weights'], m['seed'], m['step'], m['config'],
                    m['representation'], m['indexing_version'])
        if identity != (condition, config['tradeoff']['conditions'][condition], seed, step,
                        config, 'relative', INDEXING_VERSION):
            raise ValueError(f'Model identity mismatch: {path}')
        for key, expected in (('q', agent.q), ('counts', agent.counts)):
            array = package[key]
            if (array.shape != expected.shape or array.dtype != expected.dtype
                    or not np.isfinite(array).all() or array_hash(array) != m[f'{key}_sha256']):
                raise ValueError(f'Invalid {key}: {path}')
            expected[:] = array
    if int(agent.counts.sum(dtype=np.uint64)) != step:
        raise ValueError(f'Wrong model update count: {path}')
    return agent
