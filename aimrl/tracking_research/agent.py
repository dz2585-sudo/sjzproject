"""Matched Q updates with explicit observation provenance and genuine update counts."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from ..tracking25.agent import TrackingAgent

INDEXING_VERSION = 'tracking_relative_offsets_v1'
ABSOLUTE_VERSION = 'tracking_absolute_v2'


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


class ResearchAgent(TrackingAgent):
    def __init__(self, config, representation, rng=None, count_updates=True):
        if representation not in ('absolute', 'relative'):
            raise ValueError('Unknown observation representation')
        self.representation = representation
        self.indexing_version = INDEXING_VERSION if representation == 'relative' else ABSOLUTE_VERSION
        self.shape = (49, 49, 4, 2, 10) if representation == 'relative' else (25, 25, 25, 25, 4, 2, 10)
        self.q = np.zeros(self.shape, dtype=np.float32)
        self.counts = np.zeros(self.shape, dtype=np.uint32) if count_updates else None
        self.alpha, self.gamma = config['learning']['alpha'], config['learning']['gamma']
        self.rng = rng if rng is not None else np.random.default_rng()

    def index(self, state):
        if self.representation == 'absolute':
            return TrackingAgent.index(self, state)
        if (len(state) != 6 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in state)
                or any(not 0 <= x < 25 for x in state[:4]) or state[4] not in range(4) or state[5] not in (1, 2)):
            raise ValueError('Expected a valid six-integer physical observation')
        return state[2] - state[0] + 24, state[3] - state[1] + 24, state[4], state[5] - 1

    def update(self, state, action, reward, next_state):
        index = self.index(state) + (action,)
        target = reward + self.gamma * float(np.max(self.q[self.index(next_state)]))
        self.q[index] += self.alpha * (target - float(self.q[index]))
        if self.counts is not None:
            self.counts[index] += 1

    def save_package(self, path):
        np.savez_compressed(path, q_table=self.q, representation=np.array(self.representation),
                            indexing_version=np.array(self.indexing_version))
        if self.counts is not None:
            np.savez_compressed(Path(path).with_name('visitation.npz'), counts=self.counts)

    def load_package(self, path, with_counts=False):
        with np.load(path, allow_pickle=False) as package:
            if (str(package['representation']) != self.representation or str(package['indexing_version']) != self.indexing_version):
                raise ValueError('Incompatible model representation or indexing version')
            q = package['q_table']
            if q.shape != self.shape or q.dtype != np.float32 or not np.isfinite(q).all():
                raise ValueError('Incompatible or non-finite model table')
            self.q[:] = q
        if with_counts:
            with np.load(Path(path).with_name('visitation.npz'), allow_pickle=False) as package:
                counts = package['counts']
                if counts.shape != self.shape or counts.dtype != np.uint32:
                    raise ValueError('Invalid visitation counter package')
                self.counts = counts.copy()

    def snapshot(self):
        return array_hash(self.q), array_hash(self.counts) if self.counts is not None else None, copy.deepcopy(self.rng.bit_generator.state)


def coverage(agent):
    counts = agent.counts
    rows = counts.sum(axis=-1, dtype=np.uint64)
    pairs = counts[counts > 0]
    visited = rows[rows > 0]
    return {'state_rows': int(rows.size), 'state_action_pairs': int(counts.size),
            'visited_rows': int(visited.size), 'visited_pairs': int(pairs.size),
            'visited_row_fraction': float(visited.size / rows.size),
            'visited_pair_fraction': float(pairs.size / counts.size),
            'zero_pair_fraction': float(1 - pairs.size / counts.size),
            'update_sum': int(counts.sum(dtype=np.uint64)),
            'visited_pair_quantiles': np.quantile(pairs, [0, .25, .5, .75, .95, 1]).tolist() if pairs.size else [],
            'visited_row_quantiles': np.quantile(visited, [0, .25, .5, .75, .95, 1]).tolist() if visited.size else [],
            'quantile_probabilities': [0, .25, .5, .75, .95, 1]}
