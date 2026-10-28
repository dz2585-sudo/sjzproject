"""Continuing tabular Q-learning; fixed lowest-ID tie breaking in all modes."""
from pathlib import Path

import numpy as np

from ..model_io import load_model_array

from .env import ACTION_NAMES


def potential(state):
    return -(abs(state[0] - state[2]) + abs(state[1] - state[3]))


def learning_reward(reward, state, next_state, mode, gamma):
    if mode == 'base':
        return reward, 0.0
    if mode != 'potential':
        raise ValueError('Unknown tracking reward mode')
    shaped = gamma * potential(next_state) - potential(state)
    return reward + shaped, shaped


def epsilon(t, learning):
    return learning['epsilon_start'] + (learning['epsilon_end'] - learning['epsilon_start']) * min(t / learning['epsilon_decay_steps'], 1.0)


class TrackingAgent:
    def __init__(self, config, rng=None):
        if config.get('project', {}).get('schema') != 'tracking25_v2' or tuple(config.get('actions', ())) != ACTION_NAMES:
            raise ValueError('Expected Tracking25 V2 observation and action schema')
        n = config['environment']['grid_size']
        self.shape = (n, n, n, n, 4, 2, 10)
        self.q = np.zeros(self.shape, dtype=np.float32)
        self.alpha = config['learning']['alpha']
        self.gamma = config['learning']['gamma']
        self.rng = rng if rng is not None else np.random.default_rng()

    def index(self, state):
        if len(state) != 6 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in state):
            raise ValueError('Expected six integer tracking state fields')
        if not all(0 <= x < self.shape[0] for x in state[:4]) or state[4] not in range(4) or state[5] not in (1, 2):
            raise ValueError('Invalid tracking coordinate, heading or speed')
        return (*state[:5], state[5] - 1)

    def greedy(self, state):
        return int(np.argmax(self.q[self.index(state)]))

    def select(self, state, exploration):
        if self.rng.random() < exploration:
            return int(self.rng.integers(10))
        return self.greedy(state)

    def update(self, state, action, reward, next_state):
        index = self.index(state) + (action,)
        target = reward + self.gamma * float(np.max(self.q[self.index(next_state)]))
        self.q[index] += self.alpha * (target - float(self.q[index]))

    def save(self, path):
        np.save(path, self.q, allow_pickle=False)

    def load(self, path):
        data = load_model_array(path)
        if data.shape != self.shape or data.dtype != np.float32:
            raise ValueError('Incompatible model: expected Tracking25 V2 six-field, ten-action float32 table')
        for start in range(self.shape[0]):
            if not np.isfinite(data[start]).all():
                raise ValueError('Model contains non-finite values')
            self.q[start] = data[start]


def exact_reload_check(agent, path):
    data = load_model_array(path)
    return (data.shape == agent.q.shape and data.dtype == np.float32
            and all(np.array_equal(data[i], agent.q[i]) for i in range(agent.q.shape[0])))
