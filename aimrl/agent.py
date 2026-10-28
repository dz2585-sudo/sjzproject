"""Tabular Q-learning agent."""
import numpy as np

from .model_io import load_model_array


class QLearningAgent:
    def __init__(self, config, rng=None):
        if config["actions"] != ["left", "right", "up", "down", "stay", "fire"]:
            raise ValueError("Incompatible action mapping")
        settings = config["learning"]
        if settings["q_dtype"] != "float32":
            raise ValueError("Expected float32 Q table")
        size = config["environment"]["grid_size"]
        self.shape = (size, size, size, size, 2, len(config["actions"]))
        self.q = np.full(self.shape, settings["initial_q"], dtype=np.float32)
        self.alpha, self.gamma = settings["alpha"], settings["gamma"]
        self.rng = rng if rng is not None else np.random.default_rng()

    def state_index(self, state):
        xc, yc, xt, yt, direction = state
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) or not 0 <= x < self.shape[0] for x in (xc, yc, xt, yt)):
            raise ValueError("state coordinates outside grid")
        return (xc, yc, xt, yt, 0 if direction == -1 else 1)

    def greedy_action(self, state):
        return int(np.argmax(self.q[self.state_index(state)]))

    def select_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.shape[-1]))
        values = self.q[self.state_index(state)]
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, state, action, reward, next_state, terminated, truncated=False):
        index = self.state_index(state) + (action,)
        target = reward if terminated else reward + self.gamma * float(np.max(self.q[self.state_index(next_state)]))
        self.q[index] += self.alpha * (target - self.q[index])

    def save(self, path):
        if not np.isfinite(self.q).all():
            raise ValueError("Cannot save non-finite Q values")
        np.save(path, self.q, allow_pickle=False)

    def load(self, path):
        data = load_model_array(path)
        if data.shape != self.shape or data.dtype != np.float32:
            raise ValueError(f"Incompatible Q table: expected shape {self.shape} and float32, got {data.shape} and {data.dtype}")
        if not np.isfinite(data).all():
            raise ValueError("Q table contains non-finite values")
        self.q = data.copy()
