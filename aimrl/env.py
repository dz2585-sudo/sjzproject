"""Grid environment with a moving target and base rewards."""
import numbers

import numpy as np


class AimEnv:
    def __init__(self, config):
        settings = config["environment"]
        self.size = settings["grid_size"]
        self.turn_probability = settings["target_turn_probability"]
        self.move_probability = settings["target_move_probability"]
        self.max_steps = settings["max_episode_steps"]
        self.rewards = dict(config["rewards"])
        if config["actions"] != ["left", "right", "up", "down", "stay", "fire"]:
            raise ValueError("Unexpected action ordering in config")
        if self.size < 2:
            raise ValueError("grid_size must be at least 2 for a moving target")
        self.rng = np.random.default_rng()
        self.state = None
        self.step_count = 0
        self.finished = False

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        xc, yc = (int(x) for x in self.rng.integers(self.size, size=2))
        while True:
            xt, yt = (int(x) for x in self.rng.integers(self.size, size=2))
            if (xc, yc) != (xt, yt):
                break
        direction = int(self.rng.choice([-1, 1]))
        self.state = (xc, yc, xt, yt, direction)
        self.step_count = 0
        self.finished = False
        return self.state, {"step_count": 0, "hit": False, "miss": False, "base_reward": 0.0}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.finished:
            raise RuntimeError("Episode has ended; call reset() before step().")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, numbers.Integral) or not 0 <= action < 6:
            raise ValueError("action must be an integer from 0 through 5 (not bool).")
        action = int(action)
        xc, yc, xt, yt, direction = self.state
        if action == 0:
            xc = max(0, xc - 1)
        elif action == 1:
            xc = min(self.size - 1, xc + 1)
        elif action == 2:
            yc = max(0, yc - 1)
        elif action == 3:
            yc = min(self.size - 1, yc + 1)
        self.step_count += 1
        hit = action == 5 and (xc, yc) == (xt, yt)
        miss = action == 5 and not hit
        if not hit:
            if self.rng.random() < self.turn_probability:
                direction = -direction
            if self.rng.random() < self.move_probability:
                if not 0 <= xt + direction < self.size:
                    direction = -direction
                xt += direction
        reward = self.rewards["step"]
        if hit:
            reward += self.rewards["hit_bonus"]
        elif miss:
            reward += self.rewards["miss_penalty"]
        terminated = hit
        truncated = not hit and self.step_count >= self.max_steps
        self.finished = terminated or truncated
        self.state = (xc, yc, xt, yt, direction)
        info = {"step_count": self.step_count, "hit": hit, "miss": miss, "base_reward": reward}
        return self.state, reward, terminated, truncated, info
