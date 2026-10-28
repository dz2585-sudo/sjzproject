"""Fully observed continuous tracking with policy-independent target randomness."""
import numbers

import numpy as np

HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0), (-3, 0), (3, 0), (0, -3), (0, 3))
ACTION_NAMES = ('left_slow', 'right_slow', 'up_slow', 'down_slow', 'stay', 'fire', 'left_fast', 'right_fast', 'up_fast', 'down_fast')


def advance_target(x, y, heading, speed, size):
    """Reflect one unit substep at a time; return position, heading and bounce count."""
    reflections = 0
    for _ in range(speed):
        dx, dy = HEADINGS[heading]
        if not (0 <= x + dx < size and 0 <= y + dy < size):
            heading = (heading + 2) % 4
            reflections += 1
            dx, dy = HEADINGS[heading]
        x += dx; y += dy
    return x, y, heading, reflections


def clipped_position(x, y, action, size):
    dx, dy = MOVES[action]
    return max(0, min(size - 1, x + dx)), max(0, min(size - 1, y + dy))


class TrackingEnv:
    def __init__(self, config):
        e = config['environment']
        self.size = e['grid_size']
        self.minimum_distance = e['minimum_reset_distance']
        self.turn_probability = e['turn_probability']
        self.toggle_probability = e['speed_toggle_probability']
        self.cap = e['episode_ticks']
        self.rewards = dict(config['rewards'])
        if type(self.size) is not int or self.size < 2 or not 0 <= self.minimum_distance <= 2 * (self.size - 1):
            raise ValueError('Grid size or minimum reset separation is infeasible')
        if tuple(config['actions']) != ACTION_NAMES:
            raise ValueError('Incompatible tracking action schema')
        if not 0 <= self.turn_probability <= 1 or not 0 <= self.toggle_probability <= 1 or self.cap <= 0:
            raise ValueError('Invalid target probabilities or episode length')
        self.rng = np.random.default_rng()
        self.state = None
        self.elapsed = 0
        self.finished = False

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        while True:
            xc, yc, xt, yt = map(int, self.rng.integers(self.size, size=4))
            if abs(xc - xt) + abs(yc - yt) >= self.minimum_distance:
                break
        heading = int(self.rng.integers(4))
        speed = int(self.rng.integers(1, 3))
        self.state = (xc, yc, xt, yt, heading, speed)
        self.elapsed, self.finished = 0, False
        return self.state, {'elapsed_ticks': 0}

    def step(self, action):
        if self.state is None or self.finished:
            raise RuntimeError('Call reset() before stepping a new or truncated tracking window')
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, numbers.Integral) or not 0 <= action < 10:
            raise ValueError('Action must be an integer ID 0–9, excluding bool')
        action = int(action)
        xc, yc, xt, yt, heading, speed = self.state
        before_x, before_y = xc, yc
        pre_distance = abs(xc - xt) + abs(yc - yt)
        xc, yc = clipped_position(xc, yc, action, self.size)
        hit = action == 5 and (xc, yc) == (xt, yt)
        miss = action == 5 and not hit
        shot = (xc, yc) if action == 5 else None
        u_turn, u_sign, u_speed = self.rng.random(3)
        turn = u_turn < self.turn_probability
        toggle = u_speed < self.toggle_probability
        if turn:
            heading = (heading + (1 if u_sign < .5 else -1)) % 4
        if toggle:
            speed = 3 - speed
        old_xt, old_yt = xt, yt
        xt, yt, heading, reflections = advance_target(xt, yt, heading, speed, self.size)
        self.state = (xc, yc, xt, yt, heading, speed)
        self.elapsed += 1
        self.finished = self.elapsed >= self.cap
        reward = self.rewards['step'] + self.rewards['hit'] * hit + self.rewards['miss'] * miss
        info = {'elapsed_ticks': self.elapsed, 'hit': hit, 'miss': miss, 'shot': shot, 'base_reward': reward,
                'pre_distance': pre_distance, 'distance': abs(xc - xt) + abs(yc - yt),
                'movement_mode': 'fast' if action >= 6 else 'slow' if action <= 3 else 'fire' if action == 5 else 'stay',
                'actual_displacement': abs(xc - before_x) + abs(yc - before_y),
                'random_turn': bool(turn), 'speed_toggle': bool(toggle), 'reflections': reflections,
                'target_nominal_speed': speed, 'target_path_distance': speed,
                'target_net_displacement': abs(xt - old_xt) + abs(yt - old_yt)}
        return self.state, float(reward), False, self.finished, info
