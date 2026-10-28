"""Transparent tracking baselines with no access to future target randomness."""
from .env import advance_target, clipped_position


def rule_policy(state, size):
    xc, yc, xt, yt, heading, speed = state
    if (xc, yc) == (xt, yt):
        return 5
    px, py, _, _ = advance_target(xt, yt, heading, speed, size)
    candidates = []
    for action in (0, 1, 2, 3, 4, 6, 7, 8, 9):
        x, y = clipped_position(xc, yc, action, size)
        candidates.append((abs(x - px) + abs(y - py), abs(x - xc) + abs(y - yc), action))
    return min(candidates)[2]
