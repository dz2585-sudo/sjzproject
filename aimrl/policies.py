"""Untrained diagnostic policies."""


def rule_policy(observation):
    xc, yc, xt, yt, _ = observation
    if (xc, yc) == (xt, yt):
        return 5
    if xc != xt:
        return 1 if xc < xt else 0
    return 3 if yc < yt else 2


def random_policy(rng):
    return int(rng.integers(6))
