"""Check the project setup configuration and dependencies without running training."""

import importlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "base.json"
errors = []


def fail(message):
    errors.append(message)
    print(f"FAIL: {message}")


def field(data, path, expected):
    value = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            fail(f"Missing {path}; add it to {CONFIG}.")
            return None
        value = value[key]
    if expected == "number":
        valid = type(value) in (int, float) and math.isfinite(value)
    else:
        valid = type(value) is expected
    if not valid:
        fail(f"Invalid {path}: expected {expected}; correct the configuration.")
        return None
    return value


def positive(value, name):
    if value is not None and value <= 0:
        fail(f"{name} must be a positive integer; correct the configuration.")


def probability(value, name):
    if value is not None and not 0 <= value <= 1:
        fail(f"{name} must be in [0, 1]; correct the configuration.")


def main():
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    sys.path.insert(0, str(ROOT))
    from aimrl.mpl_setup import configure_mpl
    print(f"MPLCONFIGDIR: {configure_mpl()}")
    modules = {}
    for name in ("numpy", "matplotlib"):
        try:
            modules[name] = importlib.import_module(name)
            print(f"{name}: {modules[name].__version__}")
        except Exception as exc:
            fail(f"Cannot import {name}: {exc}. Install dependencies with python -m pip install -r requirements.txt in the project virtual environment.")

    try:
        with CONFIG.open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, ValueError) as exc:
        fail(f"Cannot read {CONFIG}: {exc}. Restore valid JSON configuration.")
        return 1

    fields = {
        "project.name": str, "project.stage": str,
        "environment.grid_size": int, "environment.target_move_probability": "number",
        "environment.target_turn_probability": "number", "environment.max_episode_steps": int,
        "actions": list, "rewards.step": "number", "rewards.hit_bonus": "number",
        "rewards.miss_penalty": "number", "learning.alpha": "number", "learning.gamma": "number",
        "learning.initial_q": "number", "learning.q_dtype": str, "learning.training_steps": int,
        "learning.epsilon_start": "number", "learning.epsilon_end": "number",
        "learning.epsilon_decay_steps": int, "learning.shaping_scale": "number",
        "experiments.training_seeds": list, "experiments.reward_modes": list,
        "experiments.validation.interval_steps": int, "experiments.validation.episodes": int,
        "experiments.validation.seed_start": int, "experiments.test.move_probabilities": list,
        "experiments.test.episodes_per_condition": int, "experiments.test.seed_start": int,
    }
    values = {name: field(config, name, kind) for name, kind in fields.items()}
    for name in ("environment.grid_size", "environment.max_episode_steps", "learning.training_steps",
                 "learning.epsilon_decay_steps", "experiments.validation.interval_steps",
                 "experiments.validation.episodes", "experiments.test.episodes_per_condition"):
        positive(values[name], name)
    for name in ("environment.target_move_probability", "environment.target_turn_probability",
                 "learning.epsilon_start", "learning.epsilon_end"):
        probability(values[name], name)
    for name in ("learning.alpha", "learning.gamma"):
        value = values[name]
        if value is not None and not 0 < value <= 1:
            fail(f"{name} must be in (0, 1]; correct the configuration.")
    start, end = values["learning.epsilon_start"], values["learning.epsilon_end"]
    if start is not None and end is not None and start < end:
        fail("epsilon_start must be at least epsilon_end; correct the exploration schedule.")
    decay, steps = values["learning.epsilon_decay_steps"], values["learning.training_steps"]
    if decay is not None and steps is not None and decay > steps:
        fail("epsilon_decay_steps must not exceed training_steps; correct the schedule.")
    if values["actions"] is not None and values["actions"] != ["left", "right", "up", "down", "stay", "fire"]:
        fail("actions must be [left, right, up, down, stay, fire] in that order.")
    if values["experiments.reward_modes"] is not None and values["experiments.reward_modes"] != ["base", "potential"]:
        fail("reward_modes must be [base, potential]; correct the configuration.")
    if values["learning.q_dtype"] is not None and values["learning.q_dtype"] != "float32":
        fail("q_dtype must be float32; correct the configuration.")
    seeds = values["experiments.training_seeds"]
    if seeds is not None and (not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds)):
        fail("training_seeds must be a nonempty list of distinct nonnegative integers.")
    probabilities = values["experiments.test.move_probabilities"]
    if probabilities is not None:
        if not probabilities:
            fail("test.move_probabilities must be nonempty.")
        for p in probabilities:
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                fail(f"Invalid test move probability {p!r}; use numbers in [0, 1].")
    v_start, t_start = values["experiments.validation.seed_start"], values["experiments.test.seed_start"]
    v_count, t_count = values["experiments.validation.episodes"], values["experiments.test.episodes_per_condition"]
    for name, value in (("validation.seed_start", v_start), ("test.seed_start", t_start)):
        if value is not None and value < 0:
            fail(f"{name} must be nonnegative; correct the configuration.")
    if all(type(x) is int and x >= 0 for x in (v_start, t_start)) and all(type(x) is int and x > 0 for x in (v_count, t_count)):
        if max(v_start, t_start) < min(v_start + v_count, t_start + t_count):
            fail("Validation and test seed ranges overlap; choose separate ranges.")

    grid, actions, modes = values["environment.grid_size"], values["actions"], values["experiments.reward_modes"]
    if type(grid) is int and grid > 0 and actions is not None:
        states = grid ** 4 * 2
        q_values = states * len(actions)
        print(f"State combinations: {states:,}; Q values: {q_values:,}; float32 element bytes: {q_values * 4:,}")
        if "numpy" in modules:
            try:
                table = modules["numpy"].zeros((grid, grid, grid, grid, 2, len(actions)), dtype="float32")
                if table.nbytes != q_values * 4:
                    fail("NumPy Q-table byte count differs from the expected float32 size.")
                else:
                    print(f"NumPy zero Q-table nbytes: {table.nbytes:,}")
            except Exception as exc:
                fail(f"Cannot create NumPy Q-table: {exc}; check NumPy and the grid configuration.")
    if modes is not None and seeds is not None and type(steps) is int and steps > 0:
        runs = len(modes) * len(seeds)
        print(f"Training runs: {len(modes)} reward modes x {len(seeds)} seeds = {runs}; total interaction steps: {runs * steps:,}")
    if errors:
        print(f"Check failed: {len(errors)} issue(s).")
        return 1
    print("All configuration and dependency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
