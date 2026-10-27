"""Train Q-learning with base or potential-shaped rewards."""
import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .agent import QLearningAgent
from .env import AimEnv
from .evaluate import evaluate
from .policies import random_policy, rule_policy
from .rewards import learning_reward


def epsilon_at(step, learning):
    fraction = min(step / learning["epsilon_decay_steps"], 1.0)
    return learning["epsilon_start"] + (learning["epsilon_end"] - learning["epsilon_start"]) * fraction


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train(config, seed, output, validate_mid=True, reward_mode="base"):
    if reward_mode not in ("base", "potential"):
        raise ValueError(f"Unknown reward mode: {reward_mode}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    start_time = time.perf_counter()
    with (output / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    env_seq, action_seq = np.random.SeedSequence(seed).spawn(2)
    env_seed = int(env_seq.generate_state(1, dtype=np.uint64)[0])
    env = AimEnv(config)
    agent = QLearningAgent(config, np.random.default_rng(action_seq))
    state, _ = env.reset(seed=env_seed)
    learning = config["learning"]
    validation = config["experiments"]["validation"]
    episode_rows, validation_rows, validation_episodes = [], [], []
    current_steps, current_misses, current_return = 0, 0, 0.0
    current_learning_return, current_shaping_return = 0.0, 0.0

    def checkpoint(step):
        agent.save(output / "checkpoints" / f"q_{step:06d}.npy")
        summary, rows = evaluate(config, agent.greedy_action, validation["seed_start"], validation["episodes"])
        validation_rows.append({"global_step": step, **summary})
        validation_episodes.extend({"global_step": step, **row} for row in rows)
        print(f"step {step}: success={summary['success_rate']:.1%}, mean steps={summary['mean_steps']:.2f}, misses={summary['mean_misses']:.2f}", flush=True)

    checkpoint(0)
    for t in range(learning["training_steps"]):
        epsilon = epsilon_at(t, learning)
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        learned, shaping = learning_reward(reward, state, next_state, terminated, reward_mode,
                                            learning["gamma"], learning["shaping_scale"])
        agent.update(state, action, learned, next_state, terminated, truncated)
        current_steps += 1
        current_misses += int(info["miss"])
        current_return += reward
        current_learning_return += learned
        current_shaping_return += shaping
        state = next_state
        if terminated or truncated:
            episode_rows.append({"end_step": t + 1, "length": current_steps, "hit": terminated,
                                 "terminated": terminated, "truncated": truncated, "misses": current_misses,
                                 "base_return": current_return, "learning_return": current_learning_return,
                                 "shaping_return": current_shaping_return, "epsilon": epsilon})
            state, _ = env.reset()
            current_steps, current_misses, current_return = 0, 0, 0.0
            current_learning_return, current_shaping_return = 0.0, 0.0
        if validate_mid and (t + 1) % validation["interval_steps"] == 0:
            checkpoint(t + 1)
    agent.save(output / "q_table.npy")
    reloaded = QLearningAgent(config)
    reloaded.load(output / "q_table.npy")
    if not np.array_equal(agent.q, reloaded.q) or not np.array_equal(np.argmax(agent.q, axis=-1), np.argmax(reloaded.q, axis=-1)):
        raise RuntimeError("Saved Q table failed reload verification")

    baseline_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    baselines = []
    for name, policy in (("rule", rule_policy), ("random", lambda state: random_policy(baseline_rng))):
        summary, _ = evaluate(config, policy, validation["seed_start"], validation["episodes"])
        baselines.append({"policy": name, "scenario_seed_start": validation["seed_start"],
                          "scenario_seed_end": validation["seed_start"] + validation["episodes"] - 1,
                          "policy_rng": "none" if name == "rule" else f"SeedSequence([{seed}, 2])", **summary})
    if episode_rows:
        write_csv(output / "episodes.csv", episode_rows)
    else:
        with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(["end_step", "length", "hit", "terminated", "truncated", "misses", "base_return", "learning_return", "shaping_return", "epsilon"])
    write_csv(output / "validation.csv", validation_rows)
    write_csv(output / "validation_episodes.csv", validation_episodes)
    write_csv(output / "baselines.csv", baselines)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = {"reward_mode": reward_mode, "gamma": learning["gamma"], "shaping_scale": learning["shaping_scale"],
                "return_definition": "Undiscounted sums of per-step base, learning and shaping rewards.",
                "root_seed": seed, "random_streams": "SeedSequence(root).spawn(2): environment, action exploration",
                "environment_initialization_seed": env_seed, "global_steps": learning["training_steps"],
                "unfinished_episode": {"steps": current_steps, "misses": current_misses, "base_return": current_return,
                                       "learning_return": current_learning_return, "shaping_return": current_shaping_return},
                "complete_episode_steps": sum(row["length"] for row in episode_rows),
                "elapsed_seconds": time.perf_counter() - start_time, "python": sys.version,
                "numpy": np.__version__, "matplotlib": importlib.metadata.version("matplotlib"),
                "platform": platform.platform(), "git_revision": revision,
                "git_note": "Working tree may contain uncommitted changes; revision alone does not identify them." if revision else "No Git commit available; source is unversioned.",
                "reload_verified": True}
    with (output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Completed {metadata['global_steps']} steps in {metadata['elapsed_seconds']:.2f}s; reload verified.")
    return agent, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--reward-mode", choices=["base", "potential"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)
    train(config, args.seed, args.output, reward_mode=args.reward_mode)


if __name__ == "__main__":
    main()
