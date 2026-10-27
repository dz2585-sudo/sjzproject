"""Automatic rule or random policy demonstration and diagnostic records."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .env import AimEnv
from .policies import random_policy, rule_policy


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def nonnegative_int(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return n


def positive_float(value):
    n = float(value)
    if not np.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("rule", "random", "qlearning"), default="rule")
    parser.add_argument("--model", type=Path, help="final q_table.npy with run_config.json and metadata.json alongside")
    parser.add_argument("--episodes", type=positive_int, default=3)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--interval-ms", type=positive_float, default=250)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--save-frame", type=Path)
    args = parser.parse_args()

    config = load_config()
    agent = None
    policy_label = args.policy
    if args.policy == "qlearning":
        if args.model is None:
            parser.error("--policy qlearning requires --model")
        from .agent import QLearningAgent
        with (args.model.parent / "run_config.json").open(encoding="utf-8") as file:
            model_config = json.load(file)
        if model_config["actions"] != config["actions"] or model_config["environment"]["grid_size"] != config["environment"]["grid_size"]:
            raise ValueError("Model config is incompatible with the current grid size or action mapping")
        with (args.model.parent / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)
        config = model_config
        agent = QLearningAgent(config)
        agent.load(args.model)
        mode = metadata.get("reward_mode", "base")
        if mode not in ("base", "potential"):
            raise ValueError(f"Unknown model reward mode: {mode}")
        policy_label = f"{mode.title()} Q-learning ({metadata['global_steps']:,} training steps)"

    plt = fig = ax = None
    if not args.headless or args.save_frame:
        from .mpl_setup import configure_mpl
        print(f"MPLCONFIGDIR={configure_mpl()}")
        import matplotlib
        if args.headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .render import draw_state
        fig, ax = plt.subplots(figsize=(6, 6))
        if not args.headless:
            plt.show(block=False)

    env = AimEnv(config)
    policy_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    records = []
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode - 1)
            misses = 0
            total_reward = 0.0
            while True:
                if fig is not None and not args.headless and not plt.fignum_exists(fig.number):
                    print("Window closed; ending demo.")
                    return
                if args.policy == "rule":
                    action = rule_policy(observation)
                elif args.policy == "random":
                    action = random_policy(policy_rng)
                else:
                    action = agent.greedy_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                misses += int(info["miss"])
                total_reward += reward
                status = "Hit" if terminated else "Timeout" if truncated else "In progress"
                if fig is not None and (not args.headless or (episode == args.episodes and (terminated or truncated))):
                    draw_state(ax, observation, env.size, policy_label, episode, info["step_count"], config["actions"][action], reward, status)
                    if not args.headless:
                        fig.canvas.draw_idle()
                        plt.pause(args.interval_ms / 1000)
                if terminated or truncated:
                    records.append({"policy": args.policy, "episode": episode, "scenario_seed": args.seed + episode - 1,
                                    "steps": info["step_count"], "hit": info["hit"], "misses": misses,
                                    "terminated": terminated, "truncated": truncated, "base_return": total_reward})
                    break
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_frame, dpi=150, bbox_inches="tight")
        n = len(records)
        print(f"Demo diagnostic ({args.policy}, {n} episodes): success rate {sum(r['hit'] for r in records) / n:.1%}; "
              f"mean steps {sum(r['steps'] for r in records) / n:.2f}; mean misses {sum(r['misses'] for r in records) / n:.2f}.")
        if fig is not None and not args.headless:
            print("Demo complete; close the window to exit.")
            plt.show(block=True)
    finally:
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
