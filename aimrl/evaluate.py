"""Evaluate frozen policies on independent, seeded environments."""
from .env import AimEnv


def evaluate(config, policy, seed_start, episodes):
    env = AimEnv(config)
    rows = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed_start + i)
        misses, total = 0, 0.0
        while True:
            state, reward, terminated, truncated, info = env.step(policy(state))
            misses += int(info["miss"])
            total += reward
            if terminated or truncated:
                rows.append({"scenario_seed": seed_start + i, "steps": info["step_count"], "hit": info["hit"],
                             "misses": misses, "terminated": terminated, "truncated": truncated, "base_return": total})
                break
    summary = {"episodes": len(rows), "success_rate": sum(r["hit"] for r in rows) / len(rows),
               "mean_steps": sum(r["steps"] for r in rows) / len(rows),
               "mean_misses": sum(r["misses"] for r in rows) / len(rows),
               "mean_base_return": sum(r["base_return"] for r in rows) / len(rows)}
    return summary, rows
