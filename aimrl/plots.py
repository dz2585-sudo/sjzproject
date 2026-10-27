"""Plot the observed validation results of a single run."""
import argparse
import csv
import json
from pathlib import Path

from .mpl_setup import configure_mpl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    cache = configure_mpl()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (args.run_dir / "validation.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with (args.run_dir / "metadata.json").open(encoding="utf-8") as file:
        meta = json.load(file)
    steps = [int(r["global_step"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(steps, [100 * float(r["success_rate"]) for r in rows], marker="o")
    axes[0].set_ylabel("Success rate (%)")
    axes[1].plot(steps, [float(r["mean_steps"]) for r in rows], marker="o")
    axes[1].set_ylabel("Mean steps")
    axes[1].set_xlabel("Training interaction steps")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{meta.get('reward_mode', 'base').title()} Q-learning | seed {meta['root_seed']} | Validation")
    fig.tight_layout()
    fig.savefig(args.run_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved validation curve; MPLCONFIGDIR={cache}")


if __name__ == "__main__":
    main()
