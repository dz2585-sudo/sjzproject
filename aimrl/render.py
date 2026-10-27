"""Draw an observed state without changing the environment."""
import matplotlib.pyplot as plt


def draw_state(ax, observation, size, policy, episode, step, action, reward, status):
    xc, yc, xt, yt, direction = observation
    ax.clear()
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(size)], labels=range(size))
    ax.set_yticks([i + 0.5 for i in range(size)], labels=range(size))
    ax.set_xticks(range(size + 1), minor=True)
    ax.set_yticks(range(size + 1), minor=True)
    ax.grid(which="minor", color="#b6bec9", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.scatter(xt + 0.5, yt + 0.5, s=400, facecolors="none", edgecolors="#d94f39", linewidths=3, label="Target", zorder=3)
    ax.scatter(xc + 0.5, yc + 0.5, s=190, marker="x", color="#1766b1", linewidths=3, label="Crosshair", zorder=4)
    ax.set_xlabel("x (right increases)")
    ax.set_ylabel("y (down increases)")
    ax.set_title(f"{policy} policy | Episode {episode} | Step {step}\n"
                 f"Action: {action} | Reward: {reward:+.2f} | {status}\n"
                 f"Target direction: {'right' if direction == 1 else 'left'}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.figure.tight_layout()
