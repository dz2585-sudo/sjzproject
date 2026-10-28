"""Run a bounded greedy demonstration from a saved continuing-tracking model."""
import argparse
from pathlib import Path

from ..model_io import artifact_digest, resolve_artifact
from .agent import TrackingAgent
from .env import TrackingEnv, ACTION_NAMES
from .workflow import read_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3601000)
    parser.add_argument('--output', type=Path, default=Path('.tmp/saved_tracking_demo.json'))
    args = parser.parse_args()
    if not 1 <= args.ticks <= 300: parser.error('--ticks must be in 1–300')
    path = resolve_artifact(args.model)
    config = read_json(path.parent / 'run_config.json')
    metadata = read_json(path.parent / 'metadata.json')
    if artifact_digest(path) != metadata['model_sha256']: raise ValueError('Saved model hash mismatch')
    agent = TrackingAgent(config); agent.load(path)
    env = TrackingEnv(config); state, _ = env.reset(seed=args.seed)
    rows = []
    for tick in range(1, args.ticks + 1):
        action = agent.greedy(state)
        state, reward, terminated, truncated, info = env.step(action)
        rows.append({'tick': tick, 'state': state, 'action': ACTION_NAMES[action], 'reward': reward,
                     'hit': info['hit'], 'distance': info['distance'], 'terminated': terminated, 'truncated': truncated})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {'model': str(path), 'seed': args.seed, 'ticks': len(rows), 'trajectory': rows, 'learning_updates': 0})
    print(f'Saved-model demonstration: {len(rows)} ticks, {sum(r["hit"] for r in rows)} hits; no learning updates.')


if __name__ == '__main__': main()
