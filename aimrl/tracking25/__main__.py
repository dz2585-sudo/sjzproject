"""Run independent Tracking25 V2 stages without changing historical experiments."""
import argparse
from pathlib import Path

from .workflow import ROOT, check, evaluate_stage, read_json, train_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/tracking25_v2.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/continuous_tracking')
    parser.add_argument('--stage', choices=('check', 'train', 'evaluate', 'render'), required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.stage == 'check': check(config, args.output)
    elif args.stage == 'train': train_stage(config, args.output)
    elif args.stage == 'evaluate': evaluate_stage(config, args.output)
    else:
        from .render import render_stage
        render_stage(config, args.output)


if __name__ == '__main__': main()
