"""Run an explicitly selected stage of the bounded tracking study."""
import argparse
import time
import traceback

from .io import ROOT, peak_memory_bytes, read_json, save_json
from . import workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/tracking_research.json')
    parser.add_argument('--stage', required=True, choices=('check','diagnose','pilot','train','evaluate','render','audit'))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    output = ROOT / config['study']['output']
    started = time.perf_counter()
    code = 0
    try:
        if args.stage == 'check': workflow.check(config, output)
        elif args.stage == 'diagnose': workflow.diagnose(config, output)
        elif args.stage == 'pilot': workflow.pilot(config, output)
        elif args.stage == 'train': workflow.train(config, output)
        elif args.stage == 'evaluate': workflow.final_evaluation(config, output)
        elif args.stage == 'render':
            from .render import render
            render(config, output)
        elif args.stage == 'audit':
            from .audit import audit
            audit(config, output)
    except Exception:
        code = 1; traceback.print_exc()
    if output.exists():
        records = output / 'stage_execution'
        save_json(records / f'{args.stage}_{time.time_ns()}.json', {'command': f'python -m aimrl.tracking_research --config {args.config} --stage {args.stage}',
            'exit_code': code, 'elapsed_seconds': time.perf_counter()-started, 'peak_working_set_bytes': peak_memory_bytes()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
