"""Run one isolated tracking–shooting study phase."""
import argparse
from pathlib import Path
import sys
import time

from ..tracking_research.io import save_json
from .core import Kernel
from .study import config_checked, evaluate_stage, smoke_stage, train_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('smoke', 'train', 'evaluate', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == 'smoke':
        output.mkdir(parents=True, exist_ok=False)
    execution = output / 'execution'
    execution.mkdir(exist_ok=True)
    record = execution / f'{args.phase}.json'
    if record.exists():
        parser.error(f'Phase already attempted: {record}')
    started = time.perf_counter()
    state = {'command': ['python', '-m', 'aimrl.tracking_tradeoff', *sys.argv[1:]], 'complete': False}
    try:
        config = config_checked()
        kernel = Kernel(args.kernel)
        if args.phase == 'smoke':
            smoke_stage(config, kernel, output)
        elif args.phase == 'train':
            train_stage(config, kernel, args.kernel, output)
        elif args.phase == 'evaluate':
            evaluate_stage(config, kernel, args.kernel, output)
        else:
            from .report import report
            report(config, kernel, args.kernel, output)
        state.update(complete=True, exit_code=0)
    except Exception as exc:
        state.update(exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['elapsed_seconds'] = time.perf_counter() - started
        save_json(record, state)


if __name__ == '__main__':
    main()
