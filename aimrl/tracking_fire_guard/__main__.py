"""Run the inference-only firing-guard study in ordered phases."""
import argparse
from pathlib import Path
import sys
import time

from ..tracking_research.io import save_json
from . import study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('setup','gate','evaluate','report'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prior-kernel',type=Path)
    parser.add_argument('--rebuilt-kernel',action='store_true',help='Verify a rebuilt DLL by full read-only replay parity')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.phase=='gate' and args.prior_kernel is None:
        parser.error('--prior-kernel is required for the native/Python audit')
    if args.phase!='gate' and args.rebuilt_kernel:
        parser.error('--rebuilt-kernel is only supported for the gate')
    if args.workers<1:
        parser.error('--workers must be positive')
    output=args.output.resolve()
    if args.phase=='setup':
        output.mkdir(parents=True,exist_ok=False)
    record=output/'execution'/f'{args.phase}.json'
    if record.exists():
        parser.error(f'Phase already attempted: {record}')
    state={'command':['python','-m','aimrl.tracking_fire_guard',*sys.argv[1:]],'complete':False}
    started=time.perf_counter()
    try:
        if args.phase=='setup':
            study.setup(output)
        elif args.phase=='gate':
            study.gate(output,args.prior_kernel,args.rebuilt_kernel)
        elif args.phase=='evaluate':
            study.evaluate(output,args.workers)
        else:
            from .report import report
            report(output)
        state.update(complete=True,exit_code=0)
    except Exception as exc:
        state.update(exit_code=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['elapsed_seconds']=time.perf_counter()-started
        save_json(record,state)


if __name__=='__main__':
    main()
