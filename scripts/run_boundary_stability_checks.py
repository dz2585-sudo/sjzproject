"""Record actual commands, exit codes and wall time without regenerating training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aimrl.tracking_boundary_stability.core import OUT
from aimrl.tracking_research.io import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','postflight'))
    p.add_argument('--legacy-kernel',type=Path,required=True)
    args=p.parse_args()
    prefix=args.stage
    if (OUT/f'{args.stage}.json').exists():
        number=len(list((OUT/'execution').glob(f'{args.stage}_attempt_*.json')))+1
        prior=json.loads((OUT/f'{args.stage}.json').read_text(encoding='utf-8'))
        save_json(OUT/'execution'/f'{args.stage}_attempt_{number}.json',prior)
        prefix=f'{args.stage}_retry{number}'
    env=dict(os.environ,AIMRL_BOUNDARY_KERNEL=str(OUT/'native/boundary.dll'),
             AIMRL_TRADEOFF_KERNEL=str(args.legacy_kernel.resolve()),MPLCONFIGDIR=str(ROOT/'.mplconfig'))
    commands={
        'full_tests':[sys.executable,'-m','unittest','discover','-s','tests','-v'],
        'optimized_new_tests':[sys.executable,'-O','-m','unittest','discover','-s','tests','-p','test_tracking_boundary_stability.py','-v'],
        'publication':[sys.executable,'scripts/verify_tracking_publication.py','--output',str(OUT/f'checks/{prefix}_publication.json')],
        'publication_optimized':[sys.executable,'-O','scripts/verify_tracking_publication.py','--output',str(OUT/f'checks/{prefix}_publication_optimized.json')],
        'studies':[sys.executable,'scripts/verify_tracking_studies.py','--output',str(OUT/f'checks/{prefix}_studies')],
        'studies_optimized':[sys.executable,'-O','scripts/verify_tracking_studies.py','--output',str(OUT/f'checks/{prefix}_studies_optimized')],
    }
    if args.stage=='postflight':
        commands.update(new_verify=[sys.executable,'-m','aimrl.tracking_boundary_stability','verify'],
                        new_verify_optimized=[sys.executable,'-O','-m','aimrl.tracking_boundary_stability','verify'])
    results={};folder=OUT/'execution';folder.mkdir(exist_ok=True)
    for label,command in commands.items():
        path=folder/f'{prefix}_{label}.txt'
        started=time.perf_counter()
        with path.open('w',encoding='utf-8') as stream:
            run=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
        results[label]=dict(command=command,returncode=run.returncode,elapsed_seconds=time.perf_counter()-started,
                            log=path.relative_to(OUT).as_posix())
        save_json(OUT/f'{args.stage}.json',results)
        print(f'{label}: exit {run.returncode}, {results[label]["elapsed_seconds"]:.2f}s',flush=True)
    return int(any(r['returncode']!=0 for r in results.values()))


if __name__=='__main__':raise SystemExit(main())
