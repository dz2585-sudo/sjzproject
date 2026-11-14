"""Recorded read-only acceptance commands; preserves all previous-study files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.tracking_planning.core import OUT,LIBRARY
from aimrl.tracking_research.io import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','postflight'));p.add_argument('--legacy-kernel',type=Path,required=True)
    args=p.parse_args();folder=OUT/'execution';folder.mkdir(exist_ok=True)
    prefix=args.stage
    if (OUT/f'{prefix}.json').exists():
        attempt=len(list(folder.glob(f'{prefix}_attempt_*.json')))+1
        save_json(folder/f'{prefix}_attempt_{attempt}.json',json.loads((OUT/f'{prefix}.json').read_text(encoding='utf-8')))
        prefix+=f'_retry{attempt}'
    env=dict(os.environ,AIMRL_PLANNING_NATIVE='1',AIMRL_BOUNDARY_KERNEL=str(ROOT/'results/tracking_boundary_stability/native/boundary.dll'),
             AIMRL_TRADEOFF_KERNEL=str(args.legacy_kernel.resolve()),MPLCONFIGDIR=str(ROOT/'.mplconfig'))
    env.pop('AIMRL_BOUNDARY_OUTPUT',None)
    commands={
      'full_tests':[sys.executable,'-m','unittest','discover','-s','tests','-v'],
      'new_tests_optimized':[sys.executable,'-O','-m','unittest','discover','-s','tests','-p','test_tracking_planning.py','-v'],
      'publication':[sys.executable,'scripts/verify_tracking_publication.py','--output',str(OUT/f'checks/{prefix}_publication.json')],
      'publication_optimized':[sys.executable,'-O','scripts/verify_tracking_publication.py','--output',str(OUT/f'checks/{prefix}_publication_optimized.json')],
      'studies':[sys.executable,'scripts/verify_tracking_studies.py','--output',str(OUT/f'checks/{prefix}_studies')],
      'studies_optimized':[sys.executable,'-O','scripts/verify_tracking_studies.py','--output',str(OUT/f'checks/{prefix}_studies_optimized')],
      'boundary_history':[sys.executable,'-m','aimrl.tracking_planning.historical_check','--output',str(OUT/f'checks/{prefix}_boundary_history.json')],
      'boundary_history_optimized':[sys.executable,'-O','-m','aimrl.tracking_planning.historical_check','--output',str(OUT/f'checks/{prefix}_boundary_history_optimized.json')],
      'planning_validation_optimized':[sys.executable,'-O','-c',
            'from aimrl.tracking_planning.core import config,OUT; from aimrl.tracking_planning.validation import validate; validate(config(),OUT/"validation_optimized")'],
    }
    if args.stage=='postflight':commands.update(
      artifacts=[sys.executable,'-m','aimrl.tracking_planning','verify'],
      artifacts_optimized=[sys.executable,'-O','-m','aimrl.tracking_planning','verify'])
    result={}
    for name,command in commands.items():
        target=folder/f'{prefix}_{name}.txt';start=time.perf_counter()
        with target.open('w',encoding='utf-8') as stream:r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result[name]=dict(command=command,returncode=r.returncode,elapsed_seconds=time.perf_counter()-start,log=target.relative_to(OUT).as_posix())
        save_json(OUT/f'{args.stage}.json',result)
        print(f'{name}: exit {r.returncode}, {result[name]["elapsed_seconds"]:.2f}s',flush=True)
    return int(any(r['returncode']!=0 for r in result.values()))


if __name__=='__main__':raise SystemExit(main())
