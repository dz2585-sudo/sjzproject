"""Run actual release checks on a clean publication view; never rewrite evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.tracking_research.io import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--native-directory',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);native=args.native_directory.resolve()
    env=dict(os.environ,AIMRL_BOUNDARY_KERNEL=str(native/'boundary.dll'),AIMRL_TRADEOFF_KERNEL=str(native/'tradeoff.dll'),
             AIMRL_PLANNING_NATIVE='1',MPLCONFIGDIR=str(ROOT/'.tmp/latest-mpl'),PYTHONDONTWRITEBYTECODE='1')
    env.pop('AIMRL_BOUNDARY_OUTPUT',None);env.pop('AIMRL_PLANNING_OUTPUT',None)
    commands={
      'full_tests':[sys.executable,'-m','unittest','discover','-s','tests','-v'],
      'boundary_tests_O':[sys.executable,'-O','-m','unittest','discover','-s','tests','-p','test_tracking_boundary_stability.py','-v'],
      'planning_tests_O':[sys.executable,'-O','-m','unittest','discover','-s','tests','-p','test_tracking_planning.py','-v'],
      'publication_tests_O':[sys.executable,'-O','-m','unittest','discover','-s','tests','-p','test_latest_publication.py','-v'],
    }
    for flags,suffix in (([],''),(['-O'],'_O')):
        commands['historical_publication'+suffix]=[sys.executable,*flags,'scripts/verify_tracking_publication.py','--output',str(out/f'historical_publication{suffix}.json')]
        commands['historical_studies'+suffix]=[sys.executable,*flags,'scripts/reproduce_latest_studies.py','verify-history','--output',str(out/f'historical_studies{suffix}')]
        commands['boundary_artifacts'+suffix]=[sys.executable,*flags,'scripts/reproduce_latest_studies.py','verify-boundary','--output',str(out/f'boundary{suffix}')]
        commands['planning_artifacts'+suffix]=[sys.executable,*flags,'scripts/reproduce_latest_studies.py','verify-planning','--output',str(out/f'planning{suffix}')]
    commands['rebuilt_planning_all_DP']=[sys.executable,'scripts/reproduce_latest_studies.py','verify-planning','--planning-library',str(native/'planning.dll'),'--output',str(out/'rebuilt_planning')]
    commands['release_links_inputs']=[sys.executable,'-c',
        'import json; from aimrl.latest_publication import verify_release_inputs,check_links; print(json.dumps({"inputs":verify_release_inputs(),"links":check_links()},indent=2))']
    result={}
    for name,command in commands.items():
        log=out/f'{name}.txt';start=time.perf_counter()
        with log.open('w',encoding='utf-8') as stream:r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text(encoding='utf-8');match=re.search(r'Ran (\d+) tests?',text)
        result[name]=dict(command=command,returncode=r.returncode,elapsed_seconds=time.perf_counter()-start,
                          test_count=int(match.group(1)) if match else None,log=log.name,original_log_sha256=sha(log))
        save_json(out/'commands.json',result)
        print(f'{name}: exit {r.returncode}, tests={result[name]["test_count"]}, {result[name]["elapsed_seconds"]:.2f}s',flush=True)
    return int(any(r['returncode']!=0 for r in result.values()))


if __name__=='__main__':raise SystemExit(main())
