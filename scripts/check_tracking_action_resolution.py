"""Execute and preserve real command results; never edit historical expectations."""
import argparse,os,re,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.tracking_action_resolution.core import OUT
from aimrl.tracking_research.io import save_json,sha

def clean_history_view():
    view=ROOT/'.tmp/arh';view.mkdir(parents=True,exist_ok=True)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','a5b83f569b8cd454a96403b4a4a5ef4e508e231d'],cwd=ROOT,text=True).splitlines()
    for relative in paths:
        src=ROOT/relative;dst=view/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            try:os.link(src,dst)
            except OSError:shutil.copyfile(src,dst)
        if sha(src)!=sha(dst):raise ValueError('Clean-view byte mismatch '+relative)
    save_json(OUT/'clean_history_view.json',dict(path='.tmp/arh',files=len(paths),content_equal=True,
        purpose='Exact published-file view excludes pre-existing ignored machine-local transcripts; original history checker and expected hashes unchanged. Linked/copied files are read-only checker inputs.'))
    return view

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','postflight']);args=p.parse_args()
    folder=OUT/args.phase;folder.mkdir(exist_ok=False);py=sys.executable;commands={}
    if args.phase=='preflight':
        commands['full_tests']=[py,'-m','unittest','discover','-s','tests','-v']
        commands['new_tests_O']=[py,'-O','-m','unittest','discover','-s','tests','-p','test_tracking_action_resolution.py','-v']
        for flags,suffix in [([],''),(['-O'],'_O')]:
            commands['new_validation'+suffix]=[py,*flags,'-m','aimrl.tracking_action_resolution','validate','--output',str(OUT/('validation'+suffix))]
            commands['historical_publication'+suffix]=[py,*flags,'scripts/verify_tracking_publication.py','--output',str(folder/f'historical_publication{suffix}.json')]
            for phase in ('history','boundary','planning'):
                script=str(clean_history_view()/'scripts/reproduce_latest_studies.py') if phase=='history' else 'scripts/reproduce_latest_studies.py'
                commands[phase+suffix]=[py,*flags,script,'verify-'+phase,'--output',str(folder/(phase+suffix))]
    else:
        for flags,suffix in [([],''),(['-O'],'_O')]:
            commands['new_artifacts'+suffix]=[py,*flags,'-m','aimrl.tracking_action_resolution','verify']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'.tmp/action-mpl'),
        AIMRL_BOUNDARY_KERNEL=str(ROOT/'.tmp/latest-release-native/boundary.dll'),AIMRL_TRADEOFF_KERNEL=str(ROOT/'.tmp/latest-release-native/tradeoff.dll'))
    for k in ('AIMRL_BOUNDARY_OUTPUT','AIMRL_PLANNING_OUTPUT'):env.pop(k,None)
    result={}
    for name,cmd in commands.items():
        start=time.perf_counter();log=folder/f'{name}.txt'
        with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        text=log.read_text(encoding='utf-8');match=re.search(r'Ran (\d+) tests?',text)
        result[name]=dict(command=cmd,returncode=r.returncode,elapsed_seconds=time.perf_counter()-start,test_count=int(match.group(1)) if match else None,log_sha256=sha(log))
        save_json(folder/'commands.json',result);print(name,result[name]['returncode'],result[name]['test_count'],flush=True)
    return int(any(x['returncode'] for x in result.values()))

if __name__=='__main__':raise SystemExit(main())
