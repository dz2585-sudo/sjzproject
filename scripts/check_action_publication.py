"""Release checks on a clean checkout; writes only a fresh output directory."""
import argparse,os,subprocess,sys,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aimrl.tracking_research.io import save_json,sha

def build(compiler,native):
    native.mkdir(parents=True,exist_ok=False);env=dict(os.environ,ZIG_LOCAL_CACHE_DIR=str(native/'local-cache'),ZIG_GLOBAL_CACHE_DIR=str(native/'global-cache'))
    records=[];suffix='.dll' if os.name=='nt' else '.so';version=subprocess.check_output([str(compiler),'version'],text=True).strip()
    for name,module in [('tradeoff','tracking_tradeoff'),('boundary','tracking_boundary_stability'),('planning','tracking_planning'),('action_resolution','tracking_action_resolution')]:
        source=ROOT/'aimrl'/module/'kernel.c';target=native/(name+suffix)
        command=[str(compiler),'cc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-shared',str(source),'-o',str(target)]
        start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,env=env,cwd=ROOT)
        entry=dict(name=name,source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,elapsed_seconds=time.perf_counter()-start,compiler_version=version)
        if r.returncode:save_json(native/'failure.json',entry);raise RuntimeError(r.stderr)
        entry['library_sha256']=sha(target);records.append(entry);print('Built',name,flush=True)
    save_json(native/'build.json',dict(builds=records,compiler_version=version))

def main():
    p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,default=Path('zig'));p.add_argument('--output',type=Path,required=True);p.add_argument('--native-directory',type=Path);p.add_argument('--build-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    native=a.native_directory.resolve() if a.native_directory else out/'native'
    if not a.native_directory:build(a.compiler,native)
    if a.build_only:return 0
    commands={};py=sys.executable
    commands['full_tests']=[py,'-m','aimrl.action_publication','tests','--output',str(out/'full_tests')]
    commands['action_tests_O']=[py,'-O','-m','aimrl.action_publication','tests-O','--output',str(out/'action_tests_O')]
    commands['release_tests_O']=[py,'-O','-m','unittest','discover','-s','tests','-p','test_action_publication.py','-v']
    for flags,suffix in [([],''),(['-O'],'_O')]:
        commands['representation'+suffix]=[py,*flags,'scripts/verify_tracking_publication.py','--output',str(out/('representation'+suffix+'.json'))]
        for phase in ('history','boundary','planning','action'):
            commands[phase+suffix]=[py,*flags,'-m','aimrl.action_publication',phase,'--output',str(out/(phase+suffix)),'--native-directory',str(native)]
    commands['saved_trace_GIF_pixels']=[py,'-m','aimrl.action_publication','gifs','--output',str(out/'gifs')]
    commands['protected_inputs_links']=[py,'-m','aimrl.action_publication','inputs','--output',str(out/'inputs')]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'mpl-cache'),AIMRL_PLANNING_NATIVE='1',AIMRL_BOUNDARY_KERNEL=str(native/'boundary.dll'),AIMRL_TRADEOFF_KERNEL=str(native/'tradeoff.dll'))
    for k in ('AIMRL_BOUNDARY_OUTPUT','AIMRL_PLANNING_OUTPUT','AIMRL_ACTION_OUTPUT'):env.pop(k,None)
    records={}
    for name,command in commands.items():
        start=time.perf_counter();path=out/(name+'.txt')
        with path.open('w',encoding='utf-8') as stream:r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        text=path.read_text(encoding='utf-8');match=re.search(r'Ran (\d+) tests?',text)
        records[name]=dict(command=command,returncode=r.returncode,test_count=int(match.group(1)) if match else None,elapsed_seconds=time.perf_counter()-start,log_sha256=sha(path))
        save_json(out/'commands.json',records);print(name,':',r.returncode,'tests=',records[name]['test_count'],flush=True)
    return int(any(x['returncode'] for x in records.values()))

if __name__=='__main__':raise SystemExit(main())
