"""Reproducible staged driver. No phase writes historical evidence or trains Q."""
import argparse,datetime,json,os,platform,subprocess,sys,time,importlib.metadata
from pathlib import Path
from ..tracking_research.io import ROOT,read_json,save_json,sha
from ..tracking_planning.core import model_path,frozen_agent
from ..tracking_planning.reference import require
from .core import OUT,CONFIG,LIBRARY,config,methods,precompute

def models():
    result=[]
    for seed in range(60,65):
        a=frozen_agent('R-Decay',seed);q,counts=a.snapshot()[:2];p=model_path('R-Decay',seed)
        result.append(dict(seed=seed,path=p.relative_to(ROOT).as_posix(),file_sha256=sha(p),q_sha256=q,counts_sha256=counts,shape=list(a.q.shape),updates_in_this_study=0))
    return result

def history_check():
    before=read_json(OUT/'historical_sha256_before.json');bad=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    require(not bad,'Historical file changed: '+str(bad[:8]))
    return dict(files_verified=len(before),changed=[],reference_commit='a5b83f569b8cd454a96403b4a4a5ef4e508e231d')

def source_hashes():
    paths=list((ROOT/'aimrl/tracking_action_resolution').glob('*.py'))+list((ROOT/'aimrl/tracking_action_resolution').glob('*.c'))
    paths+=[CONFIG,ROOT/'tests/test_tracking_action_resolution.py',ROOT/'scripts/check_tracking_action_resolution.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def setup(c,compiler):
    if not OUT.exists():
        OUT.mkdir(parents=True)
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only','a5b83f569b8cd454a96403b4a4a5ef4e508e231d'],cwd=ROOT,text=True).splitlines()
        save_json(OUT/'historical_sha256_before.json',{p:sha(ROOT/p) for p in paths})
        ranges=[];conflicts=[]
        new=[(c['planning'][k]['start'],c['planning'][k]['start']+c['planning'][k]['count']-1) for k in ('development','final','stress')]+[(s,s) for s in c['planning']['demo_seeds']]
        def walk(x,path,key=''):
            if isinstance(x,dict):
                if type(x.get('start')) is int and type(x.get('count')) is int:
                    a=x['start'];b=a+x['count']-1;ranges.append(dict(path=path,key=key,start=a,end=b))
                    if any(max(a,lo)<=min(b,hi) for lo,hi in new):conflicts.append(ranges[-1])
                for k,v in x.items():walk(v,path,key+'.'+k)
            elif isinstance(x,list):
                if 'seed' in key.lower():
                    for a in x:
                        if type(a) is int and any(lo<=a<=hi for lo,hi in new):conflicts.append(dict(path=path,key=key,seed=a))
                for v in x:walk(v,path,key)
        for path in paths:
            if path.endswith('.json') and any(k in path for k in ('config','protocol','seed_audit')):walk(read_json(ROOT/path),path)
        require(not conflicts,'Historical scene collision')
        save_json(OUT/'seed_audit.json',dict(historical_ranges=ranges,conflicts=conflicts,new_ranges={k:c['planning'][k] for k in ('development','final','stress','demo_seeds')}))
    require(not (OUT/'setup_complete.json').exists(),'Setup already exists')
    native=OUT/'native';native.mkdir(exist_ok=True);cache=ROOT/'.tmp/action-resolution-build';cache.mkdir(parents=True,exist_ok=True)
    command=[str(compiler),'cc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-shared',str(ROOT/'aimrl/tracking_action_resolution/kernel.c'),'-o',str(LIBRARY)]
    env=dict(os.environ,ZIG_LOCAL_CACHE_DIR=str(cache/'local'),ZIG_GLOBAL_CACHE_DIR=str(cache/'global'))
    start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,env=env)
    save_json(native/'build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
        elapsed_seconds=time.perf_counter()-start,compiler_version=subprocess.check_output([str(compiler),'version'],text=True).strip(),source_sha256=sha(ROOT/'aimrl/tracking_action_resolution/kernel.c')))
    require(r.returncode==0,r.stderr)
    save_json(OUT/'model_provenance.json',dict(models=models(),terminal='max over original ten Q_shaped actions + Phi(s), Phi=-L1. Identical for A10 and A14. No fabricated Q entries.',new_training_interactions=0,Q_updates=0))
    precompute(c);save_json(OUT/'setup_complete.json',dict(complete=True))

def verify_protocol(c):
    p=read_json(OUT/'protocol.json');require(p['config']==c and p['config_sha256']==sha(CONFIG),'Frozen config differs')
    for path,h in p['source_sha256'].items():require(sha(ROOT/path)==h,'Frozen source differs '+path)
    require(sha(LIBRARY)==p['kernel_sha256'],'Kernel changed')
    require(models()==p['models'],'Q or counts changed')
    for row in p['cache_bundles']:
        require(sha(ROOT/row['bundle'])==row['sha256'] and sha(ROOT/row['A10_reference'])==row['A10_sha256'],'Cache changed')
    return p

def freeze(c):
    require(not (OUT/'protocol.json').exists() and not (OUT/'evaluation/final').exists(),'Already frozen/accessed')
    require(read_json(OUT/'development_complete.json')['complete'],'Development incomplete')
    checks=read_json(OUT/'preflight/commands.json');require(all(x['returncode']==0 for x in checks.values()),'Preflight failed')
    for folder in ('validation','validation_O'):require(read_json(OUT/folder/'checks.json')['complete'],'Validation incomplete')
    require(read_json(OUT/'monotonicity.json')['violations']==0,'Monotonicity failed')
    save_json(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=c,config_sha256=sha(CONFIG),
        source_sha256=source_hashes(),kernel_sha256=sha(LIBRARY),models=models(),methods=methods(c),cache_bundles=read_json(OUT/'precomputation.json')['records'],
        history=history_check(),seed_audit_sha256=sha(OUT/'seed_audit.json'),
        information='Only current full physical state enters immutable lookup. No RNG, seed, future trajectory or draws enter planning.',
        planning='H=3 at every tick, including tick 300. A14 at all three levels. Exact float64 maximum stored, lowest action ID within absolute 1e-10 selected. No Fire pruning.',
        reward='task = base + 0.5*I(post distance<=1); task value = shaped value + Phi; same terminal from max of original ten frozen Q entries.',
        interventions='Only four one-tick +/-2 moves appended. Old IDs, clipping, reflection, movement-fire exclusion and three draws/tick preserved.',
        metrics=c['planning']['extra_metrics'],events=c['planning']['event_window'],behavior_selection=c['planning']['behavior_selection'],
        bootstrap=c['planning']['bootstrap'],bootstrap_replicates=c['planning']['bootstrap_replicates'],bootstrap_seed=c['planning']['bootstrap_seed'],
        thresholds=c['planning']['thresholds'],selection='All twelve instances retained. No reward/horizon/terminal/action tuning after Final.',
        runtime=dict(python=sys.version,platform=platform.platform(),dependencies={k:importlib.metadata.version(k) for k in ('numpy','matplotlib','Pillow')}),
        new_training_interactions=0,Q_updates=0,actual_player='Not performed; program decode only; no desktop/browser permission requested'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['setup','validate','development','freeze','evaluate','report','verify','inventory'])
    p.add_argument('--compiler',type=Path);p.add_argument('--output',type=Path);args=p.parse_args();c=config()
    if args.phase=='setup':setup(c,args.compiler)
    elif args.phase=='validate':
        from .validation import validate
        print(validate(c,args.output or OUT/'validation'))
    elif args.phase=='development':
        from .evaluation import evaluate_split,common_diagnostics
        r=evaluate_split(c,'development');common_diagnostics(c)
        save_json(OUT/'development_complete.json',dict(complete=True,**r,final_access=False))
    elif args.phase=='freeze':freeze(c)
    elif args.phase=='evaluate':
        from .evaluation import evaluate_split
        verify_protocol(c)
        result={s:evaluate_split(c,s) for s in ('final','stress')};save_json(OUT/'evaluation_complete.json',result)
    elif args.phase=='report':
        from .report import report
        verify_protocol(c);report(c)
    elif args.phase=='verify':
        from .verify import verify
        print(verify(c,args.output or OUT/('verification_O.json' if sys.flags.optimize else 'verification.json')))
    elif args.phase=='inventory':
        from .verify import inventory
        inventory()

if __name__=='__main__':main()
