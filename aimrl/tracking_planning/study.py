"""Protect historical work, build shared DP, validate, freeze, then evaluate."""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from ..tracking_research.io import ROOT,read_json,save_csv,save_json,sha
from .core import CONFIG,OUT,LIBRARY,PREVIOUS,Native,cache_key,config,frozen_agent,methods,model_path,precompute
from .reference import require


def protect():
    original=read_json(OUT/'protected_sha256_before.json')
    changed=[p for p,h in original.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    require(not changed,'Protected file changed: '+str(changed[:10]))
    return dict(files_verified=len(original),changed=changed,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


def initialize():
    OUT.mkdir(parents=True,exist_ok=False)
    paths={p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p}
    paths.update(p.relative_to(ROOT).as_posix() for p in PREVIOUS.rglob('*') if p.is_file())
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'aimrl/tracking_boundary_stability').rglob('*') if p.suffix in ('.py','.c'))
    paths.update(['configs/tracking_boundary_stability.json','tests/test_tracking_boundary_stability.py','scripts/run_boundary_stability_checks.py'])
    save_json(OUT/'protected_sha256_before.json',{p:sha(ROOT/p) for p in sorted(paths)})
    save_json(OUT/'workspace.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
       head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
       identity={k:subprocess.check_output(['git','config','user.'+k],cwd=ROOT,text=True).strip() for k in ('name','email')},
       protected_files=len(paths),agents_files=[]))


def seed_audit(c):
    from scripts.check_tracking_tradeoff_seeds import integers
    candidates=set(c['planning']['demo_seeds'])
    for split in ('development','final','stress'):
        spec=c['planning'][split];candidates.update(range(spec['start'],spec['start']+spec['count']))
    conflicts=[];stats=dict(json_files=0,csv_files=0,seed_values=0)
    for relative in read_json(OUT/'protected_sha256_before.json'):
        if relative=='configs/tracking_planning.json' or relative.startswith('results/tracking_planning/'):
            continue  # Only this exact study's seed reservation is reusable in a replication.
        path=ROOT/relative
        if path.suffix=='.json':
            try:data=read_json(path)
            except (ValueError,OSError):continue
            stats['json_files']+=1
            for values in integers(data):
                matches=sorted(candidates.intersection(values))
                if matches:conflicts.append(dict(path=relative,seeds=matches))
        elif path.suffix=='.csv':
            with path.open(newline='',encoding='utf-8-sig') as f:
                reader=csv.reader(f);header=next(reader,[]);cols=[i for i,k in enumerate(header) if 'seed' in k.lower()]
                if not cols:continue
                stats['csv_files']+=1
                for row in reader:
                    for col in cols:
                        if col<len(row) and row[col].isdigit():
                            stats['seed_values']+=1
                            if int(row[col]) in candidates:conflicts.append(dict(path=relative,seeds=[int(row[col])]))
    require(not conflicts,'Scenario seed collision: '+str(conflicts[:10]))
    return dict(checked=stats,conflicts=conflicts,conflict_count=0,ranges=c['planning'],reservation='Exact replication may reuse this study reservation; other protected studies were all checked.')


def provenance(c):
    old=read_json(ROOT/'configs/tracking_boundary_stability.json');protocol=read_json(PREVIOUS/'protocol.json')
    require(protocol['config']==old,'Previous frozen protocol/config differ')
    for name,digest in protocol['source_sha256'].items():require(sha(ROOT/name)==digest,'Previous frozen source differs: '+name)
    source=[]
    for condition in ('R-Fixed','R-Decay'):
        for seed in c['planning']['training_seeds']:
            agent=frozen_agent(condition,seed)
            metadata=read_json(PREVIOUS/'runs'/condition/f'seed_{seed}/metadata.json')
            require(metadata['model_sha256']==sha(model_path(condition,seed)) and metadata['interactions']==5000000
                    and metadata['updates']==5000000,'Previous model metadata mismatch')
            source.append(dict(condition=condition,seed=seed,step=5000000,path=model_path(condition,seed).relative_to(ROOT).as_posix(),
                               file_sha256=sha(model_path(condition,seed)),q_sha256=agent.snapshot()[0],counts_sha256=agent.snapshot()[1]))
    raw=read_json(PREVIOUS/'per_seed.json')
    from ..tracking_boundary_stability.analysis import aggregate
    reconciled=aggregate(raw)
    require(reconciled==read_json(PREVIOUS/'comparison.json'),'Previous per-seed/comparison discrepancy')
    save_json(OUT/'model_provenance.json',dict(models=source,previous_config_sha256=sha(ROOT/'configs/tracking_boundary_stability.json'),
              previous_protocol_sha256=sha(PREVIOUS/'protocol.json'),previous_recomputed_comparison=reconciled,
              conversion='Previous reward = task + gamma*Phi(next)-Phi(state); scale=1. Therefore V_task = max(Q_shaped)+Phi.'))
    return source


def build(c,compiler):
    (OUT/'native').mkdir(exist_ok=True);cache=ROOT/'.tmp/planning-build';cache.mkdir(parents=True,exist_ok=True)
    command=[str(compiler),'cc','-shared','-O3','-std=c11','-ffp-contract=off','-fno-fast-math',str(ROOT/'aimrl/tracking_planning/kernel.c'),'-o',str(LIBRARY)]
    environment=dict(os.environ,ZIG_LOCAL_CACHE_DIR=str(cache/'local'),ZIG_GLOBAL_CACHE_DIR=str(cache/'global'))
    start=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT,env=environment)
    save_json(OUT/'native/build.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
               elapsed_seconds=time.perf_counter()-start,compiler_version=subprocess.check_output([str(compiler),'version'],text=True).strip()))
    require(result.returncode==0,'Compiler failed: '+result.stderr[:3000])


def setup(c,compiler):
    require(not (OUT/'setup_complete.json').exists(),'Setup already completed')
    save_json(OUT/'seed_audit.json',seed_audit(c));provenance(c);build(c,compiler);precompute(c)
    save_json(OUT/'setup_complete.json',dict(complete=True,new_training_interactions=0,protection=protect()))


def sources():
    paths=list((ROOT/'aimrl/tracking_planning').glob('*.py'))+list((ROOT/'aimrl/tracking_planning').glob('*.c'))
    paths+=[CONFIG,ROOT/'tests/test_tracking_planning.py',ROOT/'scripts/check_tracking_planning.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def verify_protocol(c):
    p=read_json(OUT/'protocol.json')
    require(p['config']==c and p['config_sha256']==sha(CONFIG),'Frozen planning config differs')
    require(p['kernel_sha256']==sha(LIBRARY),'Frozen native kernel differs')
    for relative,digest in p['source_sha256'].items():require(sha(ROOT/relative)==digest,'Frozen planning source differs: '+relative)
    for m in p['models']:require(sha(ROOT/m['path'])==m['file_sha256'],'Frozen input model changed')
    for r in p['cache_bundles']:require(sha(OUT/r['path'])==r['sha256'],'Frozen DP cache changed')
    return p


def development(c):
    from .validation import validate
    from .evaluation import evaluate_split,common_diagnostics,select_behaviors
    from .visuals import render_demo
    require(read_json(OUT/'setup_complete.json')['complete'],'Setup incomplete')
    validate(c)
    evaluate_split(c,'development')
    common_diagnostics(c)
    select_behaviors(c,'development')
    # Layout and full-301-frame decoding are tested on a Development route, not Demo/Final.
    render_demo(c,c['planning']['development']['start'],OUT/'development_demo',development=True)
    save_json(OUT/'development_complete.json',dict(complete=True,scenarios=200,policy_instances=20,
              windows=4000,environment_ticks=1200000,new_training_interactions=0,final_access=False,selection=False,protection=protect()))


def freeze(c):
    require(read_json(OUT/'development_complete.json')['complete'],'Development must finish first')
    tests=read_json(OUT/'preflight.json')
    require(len(tests)==9 and all(r['returncode']==0 for r in tests.values()),'All nine preflight commands must pass')
    require(not (OUT/'protocol.json').exists() and not (OUT/'evaluation/final').exists(),'Protocol or Final already opened')
    pre=read_json(OUT/'precomputation.json')
    save_json(OUT/'protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),config=c,config_sha256=sha(CONFIG),
       source_sha256=sources(),kernel_sha256=sha(LIBRARY),models=read_json(OUT/'model_provenance.json')['models'],methods=methods(c),
       cache_bundles=[dict(path=r['bundle'],sha256=r['bundle_sha256'],cache_key=r['cache_key'],horizon=r['horizon']) for r in pre['records']],
       runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),dependencies={k:importlib.metadata.version(k) for k in ('numpy','matplotlib','Pillow')}),
       information='Planners observe full current physical state and fixed known 0.05/0.05 model; R observes relative offsets, heading and speed. No planner receives RNG, seed, future route or actual future draws.',
       feedback='Bellman maximum is inside each successor subproblem. Only root action is executed, then a new observed state is queried. Full physical-state DP tables share chance expectations across cursor-action preimages.',
       tie_rule='Planning: exact maximum value, choose lowest ID within absolute 1e-10 of maximum. Frozen R: exact lowest-ID argmax. Rule: preserve original distance/displacement/action lexicographic rule.',
       actions='All ten actions enumerated in every native planning backup; no Fire pruning. Every executed action receives identical firing guard.',
       bootstrap='Same horizon H at every tick, including ticks 298-300. The 300-tick window is not a task termination.',
       training='Zero new training interactions, zero Q updates. Planning backups, chance terms, evaluations and validations counted separately.',
       main_comparisons='Within Phi family H1/H2/H3 isolates horizon. H3 vs Zero tests heuristic. Hybrid uses five distinct frozen R-Decay Q terminals and is paired by seed. Pure planners are single instances, never five independent training seeds.',
       metrics='Historical post-motion D/C and hit/shot accounting; current observed speed and actual post-toggle speed are separate slices. Zero-shot A is NA. No tick-level training inference.',
       inference='Five frozen learned models are training units; pure planner has no training-seed SD. Pair scenarios and keep all methods. No confidence intervals or significance claims.',
       selection='No selection of model, horizon, terminal coefficient, scenario or method after Final. Main tables include all 20 fixed instances.',
       behavior_selection=c['planning']['behavior_selection'],event_definitions=c['planning']['event_window'],
       actual_player='Not performed by request. Full programmatic GIF decoding and timing/pixel checks only.',protection=protect()))
    print('Frozen all methods, models, caches and scenario ranges before Final.',flush=True)


def evaluate(c):
    from .evaluation import evaluate_split,select_behaviors
    verify_protocol(c);require(not (OUT/'evaluation/final').exists(),'Final evaluation already exists')
    start=time.perf_counter()
    for split in ('final','stress'):
        evaluate_split(c,split);select_behaviors(c,split)
    save_json(OUT/'evaluation_complete.json',dict(complete=True,final_windows=10000,stress_windows=4000,
              environment_ticks=4200000,elapsed_seconds=time.perf_counter()-start,new_training_interactions=0,Q_updates=0,protection=protect()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('initialize','setup','development','freeze','evaluate','report','verify','inventory'))
    p.add_argument('--compiler',type=Path);args=p.parse_args();c=config()
    if args.phase=='initialize':initialize()
    elif args.phase=='setup':setup(c,args.compiler)
    elif args.phase=='development':development(c)
    elif args.phase=='freeze':freeze(c)
    elif args.phase=='evaluate':evaluate(c)
    elif args.phase=='report':
        from .report import report
        report(c)
    elif args.phase=='verify':
        from .verify import verify
        print(json.dumps(verify(c),indent=2))
    elif args.phase=='inventory':
        from .verify import inventory
        inventory()


if __name__=='__main__':main()
