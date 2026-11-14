"""Execute gates, pilot, protocol freeze, exactly 100M formal updates and evaluation."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..reproduce import memory_record
from ..tracking25.env import TrackingEnv
from ..tracking_research.io import ROOT, read_json, save_csv, save_json, sha
from ..tracking_tradeoff.core import summarize, window_metrics
from .core import Agent, CONFIG, CONDITIONS, OUT, Kernel, config, load_model, require, save_model
from .diagnostics import common_states, diagnose, encoder_checks, mechanisms, parity

LIBRARY = OUT/'native/boundary.dll' if sys.platform=='win32' else OUT/'native/boundary.so'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def source_hashes():
    paths=list((ROOT/'aimrl/tracking_boundary_stability').glob('*.py'))+list((ROOT/'aimrl/tracking_boundary_stability').glob('*.c'))
    paths+=[CONFIG,ROOT/'tests/test_tracking_boundary_stability.py']
    paths+=[ROOT/'scripts/run_boundary_stability_checks.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def history_check(output=OUT):
    baseline=read_json(output/'historical_sha256_before.json')
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    require(not changed,f'Historical files changed: {changed[:10]}')
    return dict(files_verified=len(baseline),changed=changed,head=git('rev-parse','HEAD'),
                identity=dict(name=git('config','user.name'),email=git('config','user.email')))


def seed_audit(c):
    # Scan every tracked JSON declaration and every seed-bearing CSV, including
    # all prior tradeoff and guard files. Also inspect ignored original workspace.
    from scripts.check_tracking_tradeoff_seeds import integers
    candidate=set(c['study']['training_seeds']+[c['study']['pilot_seed']]+c['study']['demo_seeds'])
    for split in ('development','final','stress','diagnostic'):
        spec=c['study'][split];candidate.update(range(spec['start'],spec['start']+spec['count']))
    conflicts=[];counts=dict(json_files=0,csv_files=0,seed_values=0)
    # Original parent workspace contains the pre-publication experiments.
    original=ROOT.parents[1]
    paths=[(ROOT,p) for p in git('ls-files').splitlines()]
    for folder in ('configs','outputs','results'):
        for p in (original/folder).rglob('*'):
            if p.suffix in ('.json','.csv'): paths.append((original,p.relative_to(original).as_posix()))
    for root,relative in paths:
        if root==ROOT and (relative.startswith('results/tracking_boundary_stability/') or relative=='configs/tracking_boundary_stability.json'):
            continue  # Exact replications reuse this study's reservation, never other studies'.
        p=root/relative
        if p.suffix=='.json':
            try: data=json.loads(p.read_text(encoding='utf-8-sig'))
            except (ValueError,OSError):continue
            counts['json_files']+=1
            for values in integers(data):
                matches=sorted(candidate.intersection(values))
                if matches:conflicts.append(dict(root=str(root),path=relative,seeds=matches))
        elif p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.reader(f);head=next(reader,[])
                columns=[i for i,k in enumerate(head) if 'seed' in k.lower()]
                if not columns:continue
                counts['csv_files']+=1
                for row in reader:
                    for i in columns:
                        if i<len(row) and row[i].isdigit():
                            counts['seed_values']+=1
                            if int(row[i]) in candidate:
                                conflicts.append(dict(root=str(root),path=relative,seeds=[int(row[i])]))
    require(not conflicts,f'Seed collisions: {conflicts[:10]}')
    return dict(checked=counts,conflict_count=len(conflicts),conflicts=conflicts,candidate_seeds=sorted(candidate))


def setup(c, compiler):
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'setup_complete.json').exists(),'Setup already completed')
    if not (OUT/'historical_sha256_before.json').exists():
        baseline={p:sha(ROOT/p) for p in git('ls-files').splitlines()}
        save_json(OUT/'historical_sha256_before.json',baseline)
        save_json(OUT/'workspace.json',dict(head=git('rev-parse','HEAD'),remote=git('remote','get-url','origin'),
              initial_status=git('status','--short'),identity=dict(name=git('config','user.name'),email=git('config','user.email')),
              root=str(ROOT),agents_files=[],started_utc=datetime.now(timezone.utc).isoformat()))
    save_json(OUT/'seed_audit.json',seed_audit(c))
    LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'.tmp/boundary-build';cache.mkdir(parents=True,exist_ok=True)
    command=[str(compiler),'cc','-shared','-O3','-std=c11','-ffp-contract=off','-fno-fast-math',
             str(ROOT/'aimrl/tracking_boundary_stability/kernel.c'),'-o',str(LIBRARY)]
    environment=dict(os.environ,ZIG_LOCAL_CACHE_DIR=str(cache/'local'),ZIG_GLOBAL_CACHE_DIR=str(cache/'global'))
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,env=environment)
    buildfile=OUT/'native/build.json'
    if buildfile.exists():
        previous=read_json(buildfile);save_json(OUT/'native'/f'build_attempt_{len(list(LIBRARY.parent.glob("build_attempt_*.json")))+1}.json',previous)
    save_json(buildfile,dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
              compiler_version=subprocess.check_output([str(compiler),'version'],text=True).strip()))
    require(result.returncode==0,f'Compiler failed: {result.stderr}')
    kernel=Kernel(LIBRARY,c)
    encoder_checks(OUT);mechanisms(c,OUT);parity(c,kernel,OUT);common_states(c,OUT)
    save_json(OUT/'setup_complete.json',dict(complete=True,history=history_check(),memory=memory_record()))


def check_protocol(c):
    p=read_json(OUT/'protocol.json')
    require(p['config']==c and p['config_sha256']==sha(CONFIG),'Frozen configuration changed')
    require(p['kernel_sha256']==sha(LIBRARY),'Frozen native binary changed')
    for name,digest in p['source_sha256'].items():
        require(sha(ROOT/name)==digest,'Frozen source changed: '+name)
    return p


def evaluate(c,kernel,agent,seeds,guard=True,detailed=False,keep=False):
    before=agent.snapshot();rows=[];traces={}
    env=TrackingEnv(c)
    for seed in seeds:
        initial,_=env.reset(seed)
        ticks=kernel.run(env,initial,300,agent,guard=guard)
        rows.append(dict(scenario_seed=seed,**window_metrics(initial,ticks,detailed)))
        if keep:traces[seed]=(initial,ticks)
    require(agent.snapshot()==before,'Evaluation mutated learner')
    return rows,traces


def run_train(c,kernel,condition,seed,budget,pilot=False):
    started=time.perf_counter()
    root=OUT/('pilot' if pilot else 'runs')/condition/f'seed_{seed}'
    root.mkdir(parents=True,exist_ok=False)
    save_json(root/'run_config.json',c)
    seq=np.random.SeedSequence(seed).spawn(2)
    agent=Agent(c,condition,np.random.default_rng(seq[1]));env=TrackingEnv(c)
    state,_=env.reset(int(seq[0].generate_state(1,dtype=np.uint64)[0]))
    z=np.load(OUT/'diagnostics/common_states.npz',allow_pickle=False)
    states,frequency=z['states'],z['frequency'];z.close()
    summaries=[];diagnostics=[];previous=None
    interval=c['study']['checkpoint_interval']
    spec=c['study']['development'];seeds=range(spec['start'],spec['start']+spec['count'])
    bins=np.linspace(0,.15,31);hist=np.zeros(30,dtype=np.uint64)
    amin,amax,asum=1.0,0.0,0.0
    target_hasher=hashlib.sha256();executed=0;full=0;partial=0

    def checkpoint(step):
        nonlocal previous
        path=root/'checkpoints'/f'step_{step:07d}.npz'
        save_model(path,agent,step,c,training_seed=seed,pilot=pilot)
        rows,_=evaluate(c,kernel,agent,seeds,guard=True)
        save_csv(root/'development'/f'windows_{step:07d}.csv',rows)
        summaries.append(dict(step=step,checkpoint_sha256=sha(path),**summarize(rows)))
        diag,previous=diagnose(c,agent,states,frequency,step,previous,
                 root/'diagnostic_states_final.csv' if step==budget else None)
        diagnostics.extend(diag)
        save_csv(root/'development.csv',summaries)
        save_csv(root/'q_diagnostics.csv',diagnostics)

    checkpoint(0)
    with (root/'training_windows.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['end_step','ticks','hits','misses','distance_sum','close_ticks',
                                          'base_return','close_term','potential_term','training_return','truncated','budget_tail'])
        writer.writeheader()
        while executed<budget:
            initial=state;length=min(300,budget-executed)
            target_hasher.update(np.asarray(initial[2:],dtype=np.int16).tobytes())
            remaining=length;chunks=[];audits=[]
            while remaining:
                next_cp=(executed//interval+1)*interval
                take=min(remaining,next_cp-executed)
                ticks,audit=kernel.run(env,state,take,agent,executed,train=True,debug=True)
                chunks.append(ticks);audits.append(audit)
                target_hasher.update(ticks[:,2:6].tobytes())
                state=tuple(map(int,ticks[-1,:6]));executed+=take;remaining-=take
                alpha=audit[:,4];amin=min(amin,float(alpha.min()));amax=max(amax,float(alpha.max()));asum+=float(alpha.sum())
                hist+=np.histogram(alpha,bins=bins)[0].astype(np.uint64)
                if executed%interval==0 or executed==budget:checkpoint(executed)
            ticks=np.vstack(chunks);audit=np.vstack(audits)
            m=window_metrics(initial,ticks)
            writer.writerow(dict(end_step=executed,**{k:m[k] for k in ('ticks','hits','misses','distance_sum','close_ticks','base_return')},
                        close_term=float(audit[:,1].sum()),potential_term=float(audit[:,2].sum()),
                        training_return=float(audit[:,3].sum()),truncated=env.finished,budget_tail=length<300))
            full+=length==300;partial+=length<300
            if executed<budget:state,_=env.reset()
    require(int(agent.counts.sum(dtype=np.uint64))==budget and int(hist.sum())==budget,'Actual update / alpha accounting failed')
    require(np.isfinite(agent.q).all(),'Nonfinite Q table')
    final=root/'checkpoints'/f'step_{budget:07d}.npz'
    loaded,metadata=load_model(final,c)
    require(np.array_equal(loaded.q,agent.q) and np.array_equal(loaded.counts,agent.counts),'Checkpoint exact reload failed')
    metadata=dict(condition=condition,seed=seed,pilot=pilot,interactions=executed,updates=int(agent.counts.sum(dtype=np.uint64)),
                  full_windows=full,partial_windows=partial,tail_ticks=budget%300,checkpoints=len(summaries),
                  elapsed_seconds=time.perf_counter()-started,memory=memory_record(),target_stream_sha256=target_hasher.hexdigest(),
                  environment_rng_final=env.rng.bit_generator.state,action_rng_final=agent.rng.bit_generator.state,
                  alpha=dict(minimum=amin,maximum=amax,mean=asum/budget,bins=bins.tolist(),actual_update_histogram=hist.tolist()),
                  encoder=agent.encoder.record(),reload_exact=True,model_sha256=sha(final))
    save_json(root/'metadata.json',metadata)
    print(f'{condition} seed {seed}: {executed:,} interactions, {metadata["elapsed_seconds"]:.2f}s, '
          f'alpha {amin:.6f}..{amax:.6f}',flush=True)
    return metadata


def pilot(c,kernel):
    require(read_json(OUT/'setup_complete.json')['complete'],'Setup not complete')
    runs=[run_train(c,kernel,condition,c['study']['pilot_seed'],c['study']['pilot_steps'],True) for condition in CONDITIONS]
    require(len({r['target_stream_sha256'] for r in runs})==1,'Pilot target streams are not paired')
    save_json(OUT/'pilot_complete.json',dict(complete=True,runs=runs,interactions=sum(r['interactions'] for r in runs),
              final_access=False,selection=False,history=history_check()))


def train(c,kernel):
    p=read_json(OUT/'pilot_complete.json')
    require(p['complete'] and p['interactions']==4000000,'Four 1M pilot runs required')
    preflight=read_json(OUT/'preflight.json')
    require(set(preflight)=={'full_tests','optimized_new_tests','publication','publication_optimized','studies','studies_optimized'},
            'Preflight command matrix is incomplete')
    require(all(v['returncode']==0 for v in preflight.values()),'Preflight failed')
    require(not (OUT/'protocol.json').exists(),'Formal protocol already exists')
    save_json(OUT/'protocol.json',dict(config=c,config_sha256=sha(CONFIG),source_sha256=source_hashes(),kernel_sha256=sha(LIBRARY),
              frozen_utc=datetime.now(timezone.utc).isoformat(),git_head=git('rev-parse','HEAD'),
              runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                           dependencies={k:importlib.metadata.version(k) for k in ('numpy','matplotlib','Pillow')}),
              seed_audit_sha256=sha(OUT/'seed_audit.json'),common_states_sha256=sha(OUT/'diagnostics/common_states.npz'),
              rng='SeedSequence(seed).spawn(2): first child generates uint64 reset seed; second owns action PCG64. Three target doubles per tick, unchanged order.',
              state='R: relative dx,dy,heading,speed. B: append both objects x/y boundary groups, valid axis-key enumeration. B remains approximate aggregation.',
              update='Float32 Q and increments; float64 reward, target and alpha. Counts uint32, increment after each actual update. No guard in training selection or bootstrap.',
              bootstrap='Actual next state including window truncation and partial budget tail. Continuing-task semantics.',
              selection='Every main result uses exactly step 5,000,000. No condition, checkpoint or seed selection. Demo seed 60.',
              primary='Guarded 5M models against paired R-Fixed. Raw models study shooting ranking. Final and Stress reported separately.',
              inference='Five paired training seeds are the independent training units. Sample SD ddof=1. No tick-level confidence intervals. Descriptive policy-specific event slices.',
              factorial='Bfixed-Rfixed; Bdecay-Rdecay; Rdecay-Rfixed; Bdecay-Bfixed; interaction=(Bdecay-Bfixed)-(Rdecay-Rfixed).',
              event_definitions=c['study']['event_window'],diagnostic_weighting='Distinct physical states equal-weight and frequency of fixed common-state sampling events; exact conditional six-outcome Bellman expectation for Stay and unaligned Fire.',
              history=history_check(),pilot_interactions=4000000,formal_interactions_planned=100000000))
    started=time.perf_counter();runs=[]
    for seed in c['study']['training_seeds']:
        paired=[run_train(c,kernel,condition,seed,c['learning']['training_steps']) for condition in CONDITIONS]
        require(len({r['target_stream_sha256'] for r in paired})==1,'Formal target streams are not paired')
        runs.extend(paired)
    require(sum(r['interactions'] for r in runs)==100000000 and len(runs)==20,'Formal budget differs')
    check_protocol(c)
    save_json(OUT/'training_complete.json',dict(complete=True,runs=runs,interactions=100000000,updates=100000000,
              elapsed_seconds=time.perf_counter()-started,history=history_check()))


def evaluation(c,kernel):
    check_protocol(c)
    require(read_json(OUT/'training_complete.json')['complete'],'Formal training is incomplete')
    from .analysis import evaluate_all
    evaluate_all(c,kernel)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('setup','pilot','train','evaluate','report','verify','inventory'))
    parser.add_argument('--compiler',type=Path)
    args=parser.parse_args();c=config()
    if args.phase=='setup':setup(c,args.compiler)
    elif args.phase=='inventory':
        from .verify import inventory
        inventory()
    elif args.phase=='report':
        from .report import report
        report(c)
    elif args.phase=='verify':
        from .verify import verify
        print(json.dumps(verify(c),indent=2))
    else:
        kernel=Kernel(LIBRARY,c)
        if args.phase=='pilot':pilot(c,kernel)
        elif args.phase=='train':train(c,kernel)
        elif args.phase=='evaluate':evaluation(c,kernel)


if __name__=='__main__':main()
