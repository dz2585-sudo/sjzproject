"""Portable entry points; frozen evidence is never rewritten by verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def build(args):
    from aimrl.tracking_research.io import save_json,sha
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    environment=dict(os.environ,ZIG_LOCAL_CACHE_DIR=str(output/'local-cache'),ZIG_GLOBAL_CACHE_DIR=str(output/'global-cache'))
    version=subprocess.check_output([str(args.compiler),'version'],text=True).strip()
    results=[];suffix='.dll' if os.name=='nt' else '.so'
    for name,source in (('tradeoff','aimrl/tracking_tradeoff/kernel.c'),('boundary','aimrl/tracking_boundary_stability/kernel.c'),('planning','aimrl/tracking_planning/kernel.c')):
        target=output/(name+suffix)
        command=[str(args.compiler),'cc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-shared',str(ROOT/source),'-o',str(target)]
        start=time.perf_counter();r=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True)
        result=dict(name=name,source=source,source_sha256=sha(ROOT/source),command=command,returncode=r.returncode,stderr=r.stderr,
                    elapsed_seconds=time.perf_counter()-start,compiler_version=version)
        if r.returncode:save_json(output/'build_failure.json',result);raise RuntimeError('Native rebuild failed: '+r.stderr[:2000])
        result.update(library=target.name,library_sha256=sha(target));results.append(result)
        print(f'Rebuilt {target.name}: {result["elapsed_seconds"]:.2f}s',flush=True)
    save_json(output/'build.json',dict(compiler_version=version,builds=results,flags='C11 -O3 -fno-fast-math -ffp-contract=off -shared'))


def verify_boundary(args):
    from aimrl.latest_publication import boundary_verify
    boundary_verify(args.output)


def verify_planning(args):
    from aimrl.latest_publication import planning_verify
    planning_verify(args.output,args.planning_library)


def verify_history(args):
    from aimrl.latest_publication import historical_studies_verify
    historical_studies_verify(args.output)


def reevaluate(args):
    from aimrl.tracking_research.io import read_json,save_csv,save_json
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if args.study=='boundary':
        from aimrl.tracking_boundary_stability import core,study
        c=core.config();cc=json.loads(json.dumps(c));spec=c['study'][args.split]
        if args.split=='stress':cc['environment'].update(turn_probability=.1,speed_toggle_probability=.1)
        kernel=core.Kernel(args.boundary_library or study.LIBRARY,c);result=[]
        for condition in core.CONDITIONS:
            for seed in c['study']['training_seeds']:
                agent,_=core.load_model(core.OUT/'runs'/condition/f'seed_{seed}/checkpoints/step_5000000.npz',c)
                for guard in (False,True):
                    rows,_=study.evaluate(cc,kernel,agent,range(spec['start'],spec['start']+spec['count']),guard=guard,detailed=True)
                    label=f'{condition}_seed{seed}_'+('guard' if guard else 'original')
                    save_csv(output/f'{label}_windows.csv',rows)
                    result.append(dict(condition=condition,seed=seed,guard=guard,windows=len(rows)))
        save_json(output/'evaluation.json',dict(study='boundary',split=args.split,models=result,Q_updates=0,training_interactions=0))
    else:
        from aimrl.tracking_planning import core,evaluation
        if args.planning_library:core.LIBRARY=args.planning_library.resolve()
        # Only the output module path changes; immutable policies/caches still
        # resolve inside the published original planning study.
        evaluation.OUT=output
        evaluation.evaluate_split(core.config(),args.split)
    print('Fresh frozen-model evaluation completed in a new output directory.',flush=True)


def train_boundary(args):
    # Exact replication of an already published protocol intentionally reuses
    # its original reservation. This does not claim a new untouched holdout.
    os.environ['AIMRL_BOUNDARY_OUTPUT']=str(args.output.resolve())
    from aimrl.tracking_research.io import read_json,sha
    from aimrl.tracking_boundary_stability import core,study
    c=core.config();original=read_json(ROOT/'results/tracking_boundary_stability/protocol.json')
    if original['config']!=c:raise ValueError('Exact replication configuration differs from frozen original')
    if args.phase=='setup':
        def reservation(config):
            if config!=original['config']:raise ValueError('Replication reservation differs')
            return dict(exact_replication=True,original_protocol_sha256=sha(ROOT/'results/tracking_boundary_stability/protocol.json'),
              original_seed_audit_sha256=sha(ROOT/'results/tracking_boundary_stability/seed_audit.json'),
              original_audit=read_json(ROOT/'results/tracking_boundary_stability/seed_audit.json'),
              reason='Exact replication intentionally reuses original training/scenario seeds; planning only references those frozen models. Not an independent new holdout.')
        study.seed_audit=reservation;study.setup(c,args.compiler)
    else:
        kernel=core.Kernel(study.LIBRARY,c)
        if args.phase=='pilot':study.pilot(c,kernel)
        elif args.phase=='train':study.train(c,kernel)
        elif args.phase=='evaluate':study.evaluation(c,kernel)
        else:
            from aimrl.tracking_boundary_stability.report import report
            report(c)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase_name',required=True)
    p=sub.add_parser('build');p.add_argument('--compiler',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.set_defaults(run=build)
    for name,run in (('verify-boundary',verify_boundary),('verify-planning',verify_planning),('verify-history',verify_history)):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True);p.add_argument('--planning-library',type=Path);p.set_defaults(run=run)
    p=sub.add_parser('reevaluate');p.add_argument('--study',choices=('boundary','planning'),required=True);p.add_argument('--split',choices=('final','stress'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--boundary-library',type=Path);p.add_argument('--planning-library',type=Path);p.set_defaults(run=reevaluate)
    p=sub.add_parser('train-boundary');p.add_argument('--phase',choices=('setup','pilot','train','evaluate','report'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--compiler',type=Path);p.set_defaults(run=train_boundary)
    args=parser.parse_args();args.run(args)


if __name__=='__main__':main()
