"""Freeze, gate and evaluate the original and guarded policies without learning."""
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking25.policies import rule_policy
from ..tracking_research.io import ROOT, read_json, save_csv, save_json, sha
from ..tracking_tradeoff.core import Kernel, load_model
from .history import PRIOR, SPLITS, historical_hashes, previous, rows, seed_audit, values, verify_history
from .replay import replay_pair, require

SOURCE=('__init__.py','__main__.py','history.py','replay.py','study.py')


def model_path(condition, seed):
    return PRIOR/'runs'/condition/f'seed_{seed}'/'checkpoints/step_5000000.npz'


def load_frozen(config, condition, seed, expected):
    path=model_path(condition,seed)
    require(sha(path)==expected,'Frozen checkpoint hash differs')
    agent=load_model(path,config,condition,seed,5000000)
    agent.q.flags.writeable=False
    agent.counts.flags.writeable=False
    return agent


def setup(output):
    expected=historical_hashes()
    config, record=previous(output,expected)
    collision=seed_audit(config)
    models=[{'condition':c,'training_seed':s,'sha256':expected[model_path(c,s).relative_to(PRIOR).as_posix()]}
            for c in config['tradeoff']['conditions'] for s in config['tradeoff']['training_seeds']]
    save_json(output/'prior_sha256_before.json',expected)
    save_json(output/'protocol.json',{'config':config,'history':record,'seed_audit':collision,'models':models,
              'guard':'Query original policy once. Execute Stay only for Fire with d_pre > 0.',
              'evaluation_backend':'Two independent Python TrackingEnv instances per scenario. Both call env.step.',
              'accuracy':'Within seed, hits divided by actual executed shots or NA. Primary group accuracy is the mean of defined seed accuracies, plus a separately reported pooled ratio. All three seeds must be defined for a learned method to pass the accuracy gate.',
              'threshold_reference':'Compare each weight to reference with the same variant. Also report unfiltered-to-filtered within each model.',
              'stress':'Use prior turn and speed toggle probabilities of 0.10.',
              'holdout':SPLITS,'gate_scenarios':list(range(4200000,4200010)),
              'native_audit_extra_stress_scenarios':list(range(4300000,4300010)),
              'demo':{'condition':'close_5p0','training_seed':20,'scenario_seed':4400000,
                      'reason':'Fixed prior demonstration route and condition with the highest historical mean miss count.'},
              'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in SOURCE},
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'training_interactions_this_study':0})
    print(f'Historical reconciliation passed for {record["previous_per_seed_rows"]} rows. New ranges unused.',flush=True)


def checked(output):
    protocol=read_json(output/'protocol.json')
    for name,digest in protocol['source_sha256'].items():
        require(sha(Path(__file__).with_name(name))==digest,f'Frozen study source changed: {name}')
    return protocol


def split_config(config, split):
    return {**config, 'environment':{**config['environment'],
            **({k:config['tradeoff']['stress'][k] for k in ('turn_probability','speed_toggle_probability')}
               if split=='stress' else {})}}


def check_native(config, kernel, agent, seed, initial, expected):
    env=TrackingEnv(config)
    start,_=env.reset(seed)
    got=kernel.run(env,start,300,agent)
    require(start==initial and np.array_equal(got,expected),f'Native/Python rollout differs: {seed}')


def gate(output, library, rebuilt_kernel=False):
    protocol=checked(output)
    config=protocol['config']
    expected=protocol['history']['kernel_sha256']
    actual=sha(library)
    require(actual==expected or rebuilt_kernel, 'Historical native DLL hash differs. For an independent build use --rebuilt-kernel.')
    source='aimrl/tracking_tradeoff/kernel.c'
    require(sha(ROOT/source)==read_json(PRIOR/'protocol.json')['source_sha256'][source],
            'Historical native kernel source hash differs')
    kernel=Kernel(library)
    result, traces, qevents=[],[],[]
    native_ticks=filtered=0
    seen={}
    for m in protocol['models']:
        c,s=m['condition'],m['training_seed']
        agent=load_frozen(config,c,s,m['sha256'])
        before=agent.snapshot()
        historical={int(w['scenario_seed']):w for w in rows(PRIOR/'evaluation/final'/f'{c}_seed{s}_windows.csv')}
        for scenario in protocol['gate_scenarios']:
            init, raw, _, a,b,events,trace = replay_pair(config,scenario,agent.greedy,agent,strict_rng=True,keep_trace=True)
            check_native(config,kernel,agent,scenario,init,raw)
            native_ticks+=300
            old=historical[scenario]
            for key,val in a.items():
                if key=='accuracy':
                    continue
                require(val==old[key] if isinstance(val,str) else math.isclose(val,float(old[key]),abs_tol=1e-9),
                        f'Python replay disagrees with historical record: {c}/{s}/{scenario}/{key}')
            require(seen.setdefault(scenario,a['target_sha256'])==a['target_sha256'],'Gate model target pairing differs')
            label={'condition':c,'training_seed':s}
            result.extend({**label,'variant':variant,'scenario_seed':scenario,**r}
                          for variant,r in (('original',a),('guard',b)))
            traces.extend({**label,**r} for r in trace)
            qevents.extend({**label,**r} for r in events)
            require(len(events)==a['misses'],'Gate Q diagnostics do not cover every removed miss')
            filtered+=len(events)
        stress=split_config(config,'stress')
        for scenario in protocol['native_audit_extra_stress_scenarios']:
            init,raw,*_=replay_pair(stress,scenario,agent.greedy)
            check_native(stress,kernel,agent,scenario,init,raw)
            native_ticks+=300
        require(before==agent.snapshot(),'Gate changed frozen learner state')
    dest=output/'gate'
    save_csv(dest/'windows.csv',result)
    save_csv(dest/'paired_ticks.csv',traces)
    save_csv(dest/'filtered_q_events.csv',qevents)
    record={'complete':True,'models':21,'scenarios_per_model':10,'paired_windows':len(result)//2,
            'paired_ticks':len(traces),'target_rng_state_compared_every_gate_tick':True,
            'native_python_read_only_ticks_matched':native_ticks,'removed_misses':filtered,
            'native_kernel_sha256':actual,'historical_kernel_sha256':expected,
            'native_kernel_matches_historical_binary':actual==expected,'rebuilt_kernel_allowed':rebuilt_kernel,
            'training_interactions':0,'learner_snapshots_unchanged':True}
    save_json(dest/'checks.json',record)
    print(f'Gate passed: {len(traces):,} paired ticks, {native_ticks:,} native/Python ticks checked.',flush=True)


def identity(m):
    condition,seed=m['condition'],m['training_seed']
    return condition if seed is None else f'{condition}_seed{seed}'


def evaluate_instance(output,config,splits,m):
    c,s=m['condition'],m['training_seed']
    agent=load_frozen(config,c,s,m['sha256']) if s is not None else None
    policy=agent.greedy if agent is not None else lambda state:rule_policy(state,25)
    snapshot=agent.snapshot() if agent is not None else None
    instance=identity(m)
    records=[]
    for split,(start,n) in splits.items():
        env_config=split_config(config,split)
        window_rows={'original':[],'guard':[]}
        diagnostics=[]
        for scenario in range(start,start+n):
            _,_,_,a,b,events,_=replay_pair(env_config,scenario,policy,agent)
            require(len(events)==a['misses'],'Not all removed misses were logged')
            diagnostics.extend(events)
            window_rows['original'].append({'scenario_seed':scenario,**a})
            window_rows['guard'].append({'scenario_seed':scenario,**b})
        if agent is None:
            require(window_rows['original']==window_rows['guard'],'Rule was changed by the guard')
        for variant in ('original','guard') if agent is not None else ('original',):
            r=window_rows[variant]
            for item in r:
                if item['accuracy'] is None:
                    item['accuracy']='NA'
            save_csv(output/'evaluation'/split/f'{instance}_{variant}_windows.csv',r)
            records.append({'split':split,'condition':c,'training_seed':s,'variant':variant,**values(r)})
        if diagnostics:
            save_csv(output/'q_events'/split/f'{instance}.csv',
                     [{'condition':c,'training_seed':s,'split':split,**e} for e in diagnostics])
    require(agent is None or agent.snapshot()==snapshot,'Evaluation changed frozen learner state')
    return records


def evaluate(output,workers):
    protocol=checked(output)
    require(read_json(output/'gate/checks.json')['complete'],'Pilot pairing gate incomplete')
    require(not (output/'evaluation').exists(),'Choose a fresh evaluation destination')
    start_time=time.perf_counter()
    instances=[*protocol['models'],{'condition':'rule','training_seed':None}]
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate_instance,output,protocol['config'],protocol['holdout'],m) for m in instances]
        for m,future in zip(instances,futures):
            records.extend(future.result())
            print(f'{identity(m)}: final and stress completed',flush=True)
    sums={}
    for split,(start,n) in protocol['holdout'].items():
        subset=[r for r in records if r['split']==split]
        target={}
        for r in subset:
            name=identity(r)
            raw=rows(output/'evaluation'/split/f'{name}_{r["variant"]}_windows.csv')
            require([int(w['scenario_seed']) for w in raw]==list(range(start,start+n)),'Missing or duplicate held-out scenario')
            for w in raw:
                require(target.setdefault(w['scenario_seed'],w['target_sha256'])==w['target_sha256'],
                        'Held-out target routes not paired')
        require(len(subset)==43,'Incomplete held-out variant matrix')
        save_json(output/'evaluation'/split/'paired_targets.json',target)
        sums[split]={'policies':len(subset),'scenarios':n,'windows':sum(r['windows'] for r in subset)}
    save_json(output/'evaluation/per_seed.json',records)
    save_csv(output/'per_seed.csv',[{k:('NA' if v is None else v) for k,v in r.items()} for r in records])
    verify_history(read_json(output/'prior_sha256_before.json'))
    save_json(output/'evaluation/checks.json',{'complete':True,'splits':sums,'elapsed_seconds':time.perf_counter()-start_time,
              'workers':workers,'paired_physical_states_checked_each_tick':True,'learner_snapshots_unchanged':True,
              'training_interactions':0})
