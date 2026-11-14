"""Exhaustive DP/cache and saved-record audits; no training or formal replay."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..tracking_research.agent import array_hash
from ..tracking_research.io import ROOT,read_csv,read_json,save_csv,save_json,sha
from .analysis import aggregate,paired_comparisons,thresholds,scenario_differences
from .core import OUT,Native,Policy,frozen_agent,methods,transition_arrays
from .evaluation import canonical_metrics,summary,slices
from .reference import require


def excluded(path):
    name=path.relative_to(OUT).as_posix()
    return (name in ('output_inventory.csv','inventory_scope.json','verification_python.json','verification_optimized.json','postflight.json')
            or name.startswith('execution/postflight') or name.startswith('checks/postflight') or name.startswith('validation_optimized/'))


def inventory():
    from .study import sources
    save_json(OUT/'source_sha256_delivery.json',sources())
    entries=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(OUT.rglob('*')) if p.is_file() and not excluded(p)]
    save_csv(OUT/'output_inventory.csv',entries)
    save_json(OUT/'inventory_scope.json',dict(files=len(entries),bytes=sum(r['bytes'] for r in entries),
              excludes=['output_inventory.csv','inventory_scope.json','verification_python.json','verification_optimized.json','postflight.json',
                        'execution/postflight*','checks/postflight*','validation_optimized/*'],
              reason='Mutable verification receipts cannot hash themselves. Protocol, input models, mathematical cache arrays and all protected historical files remain separately checked.'))
    return entries


def equal(a,b,label):
    require(len(a)==len(b),'Row count differs: '+label)
    for ra,rb in zip(a,b):
        require(set(ra)==set(rb),'Columns differ: '+label)
        for k,v in ra.items():
            x=rb[k]
            if isinstance(v,float):require(x is not None and math.isclose(v,x,rel_tol=1e-12,abs_tol=1e-12),'Number differs: '+label+'/'+k)
            else:require(v==x,'Field differs: '+label+'/'+k)


def csv_equal(path,rows):
    saved=read_csv(path);require(len(saved)==len(rows),'CSV row count differs: '+str(path))
    for a,b in zip(rows,saved):
        require(set(a)==set(b),'CSV fields differ')
        for key,value in a.items():
            if isinstance(value,float):require(math.isclose(value,float(b[key]),rel_tol=1e-12,abs_tol=1e-12),'CSV number differs: '+key)
            else:require(b[key]==('' if value is None else str(value)),'CSV value differs: '+key)


def verify(c):
    from .study import verify_protocol,protect
    protocol=verify_protocol(c);integrity=protect()
    require(read_json(OUT/'evaluation_complete.json')['new_training_interactions']==0,'Unexpected training')
    native=Native(c);arrays=transition_arrays(c);dp_rows=0;transitions=0
    pre=read_json(OUT/'precomputation.json');records=pre['records']
    for terminal,seed in [('Phi',None),('Zero',None)]+[('Q+Phi',s) for s in c['planning']['training_seeds']]:
        agent=frozen_agent('R-Decay',seed) if seed is not None else None
        value=native.initialize(terminal,agent)
        for h in (1,2,3):
            r=next(r for r in records if (r['terminal'],r['seed'],r['horizon'])==(terminal,seed,h))
            expected_value,expected_policy,expected_chance,stats=native.advance(value,arrays)
            with np.load(OUT/r['bundle'],allow_pickle=False) as z:
                values={k:z[k] for k in ('value','policy','expected')}
                meta=json.loads(str(z['metadata']))
            for key,actual in values.items():
                require(array_hash(actual)==meta[key+'_sha256'],'Saved DP array digest mismatch')
            require(np.array_equal(values['value'],expected_value) and np.array_equal(values['policy'],expected_policy)
                    and np.array_equal(values['expected'],expected_chance),'Exhaustive Bellman cache reconstruction differs')
            require(list(map(int,stats))==[r[k] for k in ('value_backups','chance_transition_terms','action_value_lookups','shared_expectation_cache_hits')], 'Actual DP counters differ')
            policy=values['policy'].reshape(625,5000)
            invalid=(policy==5)&(np.arange(625)[:,None]!=np.arange(5000)[None,:]//8)
            require(not invalid.any(),'Dominated Fire in a precomputed policy')
            value=values['value'];dp_rows+=len(value);transitions+=int(stats[1])
            del expected_value,expected_policy,expected_chance,values,policy,invalid
    require(dp_rows==pre['value_backups'] and transitions==pre['chance_transition_terms'],'Planning budget counters differ')
    all_rows=[];windows=0;environment_ticks=0;planning_lookups=0
    for split in ('development','final','stress'):
        folder=OUT/'evaluation'/split;spec=c['planning'][split];seen={};recomputed=[]
        for method in methods(c):
            path=folder/f'{method["id"]}_rollouts.npz'
            with np.load(path,allow_pickle=False) as z:
                initial,ticks,ns,seeds=(z[k] for k in ('initial','ticks','decision_ns','scenario_seeds'))
            require(ticks.shape==(spec['count'],300,18) and initial.shape==(spec['count'],6) and ns.shape==(spec['count'],300),'Saved rollout dimensions differ')
            require(list(map(int,seeds))==list(range(spec['start'],spec['start']+spec['count'])),'Scenario schedule differs')
            require((ns>=0).all(),'Negative decision latency')
            rows=[];policy=Policy(c,method)
            before=policy.agent.snapshot() if policy.agent is not None else None
            for i,seed in enumerate(seeds):
                row=dict(scenario_seed=int(seed),**canonical_metrics(initial[i],ticks[i]));rows.append(row)
                require(seen.setdefault(int(seed),(initial[i].tolist(),row['target_sha256']))==(initial[i].tolist(),row['target_sha256']),'Saved target routes are not paired')
                require(np.array_equal(ticks[i,:,17],np.r_[initial[i,5],ticks[i,:-1,5]]),'Pre-action speed slice corrupted')
                proposed=ticks[i,:,16];executed=ticks[i,:,6];distance=ticks[i,:,7]
                require(np.array_equal(executed,np.where((proposed==5)&(distance>0),4,proposed)),'Guard execution differs')
                if method['kind']=='plan':require(not np.any((proposed==5)&(distance>0)),'Planner proposed invalid Fire')
                # Recompute policy proposals on all saved states, vectorized for
                # immutable policy/Q lookups. This never replays an environment.
                states=np.vstack([initial[i],ticks[i,:-1,:6]])
                if method['kind']=='plan':
                    ids=((states[:,0]*25+states[:,1]).astype(np.int64)*625+states[:,2]*25+states[:,3])*8+states[:,4]*2+states[:,5]-1
                    observed=policy.table[ids]
                elif method['kind']=='learned':observed=policy.agent.q[policy.agent.encoder.indices(states)].argmax(axis=1)
                else:observed=np.array([policy.propose(tuple(map(int,s))) for s in states])
                require(np.array_equal(proposed,observed),'Saved proposals differ from frozen policy')
            require(before is None or policy.agent.snapshot()==before,'Validation changed a frozen learner')
            record=dict(split=split,policy=method['id'],family=method['family'],seed=method['seed'],**summary(rows,ticks));recomputed.append(record)
            csv_equal(folder/f'{method["id"]}_windows.csv',rows)
            windows+=spec['count'];environment_ticks+=ticks.shape[0]*300
            if method['kind']=='plan':planning_lookups+=ticks.shape[0]*300
        equal(recomputed,read_json(folder/'per_policy.json'),split+' per_policy')
        if split!='development':all_rows+=recomputed
        complete=read_json(folder/'complete.json')
        require(complete['fixed_planning_probabilities']==[.05,.05],'Stress changed planning model')
        expected_prob=.10 if split=='stress' else .05
        require(complete['real_environment']['turn_probability']==expected_prob and complete['real_environment']['speed_toggle_probability']==expected_prob,'Real environment probabilities differ')
    require(windows==18000 and environment_ticks==5400000 and planning_lookups==2430000,'Window / environment / planning counts differ')
    equal(all_rows,read_json(OUT/'per_policy.json'),'all per_policy')
    comparison=aggregate(all_rows);paired,contrasts=paired_comparisons(all_rows,comparison);gates=thresholds(c,all_rows,comparison)
    equal(comparison,read_json(OUT/'comparison.json'),'comparison');equal(gates,read_json(OUT/'thresholds.json'),'thresholds')
    for name,rows in (('comparison.csv',comparison),('paired_differences.csv',paired),('factor_contrasts.csv',contrasts),('thresholds.csv',gates),
                      ('scenario_paired_differences.csv',scenario_differences(c))):csv_equal(OUT/name,rows)
    frames=0
    for receipt in read_json(OUT/'demos/decode_checks.json'):
        folder=OUT/'demos'/f'scenario_{receipt["scenario_seed"]}';path=folder/'synchronized.gif'
        require(sha(path)==receipt['sha256'],'GIF file hash differs')
        with np.load(folder/'rollouts.npz',allow_pickle=False) as z:initial,ticks=z['initial'],z['ticks'];ids=z['policy_ids']
        for i,key in enumerate(ids):
            metric=canonical_metrics(initial[i],ticks[i]);expected=read_json(folder/'metrics.json')[i]
            recomputed=summary([metric],ticks[i:i+1])
            equal([dict(policy=str(key),**recomputed)],[expected],'demo metrics')
        duration=0
        with Image.open(path) as im:
            require(im.n_frames==301,'GIF does not contain all 301 frames')
            for i in range(301):
                im.seek(i);im.load();duration+=im.info['duration'];frames+=1
                if str(i) in receipt['samples']:require(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==receipt['samples'][str(i)],'GIF sample pixel hash differs')
        require(duration==receipt['duration_ms']==20070 and receipt['decoded_all_frame_pixels_and_timing'],'GIF decode/timing receipt differs')
    inventory_count=0
    if (OUT/'output_inventory.csv').exists():
        entries=read_csv(OUT/'output_inventory.csv')
        for r in entries:
            path=OUT/r['path'];require(path.is_file() and path.stat().st_size==int(r['bytes']) and sha(path)==r['sha256'],'Delivery inventory differs: '+r['path'])
        require({r['path'] for r in entries}=={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file() and not excluded(p)},'Delivery file set differs')
        inventory_count=len(entries)
    result=dict(complete=True,optimized=not __debug__,new_training_interactions=0,Q_updates=0,
            formal_windows=14000,formal_environment_ticks=4200000,development_windows=4000,development_environment_ticks=1200000,
            checked_policy_proposals=environment_ticks,formal_planning_lookups=1890000,verified_value_backups=dp_rows,
            verified_chance_terms=transitions,verification_planning_recomputations=dp_rows,formal_environment_replays=0,
            decoded_demo_frames=frames,inventory_files=inventory_count,protection=integrity,actual_player_check='Not performed, as requested.')
    save_json(OUT/('verification_optimized.json' if not __debug__ else 'verification_python.json'),result)
    return result
