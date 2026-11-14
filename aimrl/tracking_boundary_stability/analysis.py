"""Frozen-model evaluation, paired factorial contrasts and descriptive dynamics."""
import copy
import json
from collections import defaultdict

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking_fire_guard.replay import FiringGuard
from ..tracking_research.io import read_json, save_csv, save_json, sha
from ..tracking_tradeoff.core import summarize, window_metrics
from .core import CONDITIONS, OUT, load_model, require

METRICS=('D','C','H','A','first_close_tick','longest_far_streak','alignment_opportunity_fraction','fire_given_alignment','misses')
CONTRASTS={
    'boundary_at_fixed':{'B-Fixed':1,'R-Fixed':-1},
    'boundary_at_decay':{'B-Decay':1,'R-Decay':-1},
    'decay_in_relative':{'R-Decay':1,'R-Fixed':-1},
    'decay_in_boundary':{'B-Decay':1,'B-Fixed':-1},
    'interaction':{'B-Decay':1,'B-Fixed':-1,'R-Decay':-1,'R-Fixed':1},
}


def python_check(c,agent,seed,raw,guarded):
    e0,e1=TrackingEnv(c),TrackingEnv(c)
    s0,_=e0.reset(seed);s1,_=e1.reset(seed)
    guard=FiringGuard(agent.greedy)
    for i in range(300):
        a0=agent.greedy(s0);a1=guard(s1)
        s0,r0,_,_,j0=e0.step(a0);s1,r1,_,_,j1=e1.step(a1)
        require((*s0,a0)==tuple(raw[i,:7]) and (*s1,a1)==tuple(guarded[i,:7]),'Independent Python evaluation mismatch')
        require(s0==s1 and j0['distance']==j1['distance'] and j0['hit']==j1['hit'],'Guard physical invariance failed in Python')
        require(abs(r1-r0-int(a0!=a1))<1e-12,'Guard unit base reward improvement failed')
        require(e0.rng.bit_generator.state==e1.rng.bit_generator.state,'Guard target RNG mismatch')


def dynamic_slices(c,initial,ticks):
    spec=c['study']['event_window'];pre=spec['before'];post=spec['after'];horizon=spec['recovery_horizon']
    consecutive=spec['recovery_consecutive_close'];d=ticks[:,9];curves=[];recoveries=[]
    random_events=(ticks[:,14]>0)|(ticks[:,15]>0)
    for kind,col in (('bounce',13),('turn',14),('toggle',15)):
        last=-horizon-1
        for t in np.flatnonzero(ticks[:,col]):
            if t-last<=horizon:continue
            last=int(t)
            for offset in range(-pre,post+1):
                if 0<=t+offset<300:curves.append((kind,offset,int(d[t+offset])))
            if kind=='bounce':continue
            following=np.flatnonzero(random_events[t+1:])
            stop=min(300,t+horizon+consecutive,int(t+1+following[0]) if len(following) else 300)
            recovered=None
            for start in range(int(t),min(int(t)+horizon+1,stop-consecutive+1)):
                if np.all(d[start:start+consecutive]<=1):recovered=start-int(t);break
            recoveries.append(dict(kind=kind,tick=int(t)+1,recovered=recovered is not None,
                                   recovery_ticks=recovered,censor_ticks=int(stop-t),
                                   reason='recovered' if recovered is not None else 'next_random_event_or_horizon_or_window_end'))
    return curves,recoveries


def evaluate_all(c,kernel):
    from .study import history_check
    destination=OUT/'evaluation';destination.mkdir(exist_ok=False)
    per_seed=[];checks=[];event_curves=[];recovery_rows=[];slice_rows=[];own_q=[]
    for split in ('final','stress'):
        cc=copy.deepcopy(c);spec=c['study'][split]
        if split=='stress':cc['environment'].update({k:spec[k] for k in ('turn_probability','speed_toggle_probability')})
        seeds=list(range(spec['start'],spec['start']+spec['count']))
        target_hashes={}
        for training_seed in c['study']['training_seeds']:
            for condition in CONDITIONS:
                agent,_=load_model(OUT/'runs'/condition/f'seed_{training_seed}/checkpoints/step_5000000.npz',c)
                before=agent.snapshot();raw_rows=[];guard_rows=[];all_ticks=[];all_raw_actions=[];all_initial=[]
                events=defaultdict(list);recoveries=[];own_states=[];own_actions=[]
                for scenario in seeds:
                    e0,e1=TrackingEnv(cc),TrackingEnv(cc)
                    initial,_=e0.reset(scenario);s1,_=e1.reset(scenario)
                    require(initial==s1,'Reset pairing differs')
                    raw=kernel.run(e0,initial,300,agent)
                    guarded=kernel.run(e1,s1,300,agent,guard=True)
                    require(np.array_equal(raw[:,:6],guarded[:,:6]) and np.array_equal(raw[:,7:12],guarded[:,7:12])
                            and np.array_equal(raw[:,13:],guarded[:,13:]),'Guard changed physical rollout')
                    changed=raw[:,6]!=guarded[:,6]
                    require(np.array_equal(changed,(raw[:,6]==5)&(raw[:,7]>0)) and np.all(guarded[changed,6]==4),
                            'Guard filtered an unexpected action')
                    a=window_metrics(initial,raw,True);b=window_metrics(initial,guarded,True)
                    require(a['distance_sum']==b['distance_sum'] and a['close_ticks']==b['close_ticks'] and a['hits']==b['hits'],
                            'Guard primary metrics differ')
                    require(abs(b['base_return']-a['base_return']-int(changed.sum()))<1e-10,'Guard return increase mismatch')
                    digest=b['target_sha256'];require(target_hashes.setdefault(scenario,digest)==digest,'Cross-model target pairing failed')
                    raw_rows.append(dict(scenario_seed=scenario,**a));guard_rows.append(dict(scenario_seed=scenario,**b))
                    if scenario in (seeds[0],seeds[-1]):python_check(cc,agent,scenario,raw,guarded)
                    all_initial.append(initial);all_ticks.append(guarded);all_raw_actions.append(raw[:,6])
                    states=np.vstack([np.array(initial),raw[:-1,:6]])
                    own_states.append(states);own_actions.append(raw[:,6])
                    curves,rs=dynamic_slices(c,initial,guarded)
                    for kind,offset,d in curves:events[kind,offset].append(d)
                    recoveries.extend(dict(scenario_seed=scenario,**r) for r in rs)
                require(agent.snapshot()==before,'Evaluation changed frozen model')
                label=dict(split=split,condition=condition,seed=training_seed)
                folder=destination/split;folder.mkdir(exist_ok=True)
                stem=f'{condition}_seed{training_seed}'
                save_csv(folder/f'{stem}_original_windows.csv',raw_rows)
                save_csv(folder/f'{stem}_guard_windows.csv',guard_rows)
                np.savez_compressed(folder/f'{stem}_rollouts.npz',initial=np.array(all_initial,dtype=np.int16),
                                    ticks=np.array(all_ticks,dtype=np.int16),raw_actions=np.array(all_raw_actions,dtype=np.int16),
                                    scenario_seeds=np.array(seeds))
                for variant,rows in (('original',raw_rows),('guard',guard_rows)):
                    summary=summarize(rows)
                    per_seed.append(dict(**label,variant=variant,invalid_fire_tick_fraction=summary['total_unaligned_shots']/summary['ticks'],
                                         invalid_fraction_of_shots=summary['total_unaligned_shots']/summary['total_shots'] if summary['total_shots'] else None,
                                         **summary))
                    for group in ('boundary','interior','speed_1','speed_2'):
                        sums=np.array([json.loads(r['slice_counts'])[group] for r in rows],dtype=np.int64).sum(axis=0)
                        nt,ds,cl,hi,sh,mi=map(int,sums)
                        slice_rows.append(dict(**label,variant=variant,group=group,ticks=nt,D=ds/nt if nt else None,
                                               C=cl/nt if nt else None,hits=hi,shots=sh,A=hi/sh if sh else None,
                                               hits_per_300_slice_ticks=hi/nt*300 if nt else None))
                for (kind,offset),values in events.items():
                    event_curves.append(dict(**label,kind=kind,offset=offset,events=len(values),distance_mean=float(np.mean(values))))
                save_csv(folder/f'{stem}_recovery_events.csv',recoveries)
                for kind in ('turn','toggle'):
                    rs=[r for r in recoveries if r['kind']==kind];times=[r['recovery_ticks'] for r in rs if r['recovered']]
                    recovery_rows.append(dict(**label,kind=kind,events=len(rs),recovered=len(times),
                                  recovered_fraction=len(times)/len(rs) if rs else None,
                                  mean_recovery_ticks_among_recovered=float(np.mean(times)) if times else None,
                                  censored=len(rs)-len(times),interpretation='Descriptive; conditioned on recovery before censoring'))
                s=np.vstack(own_states);actions=np.hstack(own_actions)
                unique,inv,freq=np.unique(s,axis=0,return_inverse=True,return_counts=True)
                ids=agent.encoder.indices(unique);q=agent.q[ids]
                unaligned=(unique[:,0]!=unique[:,2])|(unique[:,1]!=unique[:,3])
                for weighting,weights in (('state',np.ones(len(unique))),('event',freq)):
                    m=unaligned;frequent=m&(freq>=c['study']['frequent_state_min_events'])
                    own_q.append(dict(**label,weighting=weighting,distinct_states=int(m.sum()),events=int(freq[m].sum()),
                         wrong_ranking_fraction=float(np.average((q[:,5]>q[:,4])[m],weights=weights[m])),
                         greedy_unaligned_fire_fraction=float(np.average((q.argmax(axis=1)==5)[m],weights=weights[m])),
                         frequent_wrong_ranking_fraction=float(np.average((q[:,5]>q[:,4])[frequent],weights=weights[frequent])) if frequent.any() else None,
                         frequent_states=int(frequent.sum()),sampling='Each model own pre-action trajectory; descriptive, not shared-state causal evidence'))
                checks.append(dict(**label,scenarios=len(seeds),guard_trajectory_exact=True,guard_reward_unit=True,
                                   python_scenarios_checked=2,frozen_model_unchanged=True))
                print(f'Evaluated {split} {condition} seed {training_seed}',flush=True)
        save_json(destination/split/'paired_targets.json',target_hashes)
    save_csv(OUT/'per_seed.csv',per_seed);save_json(OUT/'per_seed.json',per_seed)
    save_csv(OUT/'boundary_speed_slices.csv',slice_rows)
    save_csv(OUT/'event_curves.csv',event_curves);save_csv(OUT/'recovery.csv',recovery_rows)
    save_csv(OUT/'q_own_trajectory.csv',own_q)
    save_json(destination/'checks.json',dict(complete=True,checks=checks,history=history_check(),
              independent_python_windows=80,formal_models=20,paired_target_scenarios=700,
              raw_and_guard_windows=28000,evaluation_ticks=8400000))


def aggregate(per_seed):
    groups=defaultdict(list)
    for r in per_seed:groups[r['split'],r['condition'],r['variant']].append(r)
    comparison=[]
    metrics=METRICS+('invalid_fire_tick_fraction','invalid_fraction_of_shots')
    for (split,condition,variant),rows in groups.items():
        row=dict(split=split,condition=condition,variant=variant,n_training_seeds=len(rows))
        for k in metrics:
            vals=[r[k] for r in rows if r[k] is not None]
            higher=k in ('C','H','A','fire_given_alignment')
            row.update({k:float(np.mean(vals)) if vals else None,k+'_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                        k+'_worst':float(min(vals) if higher else max(vals)) if vals else None,
                        k+'_defined_seeds':len(vals)})
        shots=sum(r['total_shots'] for r in rows);hits=sum(r['total_hits'] for r in rows)
        row['A_pooled']=hits/shots if shots else None
        comparison.append(row)
    return comparison


def contrasts(per_seed):
    paired=[];factors=[]
    for split in ('final','stress'):
        for variant in ('original','guard'):
            rows={ (r['condition'],r['seed']):r for r in per_seed if r['split']==split and r['variant']==variant}
            seeds=sorted({s for _,s in rows})
            for seed in seeds:
                base=rows['R-Fixed',seed]
                for condition in CONDITIONS[1:]:
                    r=rows[condition,seed]
                    paired.append(dict(split=split,variant=variant,seed=seed,condition=condition,reference='R-Fixed',
                       **{k:r[k]-base[k] if r[k] is not None and base[k] is not None else None for k in METRICS}))
            for name,weights in CONTRASTS.items():
                for k in METRICS:
                    values=[sum(weights[cond]*rows[cond,s][k] for cond in weights) for s in seeds
                            if all(rows[cond,s][k] is not None for cond in weights)]
                    factors.append(dict(split=split,variant=variant,effect=name,metric=k,n_pairs=len(values),
                                  mean=float(np.mean(values)) if values else None,
                                  sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                                  paired_values=json.dumps(values),units='fraction' if k in ('C','A') else 'native metric units'))
    return paired,factors


def thresholds(c,per_seed,comparison):
    result=[]
    for split in ('final','stress'):
        for level in ('mean','seed'):
            rows=[r for r in (comparison if level=='mean' else per_seed) if r['split']==split and r['variant']=='guard']
            for r in rows:
                seed=r.get('seed');base=next(x for x in rows if x['condition']=='R-Fixed' and x.get('seed')==seed)
                d=r['D']<=base['D'];close=r['C']>=base['C']+.05
                acc=r['A'] is not None and r['A']>=.98 and (level=='seed' or r['A_defined_seeds']==5)
                retain=r['H']>=.98*base['H'];gain=r['H']>=1.02*base['H']
                result.append(dict(split=split,level=level,seed=seed,condition=r['condition'],
                              D_nonincrease=d,C_plus_5pp=close,H_retains_98pct=retain,H_plus_2pct=gain,A_at_least_98pct=acc,
                              practical_pass=d and close and retain and acc,simultaneous_pass=d and close and gain and acc,
                              absolute_C75_pass=r['C']>=.75,D_delta=r['D']-base['D'],C_delta=r['C']-base['C'],
                              H_ratio=r['H']/base['H'] if base['H'] else None))
    return result
