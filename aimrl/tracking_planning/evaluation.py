"""Paired inference-only evaluation and prespecified descriptive diagnostics."""
from collections import defaultdict
import copy
import json
import time

import numpy as np

from ..tracking25.env import TrackingEnv
from ..tracking_research.io import ROOT,read_json,save_csv,save_json,sha
from ..tracking_tradeoff.core import summarize,window_metrics
from ..tracking_boundary_stability.analysis import dynamic_slices
from .core import OUT,Native,Policy,methods
from .reference import require


def canonical_metrics(initial,ticks):
    return window_metrics(tuple(map(int,initial)),ticks[:,:16],True)


def summary(rows,ticks):
    m=summarize(rows);flat=ticks.reshape(-1,18);a=flat[:,6];pre=flat[:,7]
    counts=np.bincount(a,minlength=10)
    alignment=pre==0;movement=np.isin(a,(0,1,2,3,6,7,8,9))
    first=[]
    for window in ticks:
        hit=np.flatnonzero(window[:,7]==0);first.append(int(hit[0])+1 if hit.size else 300)
    return dict(**m,first_alignment_tick=float(np.mean(first)),invalid_proposed_fire=int(np.count_nonzero((flat[:,16]==5)&(pre>0))),
                invalid_proposed_fire_fraction=float(np.mean((flat[:,16]==5)&(pre>0))),
                filtered_proposals=int(np.count_nonzero(flat[:,16]!=a)),aligned_move_fraction=float(movement[alignment].mean()) if alignment.any() else None,
                **{f'action_{i}_fraction':float(counts[i]/len(a)) for i in range(10)})


def slices(initial,ticks):
    pre=np.vstack([np.asarray(initial),ticks[:-1,:6]])
    boundary=np.any(np.minimum(pre[:,:4],24-pre[:,:4])<=3,axis=1)
    groups=dict(boundary=boundary,interior=~boundary,pre_speed_1=pre[:,5]==1,pre_speed_2=pre[:,5]==2,
                post_speed_1=ticks[:,5]==1,post_speed_2=ticks[:,5]==2)
    result={}
    for group,mask in groups.items():
        distance=ticks[mask,9];shot=ticks[mask,6]==5;hits=ticks[mask,11]
        result[group]=(int(mask.sum()),int(distance.sum()),int(np.count_nonzero(distance<=1)),int(hits.sum()),int(shot.sum()),
                       int(np.count_nonzero(ticks[mask,7]==0)),int(np.count_nonzero((ticks[mask,7]==0)&shot)),
                       int(np.count_nonzero((ticks[mask,7]==0)&np.isin(ticks[mask,6],(0,1,2,3,6,7,8,9)))))
    return result


def latency_record(times):
    return dict(decisions=int(times.size),median_ns=float(np.median(times)),p95_ns=float(np.quantile(times,.95)),maximum_ns=int(times.max()),
                mean_ns=float(times.mean()),scope='Native proposal plus firing guard only; includes timer overhead. Precomputation and Python/load/window overhead reported separately.')


def evaluate_split(c,split):
    spec=c['planning'][split];cc=copy.deepcopy(c)
    if split=='stress':cc['environment'].update(turn_probability=spec['turn_probability'],speed_toggle_probability=spec['speed_toggle_probability'])
    folder=OUT/'evaluation'/split;folder.mkdir(parents=True,exist_ok=False)
    seeds=list(range(spec['start'],spec['start']+spec['count']))
    native=Native(c);paired={};summaries=[];costs=[];all_slices=[];event_curves=[];recoveries=[]
    started=time.perf_counter()
    for method in methods(c):
        policy=Policy(c,method);before=policy.agent.snapshot() if policy.agent is not None else None
        initial_all=[];ticks_all=[];times_all=[];rows=[];counts=defaultdict(lambda:np.zeros(8,dtype=np.int64));curves=defaultdict(list);recovery=[]
        run_start=time.perf_counter()
        for seed in seeds:
            env=TrackingEnv(cc);initial,_=env.reset(seed)
            ticks,times=native.run(env,initial,policy)
            row=dict(scenario_seed=seed,**canonical_metrics(initial,ticks))
            require(paired.setdefault(seed,(tuple(initial),row['target_sha256']))==(tuple(initial),row['target_sha256']),'Initial state / target route pairing failed')
            if method['kind']=='plan':require(not np.any((ticks[:,16]==5)&(ticks[:,7]>0)),'Planner proposed dominated unaligned Fire')
            require(not np.any((ticks[:,6]==5)&(ticks[:,7]>0)),'Executed invalid Fire after guard')
            initial_all.append(initial);ticks_all.append(ticks);times_all.append(times);rows.append(row)
            for group,values in slices(initial,ticks).items():counts[group]+=values
            ev,rs=dynamic_slices({'study':{'event_window':c['planning']['event_window']}},initial,ticks)
            for kind,offset,d in ev:curves[kind,offset].append(d)
            recovery.extend(dict(scenario_seed=seed,**r) for r in rs)
        require(before is None or policy.agent.snapshot()==before,'Frozen Q/counts changed during evaluation')
        ticks=np.array(ticks_all,dtype=np.int16);timing=np.array(times_all,dtype=np.int64)
        label=dict(split=split,policy=method['id'],family=method['family'],seed=method['seed'])
        record=dict(**label,**summary(rows,ticks));summaries.append(record)
        save_csv(folder/f'{method["id"]}_windows.csv',rows)
        np.savez_compressed(folder/f'{method["id"]}_rollouts.npz',initial=np.array(initial_all,np.int16),ticks=ticks,
                            decision_ns=timing,scenario_seeds=np.array(seeds,np.int64))
        cost=dict(**label,load_seconds=policy.load_seconds,evaluation_and_storage_seconds=time.perf_counter()-run_start,
                  **latency_record(timing),online_planning_lookups=timing.size if method['kind']=='plan' else 0,
                  online_model_expansions=0,online_value_backups=0,new_training_interactions=0,Q_updates=0)
        costs.append(cost)
        for group,values in counts.items():
            nt,ds,cl,hi,sh,opp,aligned_shots,aligned_move=map(int,values)
            all_slices.append(dict(**label,group=group,ticks=nt,distance_sum=ds,close_ticks=cl,hits=hi,shots=sh,
                    alignment_opportunities=opp,aligned_shots=aligned_shots,aligned_moves=aligned_move,
                    D=ds/nt if nt else None,C=cl/nt if nt else None,A=hi/sh if sh else None,
                    hits_per_300_slice_ticks=300*hi/nt if nt else None,
                    fire_given_alignment=aligned_shots/opp if opp else None))
        for (kind,offset),values in curves.items():event_curves.append(dict(**label,kind=kind,offset=offset,events=len(values),D=float(np.mean(values))))
        save_csv(folder/f'{method["id"]}_recovery_events.csv',recovery)
        for kind in ('turn','toggle'):
            rs=[r for r in recovery if r['kind']==kind];recovered=[r['recovery_ticks'] for r in rs if r['recovered']]
            recoveries.append(dict(**label,kind=kind,events=len(rs),recovered=len(recovered),censored=len(rs)-len(recovered),
                                  recovered_fraction=len(recovered)/len(rs) if rs else None,
                                  conditional_recovery_mean=float(np.mean(recovered)) if recovered else None))
        print(f'{split}: {method["id"]}, {len(rows)} windows, D={record["D"]:.4f}, C={record["C"]:.4f}, H={record["H"]:.3f}',flush=True)
    save_json(folder/'per_policy.json',summaries);save_csv(folder/'per_policy.csv',summaries)
    save_csv(folder/'runtime.csv',costs);save_json(folder/'runtime.json',costs)
    save_csv(folder/'slices.csv',all_slices);save_csv(folder/'event_curves.csv',event_curves);save_csv(folder/'recovery.csv',recoveries)
    save_json(folder/'paired_targets.json',{str(seed):dict(initial=list(value[0]),target_sha256=value[1]) for seed,value in paired.items()})
    save_json(folder/'complete.json',dict(complete=True,policy_instances=20,windows=20*len(seeds),environment_ticks=6000*len(seeds),
              actual_planning_decisions=9*len(seeds)*300,new_training_interactions=0,Q_updates=0,elapsed_seconds=time.perf_counter()-started,
              real_environment=c['environment'] if split!='stress' else cc['environment'],fixed_planning_probabilities=[.05,.05]))
    return summaries


def common_diagnostics(c):
    path=ROOT/c['planning']['common_diagnostics']['source']
    with np.load(path,allow_pickle=False) as z:states,frequency=z['states'],z['frequency']
    folder=OUT/'diagnostics';folder.mkdir(exist_ok=True)
    all_actions={};rows=[];latencies=[]
    for method in methods(c):
        policy=Policy(c,method);actions=[];ns=[];first=None
        for s in states:
            state=tuple(map(int,s));start=time.perf_counter_ns();a=policy.act(state);elapsed=time.perf_counter_ns()-start
            if first is None:first=elapsed
            actions.append(a);ns.append(elapsed)
        # A second pass uses precisely the same states and immutable precomputed cache.
        warm=[]
        for s in states:
            start=time.perf_counter_ns();policy.act(tuple(map(int,s)));warm.append(time.perf_counter_ns()-start)
        all_actions[method['id']]=np.array(actions,np.int16)
        latencies.append(dict(policy=method['id'],family=method['family'],seed=method['seed'],load_seconds=policy.load_seconds,
                       first_api_call_ns=first,first_pass_median_ns=float(np.median(ns)),first_pass_p95_ns=float(np.quantile(ns,.95)),first_pass_max_ns=max(ns),
                       warm_median_ns=float(np.median(warm)),warm_p95_ns=float(np.quantile(warm,.95)),warm_max_ns=max(warm),
                       scope='Python validated observation API; first pass is not OS-cache flushed. Full DP startup is separately charged.'))
    for i,s in enumerate(states):
        rows.append(dict(zip(('xc','yc','xt','yt','heading','speed'),map(int,s)),frequency=int(frequency[i]),
                         **{key:int(value[i]) for key,value in all_actions.items()}))
    save_csv(folder/'common_actions.csv',rows);save_csv(folder/'api_latency.csv',latencies)
    comparisons=[]
    for seed in c['planning']['training_seeds']:
        baseline=all_actions[f'R-Decay_s{seed}']
        for key in ('Plan-H1','Plan-H2','Plan-H3','Plan-H3-Zero',f'Hybrid-H3_s{seed}'):
            a=all_actions[key];speed2=states[:,5]==2;aligned=(states[:,0]==states[:,2])&(states[:,1]==states[:,3])
            for group,m in (('all',np.ones(len(states),bool)),('speed_2',speed2),('aligned',aligned)):
                for weighting,w in (('state',np.ones(len(states))),('event',frequency)):
                    comparisons.append(dict(planner=key,baseline=f'R-Decay_s{seed}',seed=seed,group=group,weighting=weighting,
                        distinct_states=int(m.sum()),events=int(frequency[m].sum()),disagreement=float(np.average((a!=baseline)[m],weights=w[m])),
                        aligned_fire_to_move=float(np.average(((baseline==5)&np.isin(a,(0,1,2,3,6,7,8,9)))[m],weights=w[m]))))
    save_csv(folder/'common_action_comparisons.csv',comparisons)
    save_json(folder/'common_state_source.json',dict(path=str(path.relative_to(ROOT)),sha256=sha(path),distinct_states=len(states),sampling_events=int(frequency.sum()),
                 state_weighting='Equal distinct physical states',event_weighting='Repeated visitation events in the previously frozen common set',
                 note='No learned policy trajectory or new held-out future is exposed to any planner.'))


def recovery_alignment(ticks,event):
    end=min(300,event+31)
    indices=np.flatnonzero(ticks[event+1:end,7]==0)
    return int(indices[0]) if indices.size else None


def select_behaviors(c,split):
    folder=OUT/'evaluation'/split;records=[];lookup={s['id']:s for s in methods(c)}
    n=c['planning'][split]['count'];start=c['planning'][split]['start']
    for key in ('Plan-H1','Plan-H2','Plan-H3','Plan-H3-Zero','Hybrid-H3'):
        chosen={};examined=0
        # Sort globally by scenario, then training seed, then tick, independently of performance.
        data={}
        for seed in c['planning']['training_seeds']:
            planner_id=f'Hybrid-H3_s{seed}' if key=='Hybrid-H3' else key
            with np.load(folder/f'R-Decay_s{seed}_rollouts.npz',allow_pickle=False) as z:initial=z['initial'];base=z['ticks']
            with np.load(folder/f'{planner_id}_rollouts.npz',allow_pickle=False) as z:plan=z['ticks']
            data[seed]=(initial,base,plan,Policy(c,lookup[planner_id]))
        for scenario in range(n):
            for seed in c['planning']['training_seeds']:
                initial,base,plan,policy=data[seed];b=base[scenario];p=plan[scenario]
                states=np.vstack([initial[scenario],b[:-1,:6]])
                for tick,s in enumerate(states):
                    state=tuple(map(int,s));a=policy.act(state);ra=int(b[tick,6]);examined+=1
                    common=dict(planner=key,split=split,scenario_seed=start+scenario,training_seed=seed,tick=tick+1,
                                state=json.dumps(state),R_Decay_action=ra,planner_on_same_state_action=a,
                                R_Decay_own_post_distance=int(b[tick,9]),planner_own_post_distance=int(p[tick,9]))
                    if 'speed2_disagreement' not in chosen and state[5]==2 and a!=ra:chosen['speed2_disagreement']=dict(rule='speed2_disagreement',**common)
                    if 'aligned_fire_to_move' not in chosen and state[:2]==state[2:4] and ra==5 and a in (0,1,2,3,6,7,8,9):
                        chosen['aligned_fire_to_move']=dict(rule='aligned_fire_to_move',**common)
                    if 'bounce_alignment_recovery_differs' not in chosen and b[tick,13]>0:
                        rb=recovery_alignment(b,tick);rp=recovery_alignment(p,tick)
                        if rb!=rp:chosen['bounce_alignment_recovery_differs']=dict(rule='bounce_alignment_recovery_differs',**common,
                                         R_recovery_after_bounce=rb,planner_recovery_after_bounce=rp)
                if len(chosen)==3:break
            if len(chosen)==3:break
        for rule in ('speed2_disagreement','bounce_alignment_recovery_differs','aligned_fire_to_move'):
            if rule in chosen:records.append(dict(found=True,**chosen[rule]))
            else:records.append(dict(found=False,planner=key,split=split,rule=rule,note='No event under the frozen selection rule in the evaluated trajectories.'))
    save_json(folder/'behavior_examples.json',dict(selection=c['planning']['behavior_selection'],examples=records,
              note='First qualifying event, never selected for better planner performance. Same-state actions and policy-specific post-distances are explicitly distinguished.'))
