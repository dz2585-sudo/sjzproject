"""Paired 300-tick evaluation; extended columns never use historical fast>=6 logic."""
from collections import defaultdict
import copy,time
import numpy as np
from ..tracking_research.io import ROOT,read_json,save_json,save_csv
from ..tracking_tradeoff.core import window_metrics,summarize
from ..tracking_boundary_stability.analysis import dynamic_slices
from ..tracking_planning.reference import require
from .env import ActionEnv,mode
from .core import OUT,Native,Policy,methods,bundle,state_id
from .study import models

COLUMNS=('xc','yc','xt','yt','heading','speed','action','d_pre','d_before_target_motion','d_post','target_net_displacement',
    'hit','miss','reflections','turn','toggle','proposed','observable_speed','nominal_move','actual_dx','actual_dy','actual_move')

def metric_row(initial,ticks):
    r=window_metrics(tuple(map(int,initial)),ticks[:,:16],False)
    aligned=np.flatnonzero(ticks[:,7]==0)
    return dict(**r,first_alignment_tick=int(aligned[0])+1 if len(aligned) else 300,
        medium_count=int(np.count_nonzero(ticks[:,6]>=10)),medium_actual_2=int(np.count_nonzero((ticks[:,6]>=10)&(ticks[:,21]==2))),
        medium_actual_1=int(np.count_nonzero((ticks[:,6]>=10)&(ticks[:,21]==1))),medium_actual_0=int(np.count_nonzero((ticks[:,6]>=10)&(ticks[:,21]==0))))

def summary(rows,ticks):
    r=summarize(rows);flat=ticks.reshape(-1,22);a=flat[:,6]
    r.update(first_alignment_tick=float(np.mean([x['first_alignment_tick'] for x in rows])),
        medium_count=sum(x['medium_count'] for x in rows),**{f'medium_actual_{v}':sum(x[f'medium_actual_{v}'] for x in rows) for v in (0,1,2)},
        **{f'{m}_fraction':float(np.mean(np.array([mode(int(x))==m for x in a]))) for m in ('slow','fast','medium','stay','fire')})
    return r

def slice_counts(initial,ticks):
    before=np.vstack([initial,ticks[:-1,:6]]);boundary=np.any(np.minimum(before[:,:4],24-before[:,:4])<=3,axis=1)
    close=np.flatnonzero(ticks[:,9]<=1);acquisition=np.arange(300)<=(int(close[0]) if len(close) else 299)
    groups=dict(all=np.ones(300,bool),boundary=boundary,interior=~boundary,pre_speed_1=before[:,5]==1,pre_speed_2=before[:,5]==2,
        post_speed_1=ticks[:,5]==1,post_speed_2=ticks[:,5]==2,acquisition=acquisition,sustained=~acquisition)
    for speed in (1,2):
        for edge,m in [('boundary',boundary),('interior',~boundary)]:groups[f'pre_speed_{speed}_{edge}']=(before[:,5]==speed)&m
    a=ticks[:,6];aligned=ticks[:,7]==0;result={}
    for name,m in groups.items():
        result[name]=np.array([m.sum(),ticks[m,9].sum(),np.count_nonzero(ticks[m,9]<=1),ticks[m,11].sum(),np.count_nonzero(a[m]==5),
            np.count_nonzero(aligned&m),np.count_nonzero(aligned&m&(a==5)),np.count_nonzero(m&(a>=10)),
            *[np.count_nonzero(m&(a>=10)&(ticks[:,21]==v)) for v in (0,1,2)]],np.int64)
    return result

def evaluate_split(c,split):
    spec=c['planning'][split];cc=copy.deepcopy(c)
    if split=='stress':cc['environment'].update(turn_probability=spec['turn_probability'],speed_toggle_probability=spec['speed_toggle_probability'])
    folder=OUT/'evaluation'/split;folder.mkdir(parents=True,exist_ok=False);target={};per=[];cost=[];slices=[];events=[]
    native=Native(c);before_models=models();start=time.perf_counter()
    for method in methods(c):
        load=time.perf_counter();p=Policy(c,method);load=time.perf_counter()-load;beg=time.perf_counter();initials=[];traces=[];times=[];rows=[]
        groups=defaultdict(lambda:np.zeros(11,np.int64));curves=defaultdict(list);recoveries=[]
        for seed in range(spec['start'],spec['start']+spec['count']):
            e=ActionEnv(cc,method['actions']);initial,_=e.reset(seed);ticks,ns=native.run(e,initial,p)
            row=dict(scenario_seed=seed,**metric_row(initial,ticks));signature=(tuple(initial),row['target_sha256'])
            require(target.setdefault(str(seed),signature)==signature,'Target/initial scenario pairing differs')
            require(not np.any((ticks[:,6]==5)&(ticks[:,7]>0)),'Guard violation')
            initials.append(initial);traces.append(ticks);times.append(ns);rows.append(row)
            for k,v in slice_counts(initial,ticks).items():groups[k]+=v
            ev,rs=dynamic_slices({'study':{'event_window':c['planning']['event_window']}},initial,ticks)
            for kind,offset,d in ev:curves[kind,offset].append(d)
            recoveries.extend(dict(scenario_seed=seed,**r) for r in rs)
        arr=np.asarray(traces,np.int16);ns=np.asarray(times,np.int64);label=dict(split=split,policy=method['id'],family=method['family'],seed=method['seed'],actions=method['actions'])
        per.append(dict(**label,**summary(rows,arr)));save_csv(folder/f'{method["id"]}_windows.csv',rows)
        np.savez_compressed(folder/f'{method["id"]}_rollouts.npz',initial=np.asarray(initials,np.int16),ticks=arr,decision_ns=ns,
            scenario_seeds=np.arange(spec['start'],spec['start']+spec['count']),columns=np.asarray(COLUMNS))
        for group,values in groups.items():
            nt,ds,cl,hi,sh,opp,af,med,m0,m1,m2=map(int,values)
            slices.append(dict(**label,group=group,ticks=nt,distance_sum=ds,close_ticks=cl,hits=hi,shots=sh,alignment_opportunities=opp,aligned_shots=af,
                medium_count=med,medium_actual_0=m0,medium_actual_1=m1,medium_actual_2=m2,D=ds/nt if nt else None,C=cl/nt if nt else None,
                H_per300=300*hi/nt if nt else None,A=hi/sh if sh else None,fire_given_alignment=af/opp if opp else None))
        for (kind,offset),values in curves.items():events.append(dict(**label,kind=kind,offset=offset,events=len(values),distance_sum=sum(values),D=float(np.mean(values))))
        save_csv(folder/f'{method["id"]}_recovery_events.csv',recoveries)
        cost.append(dict(**label,load_seconds=load,evaluation_storage_seconds=time.perf_counter()-beg,environment_ticks=int(ns.size),policy_lookups=int(ns.size),
            decision_mean_ns=float(ns.mean()),decision_p95_ns=float(np.quantile(ns,.95)),decision_max_ns=int(ns.max()),online_expansions=0,online_backups=0,
            scope='Native table query + firing guard + timing overhead; reset, compression, diagnostics separately in wall time.',training_interactions=0,Q_updates=0))
        print(f'{split} {method["id"]}: D={per[-1]["D"]:.6f} C={per[-1]["C"]:.6f} H={per[-1]["H"]:.4f}',flush=True)
    require(models()==before_models,'Evaluation changed frozen Q/count tables')
    save_json(folder/'per_policy.json',per);save_csv(folder/'per_policy.csv',per);save_csv(folder/'runtime.csv',cost)
    save_csv(folder/'slices.csv',slices);save_csv(folder/'event_curves.csv',events);save_json(folder/'paired_targets.json',target)
    from ..reproduce import memory_record
    result=dict(policy_instances=12,windows=12*spec['count'],environment_ticks=3600*spec['count'],policy_lookups=3600*spec['count'],
        elapsed_seconds=time.perf_counter()-start,memory=memory_record(),fixed_model_probabilities=[.05,.05],real_probabilities=[cc['environment']['turn_probability'],cc['environment']['speed_toggle_probability']],
        frozen_models_unchanged=True,training_interactions=0,Q_updates=0)
    save_json(folder/'complete.json',result);return result

def common_diagnostics(c):
    source=ROOT/c['planning']['common_diagnostics']['source']
    with np.load(source,allow_pickle=False) as z:states=z['states'];freq=z['frequency']
    ids=np.array([state_id(s) for s in states]);folder=OUT/'diagnostics';folder.mkdir(exist_ok=True);rows=[];summary_rows=[]
    for seed in [None]+c['planning']['training_seeds']:
        data=[]
        for n in (10,14):
            with np.load(bundle(c,n,seed,3),allow_pickle=False) as z:data.append((z['policy'][ids],z['value'][ids]))
        aa,va=data[0];ab,vb=data[1]
        for i,s in enumerate(states):rows.append(dict(seed=seed,**dict(zip(('xc','yc','xt','yt','heading','speed'),map(int,s))),frequency=int(freq[i]),
            A10_action=int(aa[i]),A14_action=int(ab[i]),A10_value=float(va[i]),A14_value=float(vb[i]),value_gain=float(vb[i]-va[i])))
        boundary=np.any(np.minimum(states[:,:4],24-states[:,:4])<=3,axis=1)
        for group,m in [('all',np.ones(len(states),bool)),('speed_2',states[:,5]==2),('boundary',boundary),('interior',~boundary)]:
            summary_rows.append(dict(seed=seed,group=group,states=int(m.sum()),disagreement=float(np.mean(aa[m]!=ab[m])),
                medium_fraction=float(np.mean(ab[m]>=10)),mean_value_gain=float(np.mean(vb[m]-va[m])),minimum_value_gain=float(np.min(vb[m]-va[m])),
                frequency_weighted_medium=float(np.average(ab[m]>=10,weights=freq[m]))))
    save_csv(folder/'common_states.csv',rows);save_csv(folder/'common_comparison.csv',summary_rows)
    from ..tracking_research.io import sha
    save_json(folder/'source.json',dict(source=source.relative_to(ROOT).as_posix(),sha256=sha(source),states=len(states),visitation_events=int(freq.sum()),environment_ticks=0,
        meaning='Equal distinct physical states and separately prior-visitation-weighted counts. No new holdout/future exposed.'))
