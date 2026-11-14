"""Recompute records, paired effects and paired scenario/model intervals; no environment."""
from collections import defaultdict
import json
import numpy as np
from ..tracking_research.io import ROOT,read_json,save_json,save_csv
from ..tracking_planning.reference import require
from .core import OUT,methods,Policy
from .env import MOVES,mode
from .evaluation import metric_row,summary,slice_counts

METRICS=('D','C','H','A','first_close_tick','longest_far_streak','first_alignment_tick','alignment_opportunity_fraction','fire_given_alignment')

def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['split'],r['family']].append(r)
    result=[]
    for (split,family),rs in groups.items():
        row=dict(split=split,family=family,instances=len(rs),source_models=5 if family.startswith('Hybrid') else 0)
        for k in METRICS+('slow_fraction','fast_fraction','medium_fraction','stay_fraction','fire_fraction'):
            vals=[r[k] for r in rs if r[k] is not None]
            row[k]=float(np.mean(vals)) if vals else None;row[k+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        result.append(row)
    return result

def difference(a,b):return {k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in METRICS}

def gate(c,a,b):
    t=c['planning']['thresholds'];d=a['D']<=b['D'];close=a['C']-b['C']>=t['C_gain'];retain=a['H']>=t['H_retention']*b['H'];strong=a['H']>=t['H_simultaneous']*b['H'];acc=a['A'] is not None and a['A']>=t['A_minimum'];absolute=a['C']>=t['C_absolute']
    return dict(D_nonincrease=bool(d),C_plus_5pp=bool(close),H_retains_98pct=bool(retain),H_plus_2pct=bool(strong),A_at_least_98pct=bool(acc),
        practical_pass=bool(d and close and retain and acc),simultaneous_pass=bool(d and close and strong and acc),
        absolute_C75=bool(absolute),absolute_C75_and_hits_retained=bool(absolute and retain),H_ratio=a['H']/b['H'],**difference(a,b))

def window_values(r):
    return dict(D=r['distance_sum']/300,C=r['close_ticks']/300,H=r['hits'],A=r['hits']/r['shots'] if r['shots'] else None,
        first_close_tick=r['first_close_tick'],longest_far_streak=r['longest_far_streak'],first_alignment_tick=r['first_alignment_tick'],
        alignment_opportunity_fraction=r['alignment_opportunities']/300,fire_given_alignment=r['aligned_shots']/r['alignment_opportunities'] if r['alignment_opportunities'] else None)

def bootstrap(c,base,candidate,split,terminal):
    # Arrays: model x scenario x (distance_sum, close_ticks, hits, shots).
    nmodel,nscene,_=base.shape;rng=np.random.default_rng(c['planning']['bootstrap_seed']+(1 if split=='stress' else 0)+(10 if terminal=='Hybrid' else 0));samples=[]
    for _ in range(c['planning']['bootstrap_replicates']):
        models=rng.integers(nmodel,size=nmodel);scenes=rng.integers(nscene,size=nscene)
        a=base[models][:,scenes].sum(axis=1);b=candidate[models][:,scenes].sum(axis=1)
        av=np.column_stack([a[:,0]/(nscene*300),a[:,1]/(nscene*300),a[:,2]/nscene,np.divide(a[:,2],a[:,3],out=np.full(nmodel,np.nan),where=a[:,3]>0)])
        bv=np.column_stack([b[:,0]/(nscene*300),b[:,1]/(nscene*300),b[:,2]/nscene,np.divide(b[:,2],b[:,3],out=np.full(nmodel,np.nan),where=b[:,3]>0)])
        samples.append(np.nanmean(bv-av,axis=0))
    vals=np.array(samples);return [dict(split=split,terminal=terminal,metric=k,lower_95=float(np.quantile(vals[:,i],.025)),upper_95=float(np.quantile(vals[:,i],.975)),
        replicates=len(vals),source_models=nmodel,paired_scenarios=nscene,method=c['planning']['bootstrap']) for i,k in enumerate(('D','C','H','A'))]

def load_pair(folder,terminal,seed):
    tag=f'{terminal}-H3-A{{n}}'+(f'_s{seed}' if seed is not None else '')
    data=[]
    for n in (10,14):
        with np.load(folder/f'{tag.format(n=n)}_rollouts.npz',allow_pickle=False) as z:data.append((z['initial'],z['ticks'],z['scenario_seeds']))
    require(np.array_equal(data[0][0],data[1][0]) and np.array_equal(data[0][2],data[1][2]),'Paired reset differs')
    require(np.array_equal(data[0][1][:,:,2:6],data[1][1][:,:,2:6]),'Paired target route differs')
    return tag,data

def behavior_selection(c,split):
    folder=OUT/'evaluation'/split;examples=[];fragments=[]
    for terminal,seeds in [('Plan',[None]),('Hybrid',c['planning']['training_seeds'])]:
        data={s:load_pair(folder,terminal,s) for s in seeds};policies={s:Policy(c,next(m for m in methods(c) if m['actions']==14 and m['seed']==s)) for s in seeds}
        selected={};nscene=c['planning'][split]['count']
        for i in range(nscene):
            for seed in seeds:
                tag,d=data[seed];initial,a,scenes=d[0];b=d[1][1];ta,tb=a[i],b[i];before=np.vstack([initial[i],ta[:-1,:6]])
                last=-31
                for t,state in enumerate(before):
                    common_action=policies[seed].act(state);medium=common_action>=10 and ta[t,6]<10
                    row=dict(split=split,terminal=terminal,seed=seed,scenario=int(scenes[i]),tick=t+1,state=json.dumps(list(map(int,state))),
                        A10_action=int(ta[t,6]),A14_on_A10_state_action=common_action,A14_own_action=int(tb[t,6]),A10_own_distance=int(ta[t,9]),A14_own_distance=int(tb[t,9]))
                    rule=None
                    if 'speed2_medium' not in selected and state[5]==2 and medium:rule='speed2_medium'
                    if rule:selected[rule]=(row,i,seed,t)
                    if 'medium_regression' not in selected and medium and tb[t,9]>ta[t,9]:selected['medium_regression']=(row,i,seed,t)
                    if ta[t,13]>0 and t-last>30:
                        last=t
                        def recover(w):
                            indices=np.flatnonzero(w[t+1:min(300,t+31),7]==0)
                            return int(indices[0]) if len(indices) else None
                        ra,rb=recover(ta),recover(tb)
                        if 'bounce_recovery' not in selected and ra!=rb:selected['bounce_recovery']=(dict(**row,A10_alignment_recovery=ra,A14_alignment_recovery=rb),i,seed,t)
            if len(selected)==3:break
        for rule in ('speed2_medium','bounce_recovery','medium_regression'):
            if rule not in selected:examples.append(dict(split=split,terminal=terminal,rule=rule,found=False,note='No event satisfies the fixed rule.'));continue
            row,i,seed,t=selected[rule];examples.append(dict(rule=rule,found=True,**row));tag,d=data[seed]
            for n,(initial,ticks,scenes) in zip((10,14),d):
                before=np.vstack([initial[i],ticks[i,:-1,:6]])
                for j in range(max(0,t-5),min(300,t+16)):
                    fragments.append(dict(split=split,terminal=terminal,seed=seed,rule=rule,scenario=int(scenes[i]),tick=j+1,actions=n,
                        pre_state=json.dumps(list(map(int,before[j]))),post_state=json.dumps(list(map(int,ticks[i,j,:6]))),
                        action=int(ticks[i,j,6]),distance=int(ticks[i,j,9]),hit=int(ticks[i,j,11]),nominal_move=int(ticks[i,j,18]),actual_move=int(ticks[i,j,21])))
    return examples,fragments

def movement_statistics(split,spec,initial,ticks):
    before=np.concatenate([initial[:,None,:],ticks[:,:-1,:6]],axis=1);a=ticks[:,:,6];after=ticks[:,:,:6];nominal=np.array(MOVES)[a]
    boundary=np.any(np.minimum(before[:,:,:4],24-before[:,:,:4])<=3,axis=2)
    horizontal=nominal[:,:,0]!=0;vertical=nominal[:,:,1]!=0
    crossed=(horizontal&(before[:,:,1]==before[:,:,3])&((before[:,:,0]-before[:,:,2])*(after[:,:,0]-before[:,:,2])<0))|(
        vertical&(before[:,:,0]==before[:,:,2])&((before[:,:,1]-before[:,:,3])*(after[:,:,1]-before[:,:,3])<0))
    reverse=np.zeros_like(a,bool);reverse[:,1:]=(np.sum(nominal[:,1:]*nominal[:,:-1],axis=2)<0)
    groups={'all':np.ones(a.shape,bool),'pre_speed_1':before[:,:,5]==1,'pre_speed_2':before[:,:,5]==2,'post_speed_1':ticks[:,:,5]==1,'post_speed_2':ticks[:,:,5]==2,'boundary':boundary,'interior':~boundary,
        'pre_distance_0':ticks[:,:,7]==0,'pre_distance_1':ticks[:,:,7]==1,'pre_distance_2':ticks[:,:,7]==2,'pre_distance_3plus':ticks[:,:,7]>=3}
    rows=[]
    for group,m in groups.items():
        nt=int(m.sum());counts=np.bincount(a[m],minlength=14)
        rows.append(dict(split=split,policy=spec['id'],family=spec['family'],seed=spec['seed'],group=group,ticks=nt,
            **{f'action_{i}':int(counts[i]) for i in range(14)},**{f'{kind}_fraction':float(np.mean(np.isin(a[m],[i for i in range(14) if mode(i)==kind]))) if nt else None for kind in ('slow','fast','medium','stay','fire')},
            medium_actual_0=int(np.count_nonzero(m&(a>=10)&(ticks[:,:,21]==0))),medium_actual_1=int(np.count_nonzero(m&(a>=10)&(ticks[:,:,21]==1))),medium_actual_2=int(np.count_nonzero(m&(a>=10)&(ticks[:,:,21]==2))),
            overshoot_count=int(np.count_nonzero(m&crossed)),immediate_reversal_count=int(np.count_nonzero(m&reverse)),unaligned_stay_count=int(np.count_nonzero(m&(a==4)&(ticks[:,:,7]>1)))))
    return rows

def analyze(c,write=True):
    per=[];scenarios=[];actions=[];paired=[];gates=[];intervals=[];examples=[];fragments=[];slice_rows=[];event_rows=[];recovery_rows=[]
    all_windows={}
    for split in ('final','stress'):
        folder=OUT/'evaluation'/split
        for spec in methods(c):
            with np.load(folder/f'{spec["id"]}_rollouts.npz',allow_pickle=False) as z:initial=z['initial'];ticks=z['ticks'];seeds=z['scenario_seeds']
            rows=[dict(scenario_seed=int(seed),**metric_row(s,t)) for seed,s,t in zip(seeds,initial,ticks)]
            label=dict(split=split,policy=spec['id'],family=spec['family'],seed=spec['seed'],actions=spec['actions'])
            per.append(dict(**label,**summary(rows,ticks)));all_windows[split,spec['id']]=rows;actions+=movement_statistics(split,spec,initial,ticks)
            groups=defaultdict(lambda:np.zeros(11,np.int64));curves=defaultdict(lambda:[0,0]);recs=defaultdict(list)
            for s,t,seed in zip(initial,ticks,seeds):
                for name,v in slice_counts(s,t).items():groups[name]+=v
                pre=np.vstack([s,t[:-1,:6]])
                # Same-type suppression and [-5,+15] inherited, restricted to observable speed=2 at the event.
                for kind,col in [('bounce',13),('turn',14),('toggle',15)]:
                    last=-31
                    for j in np.flatnonzero(t[:,col]):
                        if j-last<=30:continue
                        last=int(j)
                        if pre[j,5]!=2:continue
                        for off in range(-5,16):
                            if 0<=j+off<300:curves[kind,off][0]+=int(t[j+off,9]);curves[kind,off][1]+=1
                from ..tracking_boundary_stability.analysis import dynamic_slices
                _,rs=dynamic_slices({'study':{'event_window':c['planning']['event_window']}},s,t)
                for r in rs:recs[r['kind']].append(r)
            for name,v in groups.items():
                nt,ds,cl,hi,sh,opp,af,med,m0,m1,m2=map(int,v)
                slice_rows.append(dict(**label,group=name,ticks=nt,distance_sum=ds,close_ticks=cl,hits=hi,shots=sh,alignment_opportunities=opp,aligned_shots=af,medium_count=med,medium_actual_0=m0,medium_actual_1=m1,medium_actual_2=m2,
                    D=ds/nt if nt else None,C=cl/nt if nt else None,H_per300=300*hi/nt if nt else None,A=hi/sh if sh else None,fire_given_alignment=af/opp if opp else None))
            for (kind,off),(ds,nt) in curves.items():event_rows.append(dict(**label,kind=kind,offset=off,distance_sum=ds,events=nt,D=ds/nt))
            for kind,rs in recs.items():
                recovered=[r['recovery_ticks'] for r in rs if r['recovered']]
                recovery_rows.append(dict(**label,kind=kind,events=len(rs),recovered=len(recovered),recovered_fraction=len(recovered)/len(rs),conditional_recovery_mean=float(np.mean(recovered)) if recovered else None,censored=len(rs)-len(recovered)))
        for terminal,source_seeds in [('Plan',[None]),('Hybrid',c['planning']['training_seeds'])]:
            base_mats=[];candidate_mats=[]
            for seed in source_seeds:
                tag=f'{terminal}-H3-A{{n}}'+(f'_s{seed}' if seed is not None else '')
                a=next(r for r in per if r['split']==split and r['policy']==tag.format(n=10));b=next(r for r in per if r['split']==split and r['policy']==tag.format(n=14))
                paired.append(dict(split=split,terminal=terminal,seed=seed,candidate=b['policy'],baseline=a['policy'],**difference(b,a),H_ratio=b['H']/a['H']))
                gates.append(dict(split=split,terminal=terminal,level='instance',seed=seed,**gate(c,b,a)))
                aw=all_windows[split,a['policy']];bw=all_windows[split,b['policy']];am=[];bm=[]
                for x,y in zip(aw,bw):
                    require(x['scenario_seed']==y['scenario_seed'] and x['target_sha256']==y['target_sha256'],'Window target pairing differs')
                    scenarios.append(dict(split=split,terminal=terminal,seed=seed,scenario_seed=x['scenario_seed'],**difference(window_values(y),window_values(x))))
                    am.append([x[k] for k in ('distance_sum','close_ticks','hits','shots')]);bm.append([y[k] for k in ('distance_sum','close_ticks','hits','shots')])
                base_mats.append(am);candidate_mats.append(bm)
            intervals+=bootstrap(c,np.asarray(base_mats),np.asarray(candidate_mats),split,terminal)
        ex,fr=behavior_selection(c,split);examples+=ex;fragments+=fr
    comparison=aggregate(per)
    for split in ('final','stress'):
        for terminal in ('Plan','Hybrid'):
            a=next(r for r in comparison if r['split']==split and r['family']==terminal+'-H3-A10');b=next(r for r in comparison if r['split']==split and r['family']==terminal+'-H3-A14')
            gates.append(dict(split=split,terminal=terminal,level='family_mean',seed=None,**gate(c,b,a)))
    result=dict(per_policy=per,comparison=comparison,paired_differences=paired,thresholds=gates,bootstrap_intervals=intervals)
    if write:
        for name,rows in result.items():save_json(OUT/f'{name}.json',rows);save_csv(OUT/f'{name}.csv',rows)
        save_csv(OUT/'per_seed.csv',[r for r in per if r['seed'] is not None]);save_csv(OUT/'scenario_paired_differences.csv',scenarios)
        save_csv(OUT/'action_usage.csv',actions);save_csv(OUT/'speed_boundary_slices.csv',slice_rows);save_csv(OUT/'high_speed_event_curves.csv',event_rows);save_csv(OUT/'recovery.csv',recovery_rows)
        mechanism=[]
        for split in ('final','stress'):
            for terminal,seeds in [('Plan',[None]),('Hybrid',c['planning']['training_seeds'])]:
                for seed in seeds:
                    for group in ('all','pre_speed_1','pre_speed_2','boundary','interior'):
                        a=next(r for r in actions if r['split']==split and r['seed']==seed and r['family']==terminal+'-H3-A10' and r['group']==group)
                        b=next(r for r in actions if r['split']==split and r['seed']==seed and r['family']==terminal+'-H3-A14' and r['group']==group)
                        mechanism.append(dict(split=split,terminal=terminal,seed=seed,group=group,
                            **{k+'_delta':b[k+'_fraction']-a[k+'_fraction'] for k in ('slow','fast','medium','stay','fire')},
                            overshoot_rate_delta=b['overshoot_count']/b['ticks']-a['overshoot_count']/a['ticks'],
                            reversal_rate_delta=b['immediate_reversal_count']/b['ticks']-a['immediate_reversal_count']/a['ticks'],
                            unaligned_wait_rate_delta=b['unaligned_stay_count']/b['ticks']-a['unaligned_stay_count']/a['ticks']))
        save_csv(OUT/'mechanism_comparison.csv',mechanism)
        save_json(OUT/'behavior_selection.json',examples);save_csv(OUT/'behavior_fragments.csv',fragments)
    return result
