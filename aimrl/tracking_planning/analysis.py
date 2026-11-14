"""Paired policy/model summaries without inventing pure-planner training seeds."""
from collections import defaultdict
import json

import numpy as np

from ..tracking_research.io import read_csv,read_json,save_csv,save_json
from .core import OUT

METRICS=('D','C','H','A','first_close_tick','longest_far_streak','first_alignment_tick',
         'alignment_opportunity_fraction','fire_given_alignment','aligned_move_fraction','invalid_proposed_fire_fraction')


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['split'],r['family']].append(r)
    result=[]
    for (split,family),rs in groups.items():
        entry=dict(split=split,family=family,instances=len(rs),trained_model_instances=sum(r['seed'] is not None for r in rs))
        for metric in METRICS:
            values=[r[metric] for r in rs if r[metric] is not None]
            entry[metric]=float(np.mean(values)) if values else None
            entry[metric+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            entry[metric+'_defined']=len(values)
            entry[metric+'_worst']=float(min(values) if metric in ('C','H','A','alignment_opportunity_fraction','fire_given_alignment') else max(values)) if values else None
        result.append(entry)
    return result


def difference(candidate,baseline,comparison,split,seed=None,policy=None,scenario=None):
    return dict(split=split,comparison=comparison,candidate=policy or candidate.get('policy',candidate.get('family')),
                baseline=baseline.get('policy',baseline.get('family')),baseline_seed=seed,scenario_seed=scenario,
                **{k:candidate[k]-baseline[k] if candidate[k] is not None and baseline[k] is not None else None for k in METRICS})


def paired_comparisons(rows,comparison):
    paired=[];contrasts=[]
    for split in ('final','stress'):
        rs=[r for r in rows if r['split']==split];byid={r['policy']:r for r in rs}
        means={r['family']:r for r in comparison if r['split']==split}
        base=[r for r in rs if r['family']=='R-Decay']
        for r in rs:
            if r['family']=='R-Decay':continue
            refs=[b for b in base if b['seed']==r['seed']] if r['seed'] is not None else base
            for b in refs:paired.append(difference(r,b,'versus_R-Decay_model',split,b['seed']))
            if r['seed'] is None:paired.append(difference(r,means['R-Decay'],'versus_R-Decay_mean',split))
        for a,b,name in (('Plan-H2','Plan-H1','horizon_2_minus_1'),('Plan-H3','Plan-H2','horizon_3_minus_2'),
                         ('Plan-H3','Plan-H3-Zero','Phi_terminal_minus_zero'),('Hybrid-H3','Plan-H3','learned_terminal_minus_Phi')):
            contrasts.append(difference(means[a],means[b],name,split))
        for r in rs:
            if r['family']=='Hybrid-H3':contrasts.append(difference(r,byid['Plan-H3'],'each_Q_terminal_minus_Phi',split,r['seed']))
    return paired,contrasts


def gate(c,candidate,baseline):
    t=c['planning']['thresholds']
    d=candidate['D']<=baseline['D'];close=candidate['C']>=baseline['C']+t['C_gain']
    retained=candidate['H']>=t['H_retention']*baseline['H'];strong=candidate['H']>=t['H_simultaneous']*baseline['H']
    accuracy=candidate['A'] is not None and candidate['A']>=t['A_minimum']
    if 'instances' in candidate:accuracy=accuracy and candidate['A_defined']==candidate['instances']
    absolute=candidate['C']>=t['C_absolute']
    return dict(D_nonincrease=d,C_plus_5pp=close,H_retains_98pct=retained,H_plus_2pct=strong,A_at_least_98pct=accuracy,
                practical_pass=d and close and retained and accuracy,simultaneous_pass=d and close and strong and accuracy,
                absolute_C75=absolute,absolute_C75_and_H_retained=absolute and retained,
                D_difference=candidate['D']-baseline['D'],C_difference=candidate['C']-baseline['C'],
                H_ratio=candidate['H']/baseline['H'] if baseline['H'] else None)


def thresholds(c,rows,comparison):
    result=[]
    for split in ('final','stress'):
        rs=[r for r in rows if r['split']==split];base=[r for r in rs if r['family']=='R-Decay']
        means={r['family']:r for r in comparison if r['split']==split}
        for family,r in means.items():result.append(dict(split=split,level='family_mean',candidate=family,baseline='R-Decay mean',baseline_seed=None,**gate(c,r,means['R-Decay'])))
        for r in rs:
            refs=[b for b in base if b['seed']==r['seed']] if r['seed'] is not None else base
            for b in refs:result.append(dict(split=split,level='model_pair' if r['seed'] is not None else 'pure_vs_each_model',
                              candidate=r['policy'],baseline=b['policy'],baseline_seed=b['seed'],**gate(c,r,b)))
    return result


def scenario_differences(c):
    rows=[]
    for split in ('final','stress'):
        folder=OUT/'evaluation'/split
        per=read_json(folder/'per_policy.json');baselines={r['seed']:r for r in per if r['family']=='R-Decay'}
        windows={r['policy']:read_csv(folder/f'{r["policy"]}_windows.csv') for r in per}
        for r in per:
            if r['family']=='R-Decay':continue
            seeds=[r['seed']] if r['seed'] is not None else c['planning']['training_seeds']
            for seed in seeds:
                base=baselines[seed]
                for a,b in zip(windows[r['policy']],windows[base['policy']]):
                    scenario=int(a['scenario_seed'])
                    if scenario!=int(b['scenario_seed']):raise ValueError('Scenario alignment differs')
                    ash,bsh=int(a['shots']),int(b['shots']);ahi,bhi=int(a['hits']),int(b['hits'])
                    rows.append(dict(split=split,candidate=r['policy'],baseline=base['policy'],baseline_seed=seed,scenario_seed=scenario,
                        D=(int(a['distance_sum'])-int(b['distance_sum']))/300,C=(int(a['close_ticks'])-int(b['close_ticks']))/300,H=ahi-bhi,
                        A=ahi/ash-bhi/bsh if ash and bsh else None,
                        first_close_tick=int(a['first_close_tick'])-int(b['first_close_tick']),
                        longest_far_streak=int(a['longest_far_streak'])-int(b['longest_far_streak'])))
    return rows


def behavior_quantities(c):
    """Aggregate observable action patterns; no inferred future enters a policy."""
    results=[]
    for split in ('final','stress'):
        for entry in read_json(OUT/'evaluation'/split/'per_policy.json'):
            with np.load(OUT/'evaluation'/split/f'{entry["policy"]}_rollouts.npz',allow_pickle=False) as z:initial=z['initial'];ticks=z['ticks']
            before=np.concatenate([initial[:,None,:],ticks[:,:-1,:6]],axis=1)
            after=ticks[:,:,:6];a=ticks[:,:,6];h=(a==0)|(a==1)|(a==6)|(a==7);v=(a==2)|(a==3)|(a==8)|(a==9)
            crossed=(h&(before[:,:,1]==before[:,:,3])&((before[:,:,0]-before[:,:,2])*(after[:,:,0]-before[:,:,2])<0))|(
                     v&(before[:,:,0]==before[:,:,2])&((before[:,:,1]-before[:,:,3])*(after[:,:,1]-before[:,:,3])<0))
            # Int16 products are bounded by 24², safe here.
            for speed in (1,2):
                m=before[:,:,5]==speed;nt=int(m.sum());aligned=ticks[:,:,7]==0
                fast=a>=6;slow=a<=3;wait=a==4;fire=a==5
                results.append(dict(split=split,policy=entry['policy'],family=entry['family'],seed=entry['seed'],pre_speed=speed,ticks=nt,
                   fast_fraction=float(fast[m].mean()),slow_fraction=float(slow[m].mean()),stay_fraction=float(wait[m].mean()),fire_fraction=float(fire[m].mean()),
                   target_axis_crossing_fraction=float(crossed[m].mean()),
                   aligned_opportunities=int((aligned&m).sum()),aligned_move_count=int((aligned&m&(fast|slow)).sum()),
                   first_alignment_tick=entry['first_alignment_tick'],
                   interpretation='Descriptive on-policy movement across current target coordinate on its exact row/column; not proof of causal overshoot avoidance.'))
    return results


def write_tables(c):
    rows=[]
    for split in ('final','stress'):rows+=read_json(OUT/'evaluation'/split/'per_policy.json')
    comparison=aggregate(rows);paired,contrasts=paired_comparisons(rows,comparison);gates=thresholds(c,rows,comparison)
    save_json(OUT/'per_policy.json',rows);save_csv(OUT/'per_policy.csv',rows)
    save_csv(OUT/'per_seed.csv',[r for r in rows if r['seed'] is not None])
    save_json(OUT/'comparison.json',comparison);save_csv(OUT/'comparison.csv',comparison)
    save_csv(OUT/'paired_differences.csv',paired);save_csv(OUT/'factor_contrasts.csv',contrasts)
    save_json(OUT/'thresholds.json',gates);save_csv(OUT/'thresholds.csv',gates)
    save_csv(OUT/'scenario_paired_differences.csv',scenario_differences(c));save_csv(OUT/'behavior_quantities.csv',behavior_quantities(c))
    return rows,comparison,paired,contrasts,gates
