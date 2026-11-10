"""Matched contrasts, published threshold checks and Q-value diagnostics."""
from collections import Counter
import statistics

from ..tracking_research.io import read_json, save_csv, save_json
from .history import METRICS, group, number, rows, stat
from .replay import require


def export(output, name, records):
    save_csv(output/f'{name}.csv',[{k:'NA' if v is None else v for k,v in r.items()} for r in records])


def delta(candidate, reference):
    return {k:candidate[k]-reference[k] if candidate[k] is not None and reference[k] is not None else None
            for k in METRICS}


def screen(conditions, groups, key, baseline, gates):
    screens=[]
    for condition in conditions:
        r=groups[condition]
        d=delta(r,baseline)
        c=r['C']>=baseline['C']+gates['close_gain']
        distance=r['D']<=baseline['D']
        gain=r['H']>=baseline['H']*gates['simultaneous_hits_ratio']
        retain=r['H']>=baseline['H']*gates['retention_hits_ratio']
        accuracy=r['A_defined_n']==3 and r['A']>=gates['minimum_accuracy']
        pooled=r['A_pooled'] is not None and r['A_pooled']>=gates['minimum_accuracy']
        screens.append({'comparison':key,'condition':condition,'delta_D':d['D'],'delta_C':d['C'],
                        'delta_H':d['H'],'relative_H':r['H']/baseline['H']-1,
                        'A_seed_mean':r['A'],'A_pooled':r['A_pooled'],
                        'close_gate':c,'distance_gate':distance,'gain_gate':gain,'retention_gate':retain,
                        'accuracy_gate':accuracy,'accuracy_gate_pooled':pooled,
                        'simultaneous':c and distance and gain and accuracy,
                        'retention':c and distance and retain and accuracy,
                        'retention_with_hit_loss':c and distance and retain and accuracy and d['H']<0,
                        'simultaneous_pooled_sensitivity':c and distance and gain and pooled,
                        'retention_pooled_sensitivity':c and distance and retain and pooled})
    return screens


def compile_metrics(output, protocol):
    observations=read_json(output/'evaluation/per_seed.json')
    keys={(r['split'],r['condition'],r['training_seed'],r['variant']):r for r in observations}
    conditions=list(protocol['config']['tradeoff']['conditions'])
    seeds=protocol['config']['tradeoff']['training_seeds']
    summaries, within, reference, contrasts=[],[],[],[]
    screens=[]
    gates=protocol['config']['tradeoff']['thresholds']
    for split in ('final','stress'):
        for variant in ('original','guard'):
            groups={}
            for c in conditions:
                matching=[keys[split,c,s,variant] for s in seeds]
                r=group(matching)
                groups[c]=r
                summaries.append({'split':split,'condition':c,'variant':variant,
                                  'training_seeds':3,**r})
                for s,a in zip(seeds,matching):
                    reference.append({'split':split,'condition':c,'training_seed':s,'variant':variant,
                                      **delta(a,keys[split,'reference',s,variant])})
            if split=='final':
                screens.extend(screen(conditions,groups,'new_'+variant,groups['reference'],gates))
        for c in conditions:
            for s in seeds:
                a,b=keys[split,c,s,'original'],keys[split,c,s,'guard']
                d=delta(b,a)
                require(d['D']==d['C']==d['H']==0 and b['total_misses']==0
                        and abs(d['base_return']-a['misses'])<1e-10,'Invariant changed at seed level')
                within.append({'split':split,'condition':c,'training_seed':s,
                               'guard_removed_misses':a['total_misses'],**d})
            group_rows=[r for r in within if (r['split'],r['condition'])==(split,c)]
            result={'split':split,'condition':c,'removed_misses':sum(r['guard_removed_misses'] for r in group_rows)}
            for k in METRICS:
                result[k],result[k+'_sd'],result[k+'_defined_n']=stat([r[k] for r in group_rows])
            contrasts.append(result)
        summaries.append({'split':split,'condition':'rule','variant':'original',
                          'training_seeds':0,**group([keys[split,'rule',None,'original']])})
    historical={}
    for r in rows(output/'historical_comparison.csv'):
        if r['split']=='final' and r['condition']!='rule':
            historical[r['condition']]={k:number(v) for k,v in r.items() if k not in ('split','condition')}
    screens.extend(screen(conditions,historical,'previous_original',historical['reference'],gates))
    for name,data in (('comparison',summaries),('paired_differences',within),
                      ('paired_delta_summary',contrasts),('paired_vs_reference',reference),('thresholds',screens)):
        export(output,name,data)
    save_json(output/'comparison.json',summaries)
    save_json(output/'thresholds.json',screens)
    return summaries,within,screens


def q_statistics(events):
    gap=[float(r['q_gap']) for r in events]
    fire=[int(r['fire_updates']) for r in events]
    stay=[int(r['stay_updates']) for r in events]
    minimum=[min(a,b) for a,b in zip(fire,stay)]
    return {'events':len(events),'fire_stay_ties':sum(r['fire_stay_tied']=='True' for r in events),
            'max_action_ties':sum(int(r['max_tie_count'])>1 for r in events),
            'positive_q_gap':sum(v>0 for v in gap),'q_gap_median':statistics.median(gap) if gap else None,
            'q_gap_min':min(gap) if gap else None,'q_gap_max':max(gap) if gap else None,
            'fire_updates_median':statistics.median(fire) if fire else None,
            'stay_updates_median':statistics.median(stay) if stay else None,
            'fire_zero_updates':fire.count(0),'stay_zero_updates':stay.count(0),
            'either_zero_updates':minimum.count(0),'either_below_10_updates':sum(v<10 for v in minimum),
            'both_at_least_100_updates':sum(v>=100 for v in minimum)}


def diagnostics(output, protocol, within):
    per_seed, per_condition, states, totals=[],[],[],{}
    expected={(r['split'],r['condition'],r['training_seed']):r['guard_removed_misses'] for r in within}
    for split in ('final','stress'):
        all_events=[]
        for c in protocol['config']['tradeoff']['conditions']:
            condition_events=[]
            for s in protocol['config']['tradeoff']['training_seeds']:
                path=output/'q_events'/split/f'{c}_seed{s}.csv'
                events=rows(path) if path.is_file() else []
                require(len(events)==expected[split,c,s],'Q event count differs from paired miss count')
                by_state={}
                for event in events:
                    state=tuple(event[k] for k in ('relative_dx','relative_dy','heading','speed'))
                    require(int(event['d_pre'])>0 and '5' in event['tied_max_action_ids'].split()
                            and float(event['q_fire'])>float(event['q_stay']), 'Invalid greedy Fire evidence')
                    by_state.setdefault(state,[]).append(event)
                for state, repetitions in by_state.items():
                    representative=repetitions[0]
                    keys=('q_fire','q_stay','q_gap','fire_updates','stay_updates','fire_stay_tied',
                          'max_tie_count','tied_max_action_ids')
                    require(all(all(e[k]==representative[k] for k in keys) for e in repetitions),
                            'Frozen Q diagnostics changed for the same indexed state')
                    states.append({'split':split,'condition':c,'training_seed':s,'dx':state[0],'dy':state[1],
                                   'heading':state[2],'speed':state[3],'event_count':len(repetitions),
                                   **{k:representative[k] for k in keys}})
                per_seed.append({'split':split,'condition':c,'training_seed':s,
                                 'distinct_model_state_rows':len(by_state),**q_statistics(events)})
                condition_events.extend(events)
            per_condition.append({'split':split,'condition':c,
                                  'distinct_model_state_rows':sum(r['distinct_model_state_rows']
                                                                   for r in per_seed if (r['split'],r['condition'])==(split,c)),
                                  **q_statistics(condition_events)})
            all_events.extend(condition_events)
        totals[split]=q_statistics(all_events)
        totals[split]['by_pre_distance']=dict(sorted(Counter(int(r['d_pre']) for r in all_events).items()))
    export(output,'q_diagnostics_per_seed',per_seed)
    export(output,'q_diagnostics_by_condition',per_condition)
    export(output,'q_diagnostics_distinct_states',states)
    save_json(output/'q_diagnostics_summary.json',totals)
    return totals
