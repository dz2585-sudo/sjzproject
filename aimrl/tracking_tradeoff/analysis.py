"""Aggregate by training seed and audit the stored evidence."""
import csv
import json
import math
import statistics

import numpy as np

from ..tracking_research.io import ROOT, read_json, save_csv, save_json, sha
from .core import MODES, load_model, summarize
from .study import require, run_dir, scenario_seeds

METRICS = ('D', 'C', 'H', 'A', 'shots', 'misses', 'base_return', 'longest_far_streak',
           'first_close_tick', 'acquired_fraction', 'alignment_opportunity_fraction',
           'fire_given_alignment', 'hit_nonclose_fraction')


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def defined_stats(values):
    valid = [x for x in values if x is not None]
    return (statistics.mean(valid) if valid else None,
            statistics.stdev(valid) if len(valid) > 1 else None, len(valid))


def aggregate(config, output):
    design = config['tradeoff']
    per_seed = read_json(output / 'evaluation/per_seed.json')
    comparison, differences, slices, actions, histograms, mechanisms = [], [], [], [], [], []
    for split in ('final','stress'):
        selected = [r for r in per_seed if r['split'] == split]
        baseline = {r['seed']: r for r in selected if r['condition'] == 'reference'}
        for condition in (*design['conditions'], 'rule'):
            rows = [r for r in selected if r['condition'] == condition]
            weights = design['conditions'].get(condition, [None, None])
            group = {'split': split, 'condition': condition, 'lambda_distance': weights[0],
                     'beta_close': weights[1], 'training_seeds': len(rows) if condition != 'rule' else 0}
            for k in METRICS:
                group[k], group[k + '_sd'], group[k + '_defined_n'] = defined_stats([r[k] for r in rows])
            group['total_shots'] = sum(r['total_shots'] for r in rows)
            group['total_hits'] = sum(r['total_hits'] for r in rows)
            group['total_close_ticks'] = sum(r['total_close_ticks'] for r in rows)
            group['total_hit_nonclose'] = sum(r['total_hit_nonclose'] for r in rows)
            group['total_ticks'] = sum(r['ticks'] for r in rows)
            comparison.append(group)
            for r in rows:
                seed = r['seed']
                if seed is not None:
                    ref = baseline[seed]
                    paired = {'split':split,'condition':condition,'seed':seed}
                    for k in METRICS:
                        paired[k] = r[k] - ref[k] if r[k] is not None and ref[k] is not None else None
                    differences.append(paired)
                    o, f = (r[k] for k in ('alignment_opportunity_fraction','fire_given_alignment'))
                    o0, f0 = (ref[k] for k in ('alignment_opportunity_fraction','fire_given_alignment'))
                    if f is not None and f0 is not None:
                        opportunity = 300 * (o - o0) * f0
                        decision = 300 * o * (f - f0)
                        require(math.isclose(opportunity + decision, paired['H'], abs_tol=1e-9), 'Hit decomposition mismatch')
                    else:
                        opportunity = decision = None
                    mechanisms.append({'split':split,'condition':condition,'seed':seed,
                                       'delta_hits_per_window':paired['H'],
                                       'opportunity_component':opportunity,'decision_component':decision})
                name = condition if seed is None else f'{condition}_seed{seed}'
                windows = read_rows(output / 'evaluation' / split / f'{name}_windows.csv')
                action = np.sum([json.loads(w['pre_distance_action_counts']) for w in windows], axis=0)
                hit_hist = np.sum([json.loads(w['hit_post_histogram']) for w in windows], axis=0)
                require(int(action.sum()) == r['ticks'] and int(hit_hist.sum()) == r['total_hits'], 'Action or hit histogram mismatch')
                for d in range(4):
                    denominator = int(action[d].sum())
                    actions.append({'split':split,'condition':condition,'seed':seed,
                                    'pre_distance_bin':str(d) if d < 3 else '3_plus','ticks':denominator,
                                    **{f'{m}_count':int(v) for m,v in zip(MODES, action[d])},
                                    **{f'{m}_fraction':int(v)/denominator if denominator else None
                                       for m,v in zip(MODES, action[d])}})
                for distance, count in enumerate(hit_hist):
                    histograms.append({'split':split,'condition':condition,'seed':seed,
                                       'post_distance':distance,'hits':int(count),'denominator':r['total_hits']})
                by_slice = [json.loads(w['slice_counts']) for w in windows]
                for name in by_slice[0]:
                    ticks, distance, close, hits, shots, misses = map(int, np.sum([w[name] for w in by_slice], axis=0))
                    slices.append({'split':split,'condition':condition,'seed':seed,'slice':name,'ticks':ticks,
                                   'distance_sum':distance,'close_ticks':close,'hits':hits,'shots':shots,'misses':misses,
                                   'D':distance/ticks if ticks else None,'C':close/ticks if ticks else None,
                                   'hit_rate':hits/ticks if ticks else None,'A':hits/shots if shots else None})
    paired_groups = []
    for split in ('final','stress'):
        for condition in design['conditions']:
            group = {'split':split,'condition':condition}
            rows = [r for r in differences if r['split'] == split and r['condition'] == condition]
            for k in METRICS:
                group[k], group[k+'_sd'], group[k+'_defined_n'] = defined_stats([r[k] for r in rows])
            for k, direction in (('D', -1), ('C', 1), ('H', 1)):
                group[k+'_seeds_improved'] = sum(direction * r[k] > 0 for r in rows)
            paired_groups.append(group)
    slice_groups = []
    for split in ('final','stress'):
        for condition in (*design['conditions'], 'rule'):
            for name in ('speed_1','speed_2','boundary','interior'):
                rows = [r for r in slices if (r['split'],r['condition'],r['slice']) == (split,condition,name)]
                group = {'split':split,'condition':condition,'slice':name,
                         'ticks':sum(r['ticks'] for r in rows),'shots':sum(r['shots'] for r in rows)}
                for k in ('D','C','hit_rate','A'):
                    group[k], group[k+'_sd'], group[k+'_defined_n'] = defined_stats([r[k] for r in rows])
                slice_groups.append(group)
    final = [r for r in comparison if r['split'] == 'final']
    ref = next(r for r in final if r['condition'] == 'reference')
    threshold = design['thresholds']
    screens = []
    for row in final:
        if row['condition'] == 'rule':
            continue
        result = {'condition':row['condition'],'delta_D':row['D']-ref['D'],'delta_C':row['C']-ref['C'],
                  'delta_H':row['H']-ref['H'],'relative_H':row['H']/ref['H']-1,
                  'close_gate':row['C'] >= ref['C']+threshold['close_gain'],
                  'distance_gate':row['D'] <= ref['D'],
                  'accuracy_gate':row['A_defined_n'] == 3 and row['A'] >= threshold['minimum_accuracy'],
                  'hit_gain_gate':row['H'] >= ref['H']*threshold['simultaneous_hits_ratio'],
                  'hit_retention_gate':row['H'] >= ref['H']*threshold['retention_hits_ratio']}
        common = all(result[key] for key in ('close_gate','distance_gate','accuracy_gate'))
        result.update(simultaneous=common and result['hit_gain_gate'], retention=common and result['hit_retention_gate'])
        screens.append(result)
    frontiers = {}
    for metric, direction in (('D',-1),('C',1)):
        def dominates(a,b):
            x, y = direction*(a[metric]-b[metric]), a['H']-b['H']
            return x >= 0 and y >= 0 and (x > 0 or y > 0)
        frontiers[metric] = [r['condition'] for r in sorted(final,key=lambda r:r[metric])
                             if not any(dominates(s,r) for s in final)]
    for name, rows in [('comparison',comparison),('paired_differences',differences),('paired_summary',paired_groups),
                       ('screening',screens),('slice_per_seed',slices),('slice_summary',slice_groups),
                       ('action_by_pre_distance',actions),('hit_post_distance',histograms),
                       ('hit_decomposition',mechanisms)]:
        save_csv(output / f'{name}.csv', [{k:'NA' if v is None else v for k,v in row.items()} for row in rows])
    save_json(output / 'analysis_summary.json', {'comparison':comparison,'screens':screens,'frontiers':frontiers})
    return final, per_seed, screens, frontiers, actions, mechanisms, paired_groups, slice_groups


def audit(config, output):
    protocol = read_json(output / 'protocol.json')
    design = config['tradeoff']
    historical = read_json(output / 'historical_sha256_before.json')
    readme = historical.pop('README.md')
    for path, digest in historical.items():
        require(sha(ROOT / path) == digest, f'Historical file changed: {path}')
    train_pairing = {seed:{} for seed in design['training_seeds']}
    dev_pairing = {}
    tally = dict(runs=0, training_interactions=0, complete_training_windows=0, partial_training_windows=0,
                 checkpoints=0, development_windows=0, training_reward_identities=0,
                 final_windows=0, stress_windows=0, exact_checkpoint_reloads=0)
    max_residual = 0.0
    steps = list(range(0,5000001,500000))
    for condition, weights in design['conditions'].items():
        for seed in design['training_seeds']:
            directory = run_dir(output, condition, seed)
            m = read_json(directory / 'metadata.json')
            require((m['interactions'],m['full_windows'],m['partial_windows'],m['partial_ticks'],m['checkpoints'])
                    == (5000000,16666,1,200,11) and m['reload_exact'], 'Training metadata mismatch')
            rows = read_rows(directory / 'training_windows.csv')
            require(len(rows) == 16667, 'Training window count mismatch')
            stop = 0
            for index, row in enumerate(rows):
                n = int(row['ticks'])
                stop += n
                final = index == 16666
                require(int(row['end_step']) == stop and (n,row['truncated'],row['partial']) ==
                        ((200,'False','True') if final else (300,'True','False')), 'Window boundaries differ')
                terms = [float(row[k]) for k in ('base_return','distance_term','close_term','potential_term')]
                residual = abs(sum(terms) - float(row['training_return']))
                max_residual = max(max_residual,residual)
                require(residual < 1e-8 and abs(terms[1]+weights[0]*int(row['distance_sum'])) < 1e-8
                        and abs(terms[2]-weights[1]*int(row['close_ticks'])) < 1e-8
                        and abs(terms[0]-(-.05*n+10*int(row['hits'])-int(row['misses']))) < 1e-8, 'Reward accounting mismatch')
                require(train_pairing[seed].setdefault(stop,row['target_sha256']) == row['target_sha256'], 'Training target pairing mismatch')
            require(stop == 5000000, 'Training interaction mismatch')
            tally['runs'] += 1
            tally['training_interactions'] += stop
            tally['complete_training_windows'] += 16666
            tally['partial_training_windows'] += 1
            tally['training_reward_identities'] += len(rows)
            checkpoints = read_json(directory / 'development_summary.json')
            require([row['step'] for row in checkpoints] == steps, 'Checkpoint schedule differs')
            for record in checkpoints:
                step = record['step']
                path = directory / 'checkpoints' / f'step_{step:07d}.npz'
                require(sha(path) == record['checkpoint_sha256'], 'Checkpoint SHA-256 changed')
                load_model(path, config, condition, seed, step)
                tally['exact_checkpoint_reloads'] += 1
                development = read_rows(directory / 'development' / f'windows_{step:07d}.csv')
                validate_windows(development, scenario_seeds(config,'development'), dev_pairing)
                tally['checkpoints'] += 1
                tally['development_windows'] += len(development)
    per_seed = read_json(output / 'evaluation/per_seed.json')
    for split in ('final','stress'):
        pairing = {}
        for instance in (r for r in per_seed if r['split'] == split):
            condition, seed = instance['condition'], instance['seed']
            name = condition if seed is None else f'{condition}_seed{seed}'
            rows = read_rows(output / 'evaluation' / split / f'{name}_windows.csv')
            validate_windows(rows, scenario_seeds(config,split), pairing)
            numeric = [{k:float(v) for k,v in row.items() if k not in
                        ('target_sha256','hit_post_histogram','pre_distance_action_counts','slice_counts')}
                       for row in rows]
            recovered = summarize(numeric)
            require(all(math.isclose(v, instance[k], abs_tol=1e-10) if v is not None
                        else instance[k] is None for k,v in recovered.items()), f'Raw evaluation differs: {split}/{name}')
            if condition == 'rule':
                require(instance['fire_given_alignment'] == 1.0, 'Rule left a firing opportunity unused')
            tally[split+'_windows'] += len(rows)
    require((tally['runs'],tally['training_interactions'],tally['checkpoints'],tally['development_windows'],
             tally['final_windows'],tally['stress_windows']) == (21,105000000,231,46200,11000,4400), 'Study totals differ')
    return {**tally,'max_reward_identity_error':max_residual,
            'unchanged_historical_files':len(historical),'historical_readme_sha256':readme,
            'historical_results_unchanged':sum(p.startswith('results/') for p in historical),
            'training_and_evaluation_targets_paired':True,'evaluation_snapshots_unchanged':True,
            'final_only_after_selection':True,'frozen_head':protocol['git_head']}


def validate_windows(rows, seeds, pairing):
    require([int(r['scenario_seed']) for r in rows] == list(seeds), 'Missing or duplicate scenario keys')
    for row in rows:
        key = int(row['scenario_seed'])
        require(int(row['ticks']) == 300 and int(row['close_ticks']) + int(row['hit_nonclose']) <= 300
                and int(row['hits']) == int(row['aligned_shots'])
                and int(row['misses']) == int(row['unaligned_shots'])
                and int(row['shots']) == int(row['hits']) + int(row['misses']), 'Window event identity failed')
        require(pairing.setdefault(key,row['target_sha256']) == row['target_sha256'], 'Evaluation target pairing mismatch')
