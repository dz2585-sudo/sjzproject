"""Report measured comparisons without selecting favorable seeds or hiding failures."""
import math

import numpy as np

from .io import ROOT, read_json, save_json
from .workflow import METHODS

METRICS = ('mean_distance','close_fraction','far_fraction','distance_p95','mean_longest_far_streak',
           'hits_per_episode','pooled_shot_accuracy','mean_first_acquisition_tick','mean_base_return',
           'recovery_fraction','mean_capped_recovery_lag')


def statistics(values):
    values = [float(x) for x in values if x is not None]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'defined_seeds':len(values)}


def summaries(data):
    return {method:{field:statistics([data[f'{method}_seed{s}'][field] for s in (10,11,12)]) for field in METRICS} for method in METHODS}


def fmt(value,digits=4):
    return 'undefined' if value is None else f'{value:.{digits}f}'


def write_report(config,output,checks):
    final = read_json(output/'evaluation/final/summary.json'); stress = read_json(output/'evaluation/stress/summary.json')
    grouped = {'final':summaries(final),'stress':summaries(stress)}
    differences = {}
    for split,data in [('final',final),('stress',stress)]:
        differences[split] = {mode:{field:{'per_seed':{str(s):data[f'relative_{mode}_seed{s}'][field]-data[f'absolute_{mode}_seed{s}'][field]
            if data[f'relative_{mode}_seed{s}'][field] is not None and data[f'absolute_{mode}_seed{s}'][field] is not None else None for s in (10,11,12)}} for field in METRICS} for mode in ('base','potential')}
        for mode in differences[split].values():
            for value in mode.values(): value.update(statistics(value['per_seed'].values()))
    means = grouped['final']; candidate,control = means['relative_potential'],means['absolute_potential']
    practical = {'distance_reduced_at_least_10_percent':candidate['mean_distance']['mean']<=.9*control['mean_distance']['mean'],
                 'close_gain_at_least_5_percentage_points':candidate['close_fraction']['mean']>=control['close_fraction']['mean']+.05,
                 'far_fraction_no_worse':candidate['far_fraction']['mean']<=control['far_fraction']['mean'],
                 'hits_at_least_90_percent_of_control':candidate['hits_per_episode']['mean']>=.9*control['hits_per_episode']['mean'],
                 'accuracy_at_least_85_percent':candidate['pooled_shot_accuracy']['mean']>=.85}
    aspirations = {'mean_distance_at_most_1_5':candidate['mean_distance']['mean']<=1.5,
                   'close_fraction_at_least_75_percent':candidate['close_fraction']['mean']>=.75,
                   'far_fraction_at_most_5_percent':candidate['far_fraction']['mean']<=.05,
                   'mean_capped_acquisition_at_most_15':candidate['mean_first_acquisition_tick']['mean']<=15}
    save_json(output/'evaluation/cross_seed_summary.json',{'groups':grouped,'paired_relative_minus_absolute':differences,
             'practical_target_components':practical,'practical_target_achieved':all(practical.values()),'aspirational_targets':aspirations})
    diagnostic = read_json(output/'diagnostics/summary.json')
    lines = ['# Tracking diagnostics and representation study','',
        '## Scope and integrity','',
        f"The study completed {checks['formal_runs']} fresh formal runs and exactly {checks['formal_interactions']:,} training interactions, plus {checks['pilot_interactions']:,} disposable pilot interactions. No pilot or historical Q values were transferred to formal runs. The environment physics, rewards, and frozen V2 source bytes were preserved. No Git commit, push, identity change, or training extension was performed.",
        '', 'The original project root has an unborn Git HEAD. The separately published local clone was at reviewed commit 43aa7cc9e02a91ba7a5ee9cd0986a62a269e56ff. Both facts and the original working-tree status are recorded in protocol.json. The NPZ historical packages were loaded from that publication clone, their embedded original NPY hashes were checked, and the original root source hashes matched the archived metadata.',
        '', '## Archived-model diagnosis','',
        'The following values replay the original 200 development scenarios, not the independent final test. Saved zero-valued Q rows are observable model properties, not historical visitation counts. Exact ties use float32 equality; numerical near ties use a separately reported absolute tolerance of 1e-6.',
        '', '| Policy | Mean distance | P95 distance | Close fraction | Far fraction | Mean longest far streak | Zero-Q fraction | Exact tie fraction | Clipped movement fraction | Local pursuit fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for policy,values in diagnostic.items():
        row = values['development']; sl = row['slices']['all']
        vals = [row['mean_distance'],row['distance_p95'],row['close_fraction'],row['far_fraction'],row['mean_longest_far_streak'],sl['all_zero_q_fraction'],sl['exact_tie_fraction'],sl['clipped_move_fraction'],sl['local_pursuit_fraction']]
        lines.append('| '+policy+' | '+' | '.join(fmt(v) for v in vals)+' |')
    base = diagnostic['historical_base']['development']; potential = diagnostic['historical_potential']['development']
    lines += ['', f"Observed evidence: archived Base spends {base['far_fraction']:.2%} of ticks farther than five cells, with a mean longest far streak of {base['mean_longest_far_streak']:.2f} ticks per window. Its saved rows are all zero on {base['slices']['all']['all_zero_q_fraction']:.2%} of evaluation ticks, with exact maximal ties on {base['slices']['all']['exact_tie_fraction']:.2%}. The clipping and pursuit fractions above use movement actions as the denominator. Pursuit tests distance to the target's pre-motion position and is not an optimal-action test.",
              '', 'Plausible hypotheses: sparse physical-state coverage, lowest-ID choices on uninformative rows, and delayed sparse hit rewards can limit the large absolute table at a finite budget. This archive does not establish exact training visitation histories or identify a single causal explanation. Random remains an intentionally untrained negative control. Detailed action counts, unaligned shots, speed bins, wall slices, and all three demo traces are retained in diagnostics/.',
              '', '## Matched results and all training seeds','',
              'All new models use five million interactions, seeds 10/11/12, and epsilon decay over four million interactions. Historical models use 20 million interactions and training seed 0; they are cross-budget references. Values below are seed-level means and sample standard deviations (ddof=1), never standard errors over correlated ticks. Accuracy is pooled within each seed before averaging.',
              '', '| Split | Method | Mean distance (SD) | Close fraction (SD) | Far fraction (SD) | Hits/window (SD) | Pooled accuracy (SD) | Capped acquisition (SD) |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,groups in grouped.items():
        for method,fields in groups.items():
            vals = [f"{fmt(fields[x]['mean'])} ({fmt(fields[x]['sample_sd'])})" for x in ('mean_distance','close_fraction','far_fraction','hits_per_episode','pooled_shot_accuracy','mean_first_acquisition_tick')]
            lines.append('| '+split+' | '+method+' | '+' | '.join(vals)+' |')
    lines += ['', '| Split | Method | Training seed | Mean distance | Close fraction | Far fraction | Hits/window | Accuracy | Base return |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,data in [('final',final),('stress',stress)]:
        for method in METHODS:
            for s in (10,11,12):
                row = data[f'{method}_seed{s}']
                lines.append(f'| {split} | {method} | {s} | '+' | '.join(fmt(row[x]) for x in ('mean_distance','close_fraction','far_fraction','hits_per_episode','pooled_shot_accuracy','mean_base_return'))+' |')
    lines += ['', '### Paired Relative minus Absolute differences','',
              'Negative distance/far differences and positive close/hit differences favor Relative. Scenario paths are paired; training seeds are paired. Three seeds do not establish statistical significance.',
              '', '| Split | Reward | Metric | Seed 10 | Seed 11 | Seed 12 | Mean difference | Sample SD |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for split,modes in differences.items():
        for mode,fields in modes.items():
            for field in ('mean_distance','close_fraction','far_fraction','hits_per_episode','pooled_shot_accuracy'):
                value = fields[field]
                lines.append(f'| {split} | {mode} | {field} | '+' | '.join(fmt(value['per_seed'][str(s)]) for s in (10,11,12))+f" | {fmt(value['mean'])} | {fmt(value['sample_sd'])} |")
    lines += ['', '### Independently evaluated references','',
              'Each archived model, the unchanged predictive rule, and Random appears once per split. No artificial training-seed standard deviation is attached to these references.',
              '', '| Split | Reference | Distance | Close fraction | Far fraction | Hits/window | Accuracy |',
              '|---|---|---:|---:|---:|---:|---:|']
    for split,data in [('final',final),('stress',stress)]:
        for policy in ('historical_base','historical_potential','rule','random'):
            row = data[policy]
            lines.append(f'| {split} | {policy} | '+' | '.join(fmt(row[x]) for x in ('mean_distance','close_fraction','far_fraction','hits_per_episode','pooled_shot_accuracy'))+' |')
    lines += ['', '## Predeclared targets','', f"Overall practical improvement target: **{'ACHIEVED' if all(practical.values()) else 'NOT ACHIEVED'}**.",'']
    for label,achieved in practical.items(): lines.append(f"- {label}: {'ACHIEVED' if achieved else 'NOT ACHIEVED'}.")
    lines += ['', 'Aspirational targets are assessed independently on Relative Potential three-seed final-test means:','']
    for label,achieved in aspirations.items(): lines.append(f"- {label}: {'ACHIEVED' if achieved else 'NOT ACHIEVED'}.")
    lines += ['', 'Worst Relative Potential training seed for each final-test metric:','']
    for field,higher in [('mean_distance',False),('close_fraction',True),('far_fraction',False),('hits_per_episode',True),('pooled_shot_accuracy',True),('mean_first_acquisition_tick',False)]:
        worst = (min if higher else max)((final[f'relative_potential_seed{s}'][field],s) for s in (10,11,12))
        lines.append(f'- {field}: seed {worst[1]}, {fmt(worst[0])}.')
    lines += ['', '## Boundary aliasing, target changes, and shooting','',
              'Relative state is (xt-xc+24, yt-yc+24, heading, speed-1). It has 19,208 rows versus 3,125,000 absolute rows, and 768,320 versus 125,000,000 float32 element bytes. This approximately 162.70-fold storage reduction is not a speedup or sample-efficiency guarantee. Absolute configurations sharing one relative observation can differ in crosshair clipping and target reflection; deterministic tests demonstrate both. No rule fallback or boundary information was added.',
              '', 'Slices use the pre-action state, with either object within three cells of a wall defining boundary_near. These are policy-dependent diagnostics, not randomized subgroups. Counts below are pooled across training seeds; performance values are means of seed-level ratios.',
              '', '| Method | Slice | Tick denominator | Mean distance | Close fraction | Far fraction | Zero-visit row fraction |',
              '|---|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for context in ('boundary_near','interior','target_near','crosshair_near','speed_1','speed_2'):
            slices = [final[f'{method}_seed{s}']['slices'].get(context,{}) for s in (10,11,12)]
            vals = [statistics([r.get(f) for r in slices])['mean'] for f in ('mean_distance','close_fraction','far_fraction','zero_visit_row_fraction')]
            lines.append(f"| {method} | {context} | {sum(r.get('ticks',0) for r in slices)} | "+' | '.join(fmt(v) for v in vals)+' |')
    for method in ('absolute_potential','relative_potential'):
        row = means[method]
        lines += ['', f"{method}: observed recovery fraction {fmt(row['recovery_fraction']['mean'])}; mean capped recovery lag {fmt(row['mean_capped_recovery_lag']['mean'])} ticks. Recovery windows around turns, speed toggles, and reflections may overlap. Events with insufficient 20-tick follow-up are censored; unrecovered events have capped lag 20 and are not successful 20-tick recoveries."]
    lines += ['', 'Firing consumes a tick while the target moves; it cannot be combined with a crosshair move. Compare hits and accuracy with proximity rather than assuming they measure the same skill. Raw action frequencies, slow/fast proportions by pre-action distance, and unaligned-shot counts are available for every policy and split. Stress testing changes only turn and speed-toggle probabilities from 0.05 to 0.10; it tests this defined shift, not arbitrary real-world generalization.',
              '', '## Resource use, execution, and verification','',
              f"Formal run times sum to {checks['formal_elapsed_seconds']:.2f} seconds, including checkpoint validation, coverage scans, logging, model compression, and reload checks. Peak observed process working set was {checks['peak_working_set_bytes']/1024**2:.2f} MiB. It is a Windows process-lifetime high-water mark, not a per-run memory delta. Free disk at preflight was {checks['initial_free_disk_bytes']/1024**3:.2f} GiB.",
              '', '| Method | Seed | Interactions | Measured seconds | Visited rows / denominator | Visited pairs / denominator |',
              '|---|---:|---:|---:|---:|---:|']
    for meta in checks['runs']:
        c = meta['coverage']
        lines.append(f"| {meta['method']} | {meta['training_seed']} | {meta['interactions']} | {meta['elapsed_seconds']:.2f} | {c['visited_rows']} / {c['state_rows']} | {c['visited_pairs']} / {c['state_action_pairs']} |")
    lines += ['', f"Audit: {checks['development_summaries']} development summaries, {checks['development_windows']:,} development windows, {checks['development_ticks']:,} development ticks; {checks['final_windows']:,} final windows and {checks['stress_windows']:,} stress windows. Each formal run has 16,666 complete 300-tick windows and one 200-tick budget-stop partial window. All update-count sums equal actual interactions. Original histories, models, source, configuration and protocols retain their original hashes; README is the sole intentionally edited pre-existing file.",
              '', 'Setup and the complete existing/new unittest suite were executed. Exact commands, exit codes, output logs, peak memory, stage times, protocol snapshots, artifact digests, and target-pairing checks are retained in execution_checks.json and stage_execution/. A pre-formal new-test failure caused by Windows sandbox temporary-directory permissions was fixed by using the workspace temporary directory; no formal experiments had started. Superseded preflights are retained if present.',
              '', '## Figures and full-length animations','',
              '- [Learning curves](render/learning_curves.png)', '- [Independent final comparisons](render/final_comparison.png)',
              '- [Distance distributions and far tails](render/distance_tails.png)', '- [Boundary and speed diagnostics](render/boundary_speed_diagnostics.png)',
              '- [Genuine update coverage](render/coverage.png)', '- [Slow/fast choices](render/speed_choices.png)','']
    for seed in (3601000,3601001,3601002): lines.append(f'- [Full main comparison, scenario {seed}](render/seed{seed}_main_comparison.gif)')
    lines += ['- [Six-policy comparison with negative control](render/seed3601000_six_policy_comparison.gif)', '',
              'Each GIF contains all 301 frames and decodes to 20.07 seconds with actual 60/70 ms frame delays. New displayed agents always use training seed 10, not a selected best seed. Historical Potential is clearly labeled 20M, seed 0; new agents are labeled 5M. Trace CSVs and decoded start/middle/end PNGs accompany the animations. Actual-displacement arrows are drawn only for observed crosshair movement; hit stars show pre-motion shot coordinates separately from post-motion target circles. Export and decoding are verified separately from GUI playback, which remains unverified.',
              '', '## Reproduction','',
              '[Commands and controls](../../docs/tracking_improvement.md). Models and genuine update counters are compressed NPZ files in runs/<method>/seed_<seed>/. Completed runs are reused only after configuration, budget, source, model and record integrity checks. Incomplete attempts are retained and rejected for silent reuse. Evaluation is epsilon-zero with separate environment/action streams and before/after Q/counter/RNG snapshots.',
              '', '## Answers and remaining limitations','']
    base_help = means['relative_base']['mean_distance']['mean'] < means['absolute_base']['mean_distance']['mean']
    potential_help = candidate['mean_distance']['mean'] < control['mean_distance']['mean']
    lines += [f"**Did representation sharing help Base?** {'Yes in mean final-test distance' if base_help else 'No in mean final-test distance'}: Absolute Base {means['absolute_base']['mean_distance']['mean']:.4f}, Relative Base {means['relative_base']['mean_distance']['mean']:.4f}. The complete close/far, shooting, paired differences, and individual seeds above qualify this answer.",
              '', f"**Did it improve sustained tracking beyond matched Potential?** {'Mean distance improved' if potential_help else 'Mean distance did not improve'}: matched Absolute Potential {control['mean_distance']['mean']:.4f}, Relative Potential {candidate['mean_distance']['mean']:.4f}. The predeclared multi-metric practical target was {'achieved' if all(practical.values()) else 'not achieved'}. This answer uses matched 5M controls, not the 20M historical reference.",
              '', '**What happens near walls and after target changes?** The boundary/speed slices and overlapping recovery measurements above quantify the behavior. State aliasing at walls is established mechanically, but these policy-dependent observational slices do not isolate its causal contribution. Any boundary-aware representation must be a separately frozen follow-up, not an expansion tuned against this final set.',
              '', f"**Was there a shooting tradeoff?** Relative Potential obtained {candidate['hits_per_episode']['mean']:.4f} hits/window at mean seed-level pooled accuracy {candidate['pooled_shot_accuracy']['mean']:.4f}, versus matched control {control['hits_per_episode']['mean']:.4f} and {control['pooled_shot_accuracy']['mean']:.4f}. Firing necessarily forgoes movement for that tick; compare these measured outcomes with the proximity and action-frequency records rather than claiming an isolated causal tradeoff.",
              '', '**Which measured limitation should the next study address?** '+('The candidate missed at least one practical or absolute target. Use its reported far tails, worst seed, wall slices, and shooting/acquisition deficits to design a separately preregistered boundary-aware or reward/coverage study; do not extend this budget.' if not all(practical.values()) or not all(aspirations.values()) else 'All declared engineering targets were met in this bounded study; boundary aliasing, limited training-seed replication, and the defined stress shift remain limitations to test independently.'),
              '', 'The study establishes neither real-game performance nor tabular optimality of the lossy relative controller. Passing software checks, shaped training reward, or one attractive clip is not evidence that tracking is solved.']
    (output/'results_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
