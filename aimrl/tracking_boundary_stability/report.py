"""Recompute research tables from frozen artifacts and write English reports."""
from collections import defaultdict
import json

import numpy as np

from ..tracking_research.io import ROOT, read_csv, read_json, save_csv, save_json, sha
from .analysis import aggregate, contrasts, thresholds
from .core import CONDITIONS, OUT, Kernel, require


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def report(c):
    from .study import LIBRARY, check_protocol, history_check
    from .visuals import demos,plots
    check_protocol(c)
    per_seed=read_json(OUT/'per_seed.json')
    comparison=aggregate(per_seed);paired,factors=contrasts(per_seed);gates=thresholds(c,per_seed,comparison)
    save_json(OUT/'comparison.json',comparison);save_csv(OUT/'comparison.csv',comparison)
    save_csv(OUT/'paired_differences.csv',paired);save_csv(OUT/'factor_effects.csv',factors)
    save_json(OUT/'thresholds.json',gates);save_csv(OUT/'thresholds.csv',gates)
    diag=[]
    for condition in CONDITIONS:
        for seed in c['study']['training_seeds']:
            for r in read_csv(OUT/'runs'/condition/f'seed_{seed}/q_diagnostics.csv'):
                if int(r['step'])==5000000:diag.append(dict(condition=condition,seed=seed,**{k:v for k,v in r.items() if k!='condition'}))
    save_csv(OUT/'q_diagnostics_per_seed.csv',diag)
    grouped=defaultdict(list)
    for r in diag:grouped[r['condition'],r['group'],r['weighting']].append(r)
    qsummary=[]
    qkeys=('q_gap_mean','wrong_ranking_fraction','greedy_unaligned_fire_fraction','mean_fire_updates','mean_stay_updates',
           'fire_unvisited_fraction','stay_unvisited_fraction','stay_bellman_mae','fire_bellman_mae','q_change_mae','greedy_change_fraction')
    for (condition,group,weighting),rows in grouped.items():
        r=dict(condition=condition,group=group,weighting=weighting)
        for key in qkeys:
            vals=[float(x[key]) for x in rows if x[key]!='']
            r[key]=float(np.mean(vals)) if vals else None
            r[key+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        qsummary.append(r)
    save_csv(OUT/'q_diagnostics_summary.csv',qsummary);save_json(OUT/'q_diagnostics_summary.json',qsummary)
    # Distribution bins use the same fixed physical states; repeated events are
    # weights, not additional independent states or training replicates.
    distributions=[];changes=[]
    bins=np.array([-np.inf,-20,-10,-5,-2,-1,-.5,0,.5,1,2,5,10,20,np.inf])
    for condition in CONDITIONS:
        for seed in c['study']['training_seeds']:
            root=OUT/'runs'/condition/f'seed_{seed}'
            states=read_csv(root/'diagnostic_states_final.csv')
            gaps=np.array([float(r['q_gap']) for r in states]);frequencies=np.array([int(r['frequency']) for r in states])
            for weighting,weights in (('state',np.ones(len(states))),('event',frequencies)):
                hist=np.histogram(gaps,bins=bins,weights=weights)[0]
                for i,n in enumerate(hist):distributions.append(dict(condition=condition,seed=seed,weighting=weighting,
                    low=str(bins[i]),high=str(bins[i+1]),weight=float(n),fraction=float(n/weights.sum()),
                    interval='[low,high), except final bin includes high'))
            previous={}
            for r in read_csv(root/'q_diagnostics.csv'):
                key=r['group'],r['weighting']
                if key in previous:
                    old=previous[key]
                    changes.append(dict(condition=condition,seed=seed,group=key[0],weighting=key[1],
                      previous_step=int(old['step']),step=int(r['step']),
                      **{k+'_change':float(r[k])-float(old[k]) for k in ('q_gap_mean','wrong_ranking_fraction','fire_bellman_mae','stay_bellman_mae')}))
                previous[key]=r
    save_csv(OUT/'q_gap_distributions.csv',distributions)
    save_csv(OUT/'diagnostic_checkpoint_changes.csv',changes)
    training=read_json(OUT/'training_complete.json');pilot=read_json(OUT/'pilot_complete.json')
    resources=[dict(stage='formal',condition=r['condition'],seed=r['seed'],interactions=r['interactions'],updates=r['updates'],
                    seconds=r['elapsed_seconds'],process_peak_bytes=r['memory']['bytes'],alpha_min=r['alpha']['minimum'],
                    alpha_mean=r['alpha']['mean'],alpha_max=r['alpha']['maximum']) for r in training['runs']]
    resources+=[dict(stage='pilot',condition=r['condition'],seed=r['seed'],interactions=r['interactions'],updates=r['updates'],
                    seconds=r['elapsed_seconds'],process_peak_bytes=r['memory']['bytes'],alpha_min=r['alpha']['minimum'],
                    alpha_mean=r['alpha']['mean'],alpha_max=r['alpha']['maximum']) for r in pilot['runs']]
    save_csv(OUT/'resources.csv',resources)
    plots(c,comparison)
    if not (OUT/'demos/decode_checks.json').exists():demos(c,Kernel(LIBRARY,c))
    table=[]
    for r in comparison:
        if r['variant']!='guard':continue
        table.append([r['split'].title(),r['condition'],f'{r["D"]:.3f} ± {r["D_sd"]:.3f}',
                      f'{100*r["C"]:.2f} ± {100*r["C_sd"]:.2f}%',f'{r["H"]:.2f} ± {r["H_sd"]:.2f}',
                      'NA' if r['A'] is None else f'{100*r["A"]:.2f}%',f'{r["first_close_tick"]:.2f}',f'{r["longest_far_streak"]:.2f}'])
    gate_table=[]
    for r in gates:
        if r['level']=='mean':
            counts=[x for x in gates if x['level']=='seed' and x['condition']==r['condition'] and x['split']==r['split']]
            gate_table.append([r['split'],r['condition'],r['practical_pass'],sum(x['practical_pass'] for x in counts),
                               r['simultaneous_pass'],sum(x['simultaneous_pass'] for x in counts),r['absolute_C75_pass'],
                               sum(x['absolute_C75_pass'] for x in counts)])
    effects=[]
    for r in factors:
        if r['variant']=='guard' and r['metric'] in ('D','C','H'):
            effects.append([r['split'],r['effect'],r['metric'],f'{r["mean"]:.5f}',f'{r["sample_sd"]:.5f}',r['paired_values']])
    qtable=[]
    for r in qsummary:
        if r['group']=='all':qtable.append([r['condition'],r['weighting'],f'{100*r["wrong_ranking_fraction"]:.2f}%',
               f'{100*r["greedy_unaligned_fire_fraction"]:.2f}%',f'{r["fire_bellman_mae"]:.3f}',f'{r["stay_bellman_mae"]:.3f}',
               f'{r["mean_fire_updates"]:.1f}',f'{r["mean_stay_updates"]:.1f}',f'{100*r["fire_unvisited_fraction"]:.1f}%'])
    observations=[]
    for split in ('final','stress'):
        for cond in CONDITIONS[1:]:
            rs=[r for r in paired if r['split']==split and r['condition']==cond and r['variant']=='guard']
            observations.append(f'- {split.title()} {cond}: C improved in {sum(r["C"]>0 for r in rs)}/5 seeds; '
                                f'D decreased in {sum(r["D"]<0 for r in rs)}/5; H increased in {sum(r["H"]>0 for r in rs)}/5. '
                                'The complete seed deltas are in paired_differences.csv.')
    summary=f'''# Boundary information and update stability

This is the prespecified 2×2 study using newly trained R-Fixed as the paired control. All four conditions retain close_0p5 reward, existing target dynamics, ten actions and the published exploration schedule. Relative state already included heading and speed. Boundary grouping is the only representation change; it remains an approximate aggregation.

## Execution

Completed 20 independent training runs, seeds 60–64 paired across four conditions, exactly 100,000,000 real interactions and updates. Each model starts from zero and uses the 5M checkpoint for every main result. Formal execution wall time: {training['elapsed_seconds']:.2f} seconds; sum of per-run times: {sum(r['elapsed_seconds'] for r in training['runs']):.2f} seconds. Pilot: 4,000,000 separate interactions, seed 1060, with no Final access and no model selection. Peak formal-process working set: {max(r['memory']['bytes'] for r in training['runs'])/1048576:.2f} MiB (process-lifetime high-water mark, not a stage delta). See resources.csv for all runs and actual per-update learning-rate histograms in each metadata.json.

R has 19,208 rows and 192,080 state-action pairs; B has 1,089,288 rows and 10,892,880 pairs (56.71× as many rows). Each B Q/count array is 43,571,520 bytes; combined 83.11 MiB. Each representation uses a 2,500-byte physical-axis index map. Valid axis keys were checked and all 3,125,000 physical states were range-checked. The compact axis product contains only valid physical feature keys.

## Held-out guarded performance

Values are means ± sample SD across five independently trained seeds. D and C use post-motion distance; H is hits per 300 ticks; A is actual executed-shot accuracy, averaged over defined seed accuracies. Zero-shot accuracy is NA. Acquisition is the first post-motion distance≤1 tick, or 300 if never acquired. Far streak means consecutive post-motion distance>5 ticks.

{md_table(['Split','Condition','D','C','H','A','First close','Longest far'],table)}

## Prespecified gates

Practical: D does not increase, C gains at least 5 percentage points, H retains at least 98%, A≥98%, relative to the same-split paired R-Fixed. Simultaneous substitutes H≥102%. Mean and seed gates are distinct; absolute C≥75% is separate. Counts are out of five seeds. R-Fixed cannot pass its own +5pp comparative gate by construction.

{md_table(['Split','Condition','Practical mean','Seeds','Simultaneous mean','Seeds','C≥75% mean','Seeds'],gate_table)}

## Factor effects and seed consistency

Effects below are signed raw differences; negative D and positive C/H are improvements. SD is across paired seed differences. Interaction = (B-Decay−B-Fixed)−(R-Decay−R-Fixed). No tick or scenario is treated as an independent training replicate. No confidence intervals or significance claims are made with five training seeds.

{md_table(['Split','Effect','Metric','Mean','Sample SD','Paired values, seeds 60–64'],effects)}

{chr(10).join(observations)}

## Mechanism and Q diagnostics

With no stochastic turn or toggle, interior (10,10,11,10,right,2) and wall (23,10,24,10,right,2) have identical R row 9993, but Stay yields relative (3,0,right,2) versus (−1,0,left,2). B assigns different rows. This proves boundary aliasing, not that any particular tabular boundary model must improve at the same budget.

128 paired unaligned Fire/Stay checks share physical next state and target RNG. Distance/close/potential terms coincide; Stay training reward and same-frozen-table one-step TD target exceed Fire by exactly one, within floating-point tolerance. This is an update-target identity, not a convergence assumption about a learned Q table.

Common-state diagnostics below use a fixed set sampled before pilot from Rule and uniform-random trajectories, independently of the learned policies. State weighting gives one vote per distinct physical state; event weighting uses its frequency in that fixed sample. Q values and Bellman residuals are computed without guard. Bellman residuals use the exact conditional expectation over six turn/toggle outcomes, not one noisy sampled target. They remain physical-state residuals for aggregated tables; irreducible aliasing can contribute.

{md_table(['Condition','Weighting','Fire>Stay','Greedy invalid Fire','Fire residual MAE','Stay residual MAE','Fire updates','Stay updates','Fire unvisited'],qtable)}

Boundary/interior, speed 1/2 and pre-action-distance groups are in q_diagnostics_per_seed.csv. Per-physical-state final records are saved for each run. Full checkpoint distributions and Q changes are in runs/*/seed_*/q_diagnostics.csv. Quantiles there are explicitly distinct-state quantiles; event-weighted means and fractions use actual repeated sampling frequencies. q_own_trajectory.csv reports the same ranking questions on each policy's own visited states, including frequency≥5 states. Those differing visitation distributions are descriptive and cannot establish a representation effect by themselves. A reduced error fraction can also reflect unvisited, zero-tied pairs; count and unvisited columns must be read with the error rates.

## Dynamics and shooting

boundary_speed_slices.csv retains historical boundary (either object within three cells of an edge before action) and post-motion target-speed slices. event_curves.csv uses −5…+15 ticks around bounce, turn and toggle events. Same-type events within 30 ticks of an accepted event are excluded; other-type overlap is retained for curves. Recovery requires three consecutive distance≤1 post-motion ticks, starts at the first of those ticks, and is censored by the next random turn/toggle, 30-tick horizon or window end. recovery.csv reports recovery fractions and conditional times among recovered events; censored events must not be read as successful fast recoveries. All are descriptive policy-specific comparisons. Alignment opportunities and firing conditional on alignment are included in per_seed.csv.

All raw/guard pairs preserve the full physical trajectory, distance, close ticks and hit count. Every discarded unaligned Fire raises base return by one. Native execution was additionally checked against two independent original Python environments and the existing FiringGuard on 80 windows. Main comparisons use guard to isolate tracking; raw invalid firing remains in per_seed.csv. Guard never participates in training or bootstrap.

## Artifacts and verification

- [Paired seed figure](figures/paired_seeds.png), [learning curves](figures/learning_curves.png), [event figure](figures/boundary_dynamic_events.png), [D/C/H tradeoffs](figures/D_C_H_tradeoff.png), [Q stability](figures/q_stability.png).
- [Demo 5400000](demos/scenario_5400000.gif), [demo 5400001](demos/scenario_5400001.gif), [demo 5400002](demos/scenario_5400002.gif). Four guarded panels share each target route and use training seed 60. Each includes the initial frame plus all 300 ticks, with all 301 decoded pixel arrays and durations checked. Actual-player observations are separately recorded in demos/player_check.json when performed.
- [Protocol](protocol.json), [reproduction](REPRODUCE.md), [addendum](paper_addendum.md), [resource log](resources.csv), [evaluation checks](evaluation/checks.json).
- Tests, commands, actual exit codes and timing are in preflight.json, postflight.json and execution/. Verification reuses saved models; it does not repeat the 100M training.

The original tracked files and model hashes are preserved; historical_integrity.json records the final comparison. No commit or push was performed. Inventory hashing excludes its own manifest and explicitly mutable validation receipts to avoid circular hashes.
'''
    (OUT/'results_summary.md').write_text(summary,encoding='utf-8')
    addendum='''# Paper addendum: boundary information × update stability

## Experimental question and design

We test whether omitted boundary context and persistent Q-update noise limit continuous tracking. The published relative encoding already observes relative position, target direction and speed. We cross its boundary augmentation with the prespecified per-pair learning rate alpha(n)=0.15(1+n/1000)^(-0.6), using n before the actual update. The augmentation preserves the original four features and adds each object's x/y boundary groups {0},{1},{2},{3},{4…20},{21},{22},{23},{24}. It is not fully Markov. The 56.71-fold row increase changes sample density and is part of this fixed-budget treatment.

All four cells use identical reward, exploration, target mechanism, training-seed pairing, reset distribution and 5M budget. Environment and exploration RNGs are separate. Counts are incremented once per genuine update. Q and increments are float32 with double-precision reward/target/alpha calculation. Continuing bootstrap includes truncation and budget tails. Python and native per-step records agree on all actions, states, reward parts, alpha, targets, updated Q and counts. Formal main endpoints are fixed at 5M.

## Evidence and interpretation

The boundary counterexample establishes a real alias: identical observed R state can have different deterministic next relative states near a wall. The Fire/Stay controlled experiment establishes a one-unit expected one-step-target disadvantage for every unaligned Fire, given the same frozen next-state Q. Persistent learned action misranking therefore is compatible with finite sampling, asynchronous updates, aggregation and noise; it does not imply the environment secretly rewards empty fire.

The factorial contrasts, exact results and all prespecified gates are reported in results_summary.md and accompanying tables. An effect of decay supports an update-stability explanation only to the extent that paired endpoint and checkpoint diagnostics improve. A null or negative result does not disprove boundary aliasing. With the much larger table, lower raw ranking-error rates must be checked against unvisited fractions, Bellman residuals and tracking performance. The rate schedule is intentionally unchanged despite differing visit frequencies.

Five paired training seeds, rather than 500 scenarios or individual ticks, are the training replication units. We report their sample SD, worst seed and all paired deltas without declaring statistical significance. Event slices condition on policy-induced trajectories and are descriptive. The common-state set provides a matched complement. Historical frozen three-seed models are not substituted for the new five-seed R-Fixed control. No historical reference models were newly evaluated in this study; that optional comparison is omitted.

## Subsequent research choice

Action timing creates a structural tension: valid Fire occurs before motion, consumes the tick and makes no tracking move. Speed-two target motion then places every successful shot outside distance≤1 unless reflection reduces net displacement. D/C/H tradeoffs should therefore be considered alongside any gain from state or update changes. A next experiment should compare a prespecified planning controller against the same action set, and then isolate action redesign (for example move-and-fire or changed timing) as a separate environment intervention. Eligibility traces would address temporal credit assignment, but these one-step identities and the shooting/motion conflict alone are not evidence that delayed credit is the leading bottleneck. Final evidence-specific recommendations are in the conclusion supplement.
'''
    (OUT/'paper_addendum.md').write_text(addendum,encoding='utf-8')
    repro='''# Reproduction and frozen-artifact verification

Use Python with requirements-reproduction.txt installed. The recorded runtime versions are in protocol.json; this execution used the existing workspace virtual environment and Zig 0.15.2. The C kernel accepts all learning/reward/probability parameters explicitly. Compile without fast-math or floating-point contraction. Grid 25 and ten actions are validated fixed research constraints.

From the repository root, verify saved artifacts without retraining:

```powershell
python -m aimrl.tracking_boundary_stability verify
python -O -m aimrl.tracking_boundary_stability verify
python scripts/run_boundary_stability_checks.py postflight --legacy-kernel <path-to-frozen-tradeoff.dll>
```

For a fresh full replication choose an unused output directory; never overwrite the published result directory. Reuse the frozen seed reservation as an exact replication, not as a new independent experiment:

```powershell
$env:AIMRL_BOUNDARY_OUTPUT = 'results/tracking_boundary_stability_replication'
python -m aimrl.tracking_boundary_stability setup --compiler <path-to-zig.exe>
python scripts/run_boundary_stability_checks.py preflight --legacy-kernel <path-to-frozen-tradeoff.dll>
python -m aimrl.tracking_boundary_stability pilot
python -m aimrl.tracking_boundary_stability train
python -m aimrl.tracking_boundary_stability evaluate
python -m aimrl.tracking_boundary_stability report
python scripts/run_boundary_stability_checks.py postflight --legacy-kernel <path-to-frozen-tradeoff.dll>
```

Setup records historical hashes, seed-collision audit, build command and exact-backend gates. Pilot uses four 1M runs only. Train requires passed preflight and pilot, freezes protocol before formal execution and runs all 20 × 5M updates. It does not resume partially created runs or silently reduce budgets. Checkpoints at 0 and every 500k are preserved; only 5M is used for primary evaluation. Development has 200 scenarios at each checkpoint. Final/Stress are inaccessible until training_complete.json exists. Their ranges, demo seeds, event windows, recovery rule, diagnostics and thresholds are fixed in the config/protocol.

Evaluation saves all guarded full rollouts plus original actions and per-window actual-action metrics. Report recomputes all tables/figures from saved artifacts, then generates fixed demos. Complete GIF decoding is automated; actual-player inspection is a separate manual/UI observation in player_check.json. Do not claim actual playback from decoding alone. Verify catches malformed counts, changed hashes, missing frames and mismatched aggregate metrics using runtime exceptions that remain active under python -O.

Manifest excludes itself and explicitly listed validation receipts; those contain manifest-verification results and cannot be included in their own digest. Update inventory only after reviewable reporting changes, never to hide a changed frozen source/model. Historical hashes and the protocol hash must remain fixed.
'''
    (OUT/'REPRODUCE.md').write_text(repro,encoding='utf-8')
    save_json(OUT/'historical_integrity.json',history_check())
    save_json(OUT/'report_complete.json',dict(complete=True,training_interactions_repeated=0,
              reports=['results_summary.md','paper_addendum.md','REPRODUCE.md']))


if __name__=='__main__':
    from .core import config
    report(config())
