"""English reporting from frozen evaluation records; no training or model selection."""
from collections import defaultdict
import json

import numpy as np

from ..tracking_research.io import ROOT,read_csv,read_json,save_csv,save_json,sha
from .analysis import METRICS,write_tables
from .core import OUT,Native,Policy,methods
from .reference import require
from .visuals import ORDER,plots,render_demo


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def number(value,digits=4):return 'NA' if value is None else f'{value:.{digits}f}'


def speed_tables():
    rows=[];records=[]
    for split in ('final','stress'):
        saved=read_csv(OUT/'evaluation'/split/'slices.csv')
        for group in ('pre_speed_1','pre_speed_2','post_speed_1','post_speed_2','boundary','interior'):
            for family in ORDER:
                rs=[r for r in saved if r['family']==family and r['group']==group]
                row=dict(split=split,family=family,group=group,instances=len(rs),
                         D=float(np.mean([float(r['D']) for r in rs])),C=float(np.mean([float(r['C']) for r in rs])),
                         hit_rate_per_300_slice_ticks=float(np.mean([float(r['hits_per_300_slice_ticks']) for r in rs])),
                         fire_given_alignment=float(np.mean([float(r['fire_given_alignment']) for r in rs if r['fire_given_alignment']!=''])))
                records.append(row)
                rows.append([split,family,group,number(row['D']),f'{100*row["C"]:.2f}%',number(row['hit_rate_per_300_slice_ticks'],2),number(row['fire_given_alignment'])])
    save_csv(OUT/'slice_comparison.csv',records)
    return rows,records


def behavior_clips(c):
    all_rows=[];selection=[];specs={m['id']:m for m in methods(c)}
    for split in ('final','stress'):
        spec=c['planning'][split]
        for e in read_json(OUT/'evaluation'/split/'behavior_examples.json')['examples']:
            selection.append(dict(split=split,**{k:v for k,v in e.items() if k!='split'}))
            if not e['found']:continue
            seed=e['training_seed'];planner_id=f'Hybrid-H3_s{seed}' if e['planner']=='Hybrid-H3' else e['planner']
            i=e['scenario_seed']-spec['start'];event=e['tick']-1
            with np.load(OUT/'evaluation'/split/f'R-Decay_s{seed}_rollouts.npz',allow_pickle=False) as z:b=z['ticks'][i];initial=z['initial'][i]
            with np.load(OUT/'evaluation'/split/f'{planner_id}_rollouts.npz',allow_pickle=False) as z:p=z['ticks'][i];pi=z['initial'][i]
            bst=np.vstack([initial,b[:-1,:6]]);pst=np.vstack([pi,p[:-1,:6]]);policy=Policy(c,specs[planner_id])
            for tick in range(max(0,event-5),min(300,event+16)):
                all_rows.append(dict(split=split,planner=e['planner'],training_seed=seed,scenario_seed=e['scenario_seed'],rule=e['rule'],
                         tick=tick+1,offset=tick-event,R_state=json.dumps(bst[tick].tolist()),planner_own_state=json.dumps(pst[tick].tolist()),
                         R_action=int(b[tick,6]),planner_own_action=int(p[tick,6]),planner_on_R_state_action=policy.act(tuple(map(int,bst[tick]))),
                         R_d_pre=int(b[tick,7]),R_d_post=int(b[tick,9]),planner_own_d_pre=int(p[tick,7]),planner_own_d_post=int(p[tick,9]),
                         R_hit=int(b[tick,11]),planner_own_hit=int(p[tick,11]),target_speed=int(b[tick,5]),target_bounce=int(b[tick,13])))
    save_csv(OUT/'behavior_fragments.csv',all_rows);save_json(OUT/'behavior_selection.json',selection)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for rid,split in enumerate(('final','stress')):
        for ax,rule in zip(axes[rid],('speed2_disagreement','bounce_alignment_recovery_differs','aligned_fire_to_move')):
            rs=[r for r in all_rows if r['split']==split and r['planner']=='Plan-H3' and r['rule']==rule]
            if rs:
                ax.plot([r['offset'] for r in rs],[r['R_d_post'] for r in rs],label='R-Decay own rollout')
                ax.plot([r['offset'] for r in rs],[r['planner_own_d_post'] for r in rs],label='Plan-H3 own rollout')
                ax.set_title(f'{split.title()}: scene {rs[0]["scenario_seed"]}, model {rs[0]["training_seed"]}',fontsize=10)
            else:ax.text(.5,.5,'No qualifying event',transform=ax.transAxes,ha='center')
            ax.axvline(0,color='#888888',lw=.8);ax.set(xlabel=rule.replace('_',' '),ylabel='Post-motion distance');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False);fig.suptitle('First events under frozen rules | descriptive paired routes; no outcome-based clip selection')
    (OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/behavior_fragments.png',dpi=170);plt.close(fig)


def report(c):
    from .study import verify_protocol,protect
    verify_protocol(c);require(read_json(OUT/'evaluation_complete.json')['complete'],'Formal evaluation incomplete')
    rows,comparison,paired,contrasts,gates=write_tables(c)
    slices,slice_records=speed_tables();behavior_clips(c);plots(c,comparison,paired)
    demos=[]
    for seed in c['planning']['demo_seeds']:
        folder=OUT/'demos'/f'scenario_{seed}'
        demos.append(read_json(folder/'decode_checks.json') if (folder/'decode_checks.json').exists() else render_demo(c,seed,folder))
    save_json(OUT/'demos/decode_checks.json',demos)
    save_json(OUT/'demos/player_check.json',dict(performed=False,reason='Programmatic decoding only, explicitly requested. No Chrome or desktop-control permission was requested.'))
    timing=[]
    for split in ('development','final','stress'):timing+=read_json(OUT/'evaluation'/split/'runtime.json')
    save_csv(OUT/'runtime.csv',timing)
    pre=read_json(OUT/'precomputation.json');evaluation=read_json(OUT/'evaluation_complete.json')
    main=[]
    for split in ('final','stress'):
        for family in ORDER:
            r=next(r for r in comparison if r['split']==split and r['family']==family)
            def cell(k,scale=1):
                val=number(r[k]*scale if r[k] is not None else None,2 if scale==100 else 4)
                return val+(f' ± {r[k+"_sd"]*scale:.4f}' if r[k+'_sd'] is not None else '')+('%' if scale==100 else '')
            main.append([split,family,r['instances'],cell('D'),cell('C',100),cell('H'),cell('A',100)])
    individual=[[r['split'],r['policy'],number(r['D']),f'{100*r["C"]:.3f}%',number(r['H'],3),
                  'NA' if r['A'] is None else f'{100*r["A"]:.2f}%',number(r['first_close_tick'],2),number(r['longest_far_streak'],2)] for r in rows]
    gt=[]
    for g in gates:
        if g['level']=='family_mean':gt.append([g['split'],g['candidate'],g['practical_pass'],g['simultaneous_pass'],g['absolute_C75'],g['absolute_C75_and_H_retained'],
                                              f'{g["C_difference"]*100:+.3f} pp',f'{g["H_ratio"]:.4f}'])
    ct=[[r['split'],r['comparison'],number(r['D']),f'{100*r["C"]:+.4f} pp',number(r['H'])] for r in contrasts if r['comparison']!='each_Q_terminal_minus_Phi']
    latency=[[r['split'],r['policy'],number(r['load_seconds']),number(r['median_ns'],1),number(r['p95_ns'],1),r['maximum_ns'],number(r['evaluation_and_storage_seconds'])]
             for r in timing if r['split'] in ('final','stress')]
    core=f'''# Short-horizon planning with the existing actions

This study reuses frozen R-Fixed and R-Decay step-5M models, seeds 60–64. New training interactions and Q updates are exactly zero. The 20 policy instances comprise Rule (1), R-Fixed (5), R-Decay (5), Plan-H1/H2/H3 with Phi terminal (3), Plan-H3-Zero (1), and Hybrid-H3 using five different frozen R-Decay Q terminals (5). Each pure planner is evaluated once per scenario, never copied into five fictitious training seeds.

## Information and task

All methods retain the ten original actions, speeds 1/2, move/fire exclusivity, pre-motion hit test, post-motion close reward, substep wall reflection and 300-tick continuing windows. Task reward is base+0.5 I(d_post≤1), gamma={c['gamma']}. Planners observe full current physical state and a known fixed stochastic motion model (turn/toggle probabilities 0.05/0.05). R policies observe relative offsets, heading and speed. Full-state knowledge and additional computation accompany the cross-family comparison; it is not a depth-only causal contrast. Only the Plan-H1→H2→H3 comparison holds representation, model and terminal fixed.

Stress changes real turn/toggle probabilities to 0.10/0.10; planning remains at 0.05/0.05. No planner receives scenario seeds, RNG states, pre-generated real future trajectories or actual future random events. Every executed action passes the same firing guard. All ten actions are evaluated in each DP backup; strictly dominated unaligned Fire is never proposed by any planned controller.

## Feedback recurrence and terminal values

V_h(s)=max_a sum_s' P(s'|s,a)[r_task+gamma V_(h−1)(s')]. Each successor has its own feedback action; only the root action is executed, then the observed new state is queried. Remaining horizon never shrinks near tick 300. Target branches enumerate no/left/right turn crossed with keep/toggle speed; duplicate complete target states are merged by summed probability.

Phi(s)=−L1 distance. The three V0 choices are Phi, zero, and max(Q_shaped)+Phi. Previous reward was task+gamma Phi(next)−Phi(current), with shaping scale one. Therefore the Q conversion has a plus Phi sign. Task+Phi terminal equals shaped+zero terminal up to root constant Phi(s); task+[max Q+Phi] equals shaped+max Q up to the same constant. Numerical/action equivalence tests passed. Planning uses float64 values and a fixed lowest-ID tie within 1e−10 of the exact maximum. R uses its original exact lowest-ID argmax; Rule preserves the original distance/displacement/action ordering.

## Main held-out results

Means ± sample SD are shown only where five distinct frozen learning models exist. Single pure controllers have no training-seed SD. A pools actual hits/shots within a policy and is NA if no shots. Metrics retain the previous definitions; all detailed seed/window records are saved.

{table(['Split','Family','Instances','D','C','H / 300 ticks','A'],main)}

## Prespecified gates relative to R-Decay

Practical: D nonincrease, C≥baseline+5pp, H≥98% baseline, A≥98%. Strong substitutes H≥102%. Absolute C≥75% is reported independently, along with hit retention. Hybrid is paired to its own model; pure methods are compared with each of five baselines and with their mean. Family mean and each-pair results are both retained in thresholds.csv.

{table(['Split','Family','Practical','Strong','C≥75%','C≥75% and H retained','C difference','H ratio'],gt)}

## Horizon and terminal contrasts

Differences are candidate minus baseline. These are deterministic empirical controller contrasts or means over the five learned terminal tables, not five independent pure-planner training runs.

{table(['Split','Contrast','D difference','C difference','H difference'],ct)}

## Complete policy-instance table

{table(['Split','Policy','D','C','H','A','First close','Longest far'],individual)}

## Speed, boundaries and behavior

Historical speed slices use actual post-toggle motion speed. New pre_speed slices use the observed speed before the action. Boundary means either object within three cells of an edge before action. Slice hit rates below normalize by 300 slice ticks, unlike H in the main table, which is per full window. All policy-trajectory slices are descriptive, not a matched-state causal decomposition. Common-state action comparisons provide a separate matched complement with distinct-state and event-frequency weights.

{table(['Split','Family','Slice','D','C','Hits / 300 slice ticks','Fire when aligned'],slices)}

The frozen −5…+15 event windows, 30-tick same-type overlap exclusion, and three-consecutive-close recovery definition are reused. Recovery times among recovered events are conditional and accompanied by censor counts. Pre-action alignment recovery after bounce is separately defined as first aligned decision within 30 ticks; no recovery is censored, not scored as zero. See evaluation/*/recovery.csv and behavior_selection.json.

Behavior clips use the earliest event under prespecified rules: speed-2 same-state disagreement; bounce with different subsequent alignment recovery; aligned R-Decay Fire versus planned movement. Absence is explicitly recorded. behavior_fragments.csv preserves every tick in each −5…+15 fragment. It distinguishes planner decisions on the R state from the planner's own evolving rollout; they must not be conflated. No clip is selected for better planner performance.

## Runtime and computation accounting

Complete DP precomputation (including saved bundles) took {pre['elapsed_seconds']:.4f} s. It performed {pre['value_backups']:,} actual state value backups, {pre['chance_transition_terms']:,} chance transition terms and {pre['action_value_lookups']:,} action-value lookups. Shared chance-expectation cache reuse was {pre['shared_expectation_cache_hits']:,} lookups; this is reuse of exact successor expectations, not sampling or deterministic target approximation. DP working arrays use {pre['working_arrays_bytes']:,} bytes and transition/movement maps {pre['transition_map_bytes']:,} bytes. Peak precomputation-process working set was {pre['memory']['bytes']/1048576:.2f} MiB (process-lifetime metric). Build time, per-pass core compute, cache size/hash and Python API first/warm timings are recorded separately.

Formal evaluation used 10,000 Final plus 4,000 Stress windows, exactly 4,200,000 environmental ticks, with 1,890,000 planned-controller table queries and no online tree expansions or value backups. Formal evaluation time (including saved complete rollouts and behavior selection) was {evaluation['elapsed_seconds']:.4f} s. Development: 4,000 windows / 1,200,000 ticks. Validation and demonstration counts are separate. A cached lookup does not erase the above startup cost.

Native step timing includes proposal plus firing guard and timer overhead; excludes environment stepping, Python orchestration, loading and precomputation. Python API first-pass and warm latency is separately provided in diagnostics/api_latency.csv; the first pass is not OS-cache flushed. Neither timing measure should be sold as end-to-end cold planning latency.

{table(['Split','Policy','Load seconds','Median ns','P95 ns','Maximum ns','Eval/storage seconds'],latency)}

## Artifacts and validation

- [Measured interpretation](conclusions.md), [paper addendum](paper_addendum.md), [reproduction](REPRODUCE.md).
- [Horizon response](figures/horizon_response.png), [terminal comparison](figures/terminal_values.png), [D/C/H](figures/D_C_H.png), [paired Hybrid](figures/paired_hybrid.png), [dynamic events](figures/dynamic_events.png), [behavior fragments](figures/behavior_fragments.png).
- [Demo 6400000](demos/scenario_6400000/synchronized.gif), [demo 6400001](demos/scenario_6400001/synchronized.gif), [demo 6400002](demos/scenario_6400002/synchronized.gif).
- Every GIF contains initial state plus all 300 ticks. All 301 pixel arrays and timing entries are decoded/compared; total duration is 20,070 ms. Actual player inspection is deliberately not performed, as requested; no browser/desktop permission was requested.
- Source/model/cache/scenario freeze is in protocol.json. Q and counts are read-only and protected inputs are hashed. Historical receipt writes are redirected to this study; all previous uncommitted work remains unchanged. Tests and verification commands record actual exits/timings in execution/. No commit or push is performed.
'''
    (OUT/'results_summary.md').write_text(core,encoding='utf-8')
    interpretations(c,rows,comparison,gates,contrasts,slice_records)
    addendum='''# Paper addendum: finite-horizon stochastic feedback planning

This zero-training study separates horizon from terminal value inside a full-state, known-model planner family. A complete physical-state finite-horizon dynamic program evaluates all ten original actions and merges identical stochastic successor states. Shared successor expectations avoid rebuilding a random tree per tick while preserving the feedback Bellman recurrence. Cached policies are indexed solely by current physical observation; model probabilities remain 0.05 under the real 0.10 Stress process.

The terminal conversion follows shaped reward = task reward + gamma Phi(next) - Phi(current). A task value equals shaped value plus Phi; accordingly the learned terminal is max(Q_shaped)+Phi, with no fitted coefficient. An explicit shaping/terminal equivalence test preserves root action rankings up to the root-state constant. The complete precomputation, numerical precision, tie rule and cache identities are frozen before Final access.

The same-information H1/H2/H3 contrasts test horizon. H3 versus Zero tests distance heuristic; five Hybrid instances test five distinct frozen terminal estimates and are compared to both Phi planning and their same-seed R-Decay policies. Planners additionally observe full coordinates and use a known motion model, so their differences from relative-state R cannot be assigned exclusively to search depth. Pure planners have one instance, not five independent training seeds. Scenario differences are paired; ticks are not treated as independent statistical replicates. No confidence intervals or significance claims are made.

All prespecified methods are retained, including any poor zero-terminal or hit/close tradeoff controller. The 75% close target is separated from the joint relative gate and hit retention. Speed-before-decision is separated from actual speed-after-toggle; event and trajectory slices are descriptive, with common-state action diagnostics as a matched complement. Complete event clips follow frozen earliest-event rules, not an outcome-based highlight selection. See conclusions.md for the measured results and implications; a negative horizon or terminal result does not establish an optimality bound for the action set.
'''
    (OUT/'paper_addendum.md').write_text(addendum,encoding='utf-8')
    reproduce='''# Reproduce and verify the planning study

Run from repository root with requirements-reproduction.txt installed. Runtime and compiler versions are frozen in protocol.json and native/build.json. Use the existing ten step-5M models; never rerun previous training. No code in this study performs a Q update.

Verify saved records (ordinary and optimized Python), without replaying formal evaluation:

```powershell
python -m aimrl.tracking_planning verify
python -O -m aimrl.tracking_planning verify
```

Fresh replication: choose an unused directory, establish a new protection inventory for all tracked historical paths plus the complete previous uncommitted source/models/results, then run the following. The one-time inventory bootstrap is documented in workspace.json/protected_sha256_before.json; the default delivered directory must not be overwritten. The exact replication intentionally reuses this study reservation. Setup audits every protected seed declaration/CSV.

```powershell
$env:AIMRL_PLANNING_OUTPUT = 'results/tracking_planning_replication'
python -m aimrl.tracking_planning initialize
python -m aimrl.tracking_planning setup --compiler <path-to-zig.exe>
python -m aimrl.tracking_planning development
python scripts/check_tracking_planning.py preflight --legacy-kernel <path-to-published-tradeoff.dll>
python -m aimrl.tracking_planning freeze
python -m aimrl.tracking_planning evaluate
python -m aimrl.tracking_planning report
python scripts/check_tracking_planning.py postflight --legacy-kernel <path-to-published-tradeoff.dll>
python -m aimrl.tracking_planning inventory
python -m aimrl.tracking_planning verify
python -O -m aimrl.tracking_planning verify
```

Setup compiles without fast-math or floating-point contraction, precomputes all 21 horizon layers across seven terminal families, and records genuine loop counters, memory and wall time. Development completes all 20×200 windows and a full 301-frame layout test before protocol freeze. Freeze requires all preflight commands, records source/config/models/caches/methods/seed ranges and retains all methods. Final and Stress use 20×500 and 20×200 independent environment windows. C uses frozen policy tables only; it cannot supply a scenario seed or true future to any policy query.

Formal rollouts and native decision timings are retained in evaluation/*/*_rollouts.npz. Report and verify reuse these records; no formal rollout is needed for recomputation. Cached Bellman arrays are checked by rebuilding mathematical DP sweeps (planning computations, not training interactions) and comparing every saved value/policy/expectation. Existing historical checkers write only fresh receipts inside this study. The previous canonical boundary verifier is wrapped solely to redirect its two receipt writes; previous expected hashes and sources remain unchanged.

Manifest excludes its own file and explicit verification/postflight receipts to avoid circular digests. Source hashes, model hashes, DP cache hashes and protected-input hashes are always checked separately. No exclusion may be used to hide a changed frozen input. GIF decoder checks all pixels/durations and sample frame hashes. Actual-player inspection is intentionally not performed by this protocol.
'''
    (OUT/'REPRODUCE.md').write_text(reproduce,encoding='utf-8')
    save_json(OUT/'historical_integrity.json',protect())
    save_json(OUT/'execution_accounting.json',dict(training_interactions=0,Q_updates=0,formal_environment_ticks=4200000,
              development_environment_ticks=1200000,formal_planning_lookups=1890000,development_planning_lookups=540000,
              demonstration_environment_ticks=3600,demonstration_planning_lookups=1800,layout_test_environment_ticks=1200,
              layout_test_planning_lookups=600,precomputed_value_backups=pre['value_backups'],precomputed_transition_terms=pre['chance_transition_terms'],
              validation='Separate checks.json files record their actual original/native environment-step counts. Existing unit-test toy training is validation, never study training.'))
    save_json(OUT/'report_complete.json',dict(complete=True,formal_environment_replays=0,new_training_interactions=0))


def interpretations(c,rows,comparison,gates,contrasts,slices):
    mean={(r['split'],r['family']):r for r in comparison}
    lines=['# Measured conclusions','',
       'All results below come from the saved held-out records. No method, reward, terminal coefficient or model was selected after Final. Pure planners are single controllers; five Hybrid rows refer to five different frozen terminal Q tables.','']
    for split in ('final','stress'):
        base=mean[split,'R-Decay'];lines+=['## '+split.title(),'']
        for family in ('Plan-H1','Plan-H2','Plan-H3','Plan-H3-Zero','Hybrid-H3'):
            r=mean[split,family];g=next(g for g in gates if g['split']==split and g['level']=='family_mean' and g['candidate']==family)
            lines.append(f'- {family}: D={r["D"]:.4f}, C={100*r["C"]:.3f}%, H={r["H"]:.3f}, A={"NA" if r["A"] is None else format(100*r["A"],".2f")+"%"}; '
                         f'versus R-Decay mean, C {100*(r["C"]-base["C"]):+.3f}pp, H ratio {r["H"]/base["H"]:.4f}. '
                         f'Practical={g["practical_pass"]}, strong={g["simultaneous_pass"]}, absolute C75={g["absolute_C75"]}, C75 with retained hits={g["absolute_C75_and_H_retained"]}.')
        for a,b in (('Plan-H2','Plan-H1'),('Plan-H3','Plan-H2'),('Plan-H3','Plan-H3-Zero'),('Hybrid-H3','Plan-H3')):
            ra,rb=mean[split,a],mean[split,b]
            lines.append(f'- {a} minus {b}: D {ra["D"]-rb["D"]:+.5f}, C {100*(ra["C"]-rb["C"]):+.4f}pp, H {ra["H"]-rb["H"]:+.4f}.')
        hybrid=[r for r in rows if r['split']==split and r['family']=='Hybrid-H3']
        paired=[]
        for r in hybrid:
            b=next(b for b in rows if b['split']==split and b['family']=='R-Decay' and b['seed']==r['seed'])
            paired.append((r['seed'],r['D']-b['D'],r['C']-b['C'],r['H']-b['H']))
        lines.append(f'- Hybrid improves D in {sum(x[1]<0 for x in paired)}/5 models, C in {sum(x[2]>0 for x in paired)}/5, H in {sum(x[3]>0 for x in paired)}/5. Paired values (seed, D, C fraction, H): `{json.dumps(paired)}`.')
        for group in ('pre_speed_2','post_speed_2','boundary','interior'):
            b=next(r for r in slices if r['split']==split and r['family']=='R-Decay' and r['group']==group)
            for family in ('Plan-H3','Hybrid-H3'):
                r=next(r for r in slices if r['split']==split and r['family']==family and r['group']==group)
                lines.append(f'- Descriptive {group}, {family} versus R-Decay: D {r["D"]-b["D"]:+.4f}, C {100*(r["C"]-b["C"]):+.3f}pp, '
                             f'hits/300 slice ticks {r["hit_rate_per_300_slice_ticks"]:.3f} versus {b["hit_rate_per_300_slice_ticks"]:.3f}.')
        lines.append('')
    lines+=['## Interpretation and limits','',
      'Read the horizon and terminal contrasts separately. H1/H2/H3 share the same full state, fixed stochastic model, tie rule and Phi terminal, isolating the chosen finite-horizon change. H3-Zero can be poor far from the target because a short horizon sees no differentiating near-term reward; this is an acquisition/terminal issue, not proof that planning itself fails. Hybrid tests the value of a learned terminal on top of the same H3 information/model. Its effect is not a new training effect.',
      '',
      'Any observed gain over R-Decay establishes empirical room within the original action set for those measured objectives, but combines full-state access, model knowledge and extra offline computation. A C gain with a large H loss is not joint improvement. Stress is a deliberate model-mismatch test: no planner receives the true 0.10 transition parameters. No result is a theoretical upper bound.',
      '',
      'Behavior evidence is in behavior_selection.json and behavior_fragments.csv. The frozen first-event rules may select an unfavorable planner clip and explicitly record an absent event. Common-state actions distinguish a decision difference from on-policy distribution changes. Speed before decision and speed after actual toggle are separate measurements; post-toggle slice hit rates are descriptive, not treatment effects.',
      '',
      'The next useful action-design experiment is a controlled 2-cell movement factor: speed-two targets and the existing 1/3-cell cursor moves create a coordination problem that finite-horizon planning can partly manage but does not remove. First preserve this planning controller as a reference, then add 2-cell moves in a separate environment intervention with the same scenarios and report both C and H. Do not claim a movement mismatch is proven as the sole remaining limit. A longer planning horizon or better calibrated learned terminal is another clean follow-up if the horizon/terminal contrasts warrant it; eligibility traces are not specifically supported by this zero-training experiment.',
      '',
      'Computation is part of the comparison: the full physical DP sweeps and cache generation are charged before reporting O(1) online queries. There is no hidden free search. Original model counters and hashes remain unchanged. Programmatic GIF decoding is complete; actual player inspection is intentionally not executed, per request.']
    (OUT/'conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
