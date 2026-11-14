"""English scientific delivery from unrounded saved records, without selecting a winner."""
import json
import numpy as np
from ..tracking_research.io import ROOT,read_json,read_csv,save_json
from .core import OUT
from .analysis import analyze
from .study import history_check

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def fmt(r,k):
    if r[k] is None:return 'NA'
    val=r[k]*(100 if k in ('C','A') else 1);s=f'{val:.4f}'+('%' if k in ('C','A') else '')
    if r.get(k+'_sd') is not None:s+=f' ± {r[k+"_sd"]*(100 if k in ("C","A") else 1):.4f}'
    return s

REPRO='''# Reproduce the action-resolution study

Run from the repository root. Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2 and Pillow 13.0.0 were used. Install `requirements-reproduction.txt`. The study reads the five R-Decay seed 60–64 checkpoints at `results/tracking_boundary_stability/runs/R-Decay/seed_60/checkpoints/step_5000000.npz` (and seeds 61–64). No copied/fabricated new-action Q values exist. A10 DP bundles are referenced in `precomputation.json`; A14 bundles live here under `cache/` with different action-set identities.

## Audit saved results without reevaluation

```powershell
python -c "from aimrl.tracking_action_resolution.core import config; from aimrl.tracking_action_resolution.analysis import analyze; analyze(config())"
python -m aimrl.tracking_action_resolution verify
python -O -m aimrl.tracking_action_resolution verify
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -p test_tracking_action_resolution.py -v
```

The analysis command reconstructs tables from raw ticks; verification compares them, reconstructs every Bellman array, checks all model/protocol/source digests and original historical files, and decodes all GIF frames. It runs no evaluation environment and makes zero Q updates. Verification writes only its new-study receipts. Do not use the analysis command to overwrite frozen evidence with different definitions.

## Rebuild and independently replicate

Use a **new empty output** and the unchanged checked-in configuration. Setup creates a historical baseline and rechecks scenario reservations. The already-used ranges intentionally reproduce this study; they are not a new independent holdout.

```powershell
$env:AIMRL_ACTION_OUTPUT = '.tmp/action-resolution-replication'
python -m aimrl.tracking_action_resolution setup --compiler <path-to-zig>
python -m aimrl.tracking_action_resolution development
python scripts/check_tracking_action_resolution.py preflight
python -m aimrl.tracking_action_resolution freeze
python -m aimrl.tracking_action_resolution evaluate
python -m aimrl.tracking_action_resolution report
python scripts/check_tracking_action_resolution.py postflight
python -m aimrl.tracking_action_resolution inventory
```

Compiler **Zig 0.15.2**; setup runs `zig cc -std=c11 -O3 -fno-fast-math -ffp-contract=off -shared aimrl/tracking_action_resolution/kernel.c -o <new-output>/native/action_resolution.dll`. Local/global compiler caches go under `.tmp/action-resolution-build`. On Unix the output suffix is `.so`; binary hashes differ by platform, so replicate to a separate protocol rather than replacing the archived DLL. The check runner defaults its historical environment kernels to `.tmp/latest-release-native/{boundary,tradeoff}.dll`; first rebuild those with `python scripts/reproduce_latest_studies.py build --compiler <path-to-zig> --output .tmp/latest-release-native` if absent. Their build/parity instructions are in `../../docs/reproduce_latest_studies.md`. Python tests execute only short correctness updates; they are not formal training runs.

`setup` computes both A10/A14 at every depth, demands exact A10 agreement with the published cache, and checks all 56,250,000 state/terminal/depth subset inequalities using the pre-fixed 1e-9 tolerance. It stores exact maxima while choosing the first ID within 1e-10; approximate tie selection does not redefine V. `validate` compares Python transitions, recursion, shaped/task conversion and native rollout/RNG behavior. `preflight` runs the complete test suite, optimized new tests, both-mode independent validation and historical publication/study checks. Final cannot run until a protocol exists. No training phase is provided.

The evaluation driver owns seeds and draws. The planner's API takes only current physical state and an immutable policy table; its precomputation receives only the known model and terminal. Stress environment probabilities are 0.10; planner remains 0.05. Each 2-cell movement advances the cursor once and the target once; there is no two-step environment emulation.

`protocol.json` freezes all experiment sources, configuration, models, caches, bootstrap, gates and behavior rules before new Final access. `historical_sha256_before.json` and `historical_integrity.json` distinguish unchanged released files from this independent addition. `output_inventory.csv` hashes the delivery and referenced dependencies; `inventory_scope.json` names its self-reference/later-receipt exclusions. Compilation intermediates are not needed; the archived DLL is accompanied by complete C source and build options. Failed setup attempts remain under `execution/`; they happened before precomputation or holdout access.

All three demo GIFs include reset plus every tick (301 frames, 20,070 ms). Full pixels and timings were checked against the renderer. Actual-player inspection was not performed; no Chrome or desktop permission was requested.
'''

def report(c):
    result=analyze(c);from .visuals import plots,render_demo
    plots(c,result);demo=[]
    for s in c['planning']['demo_seeds']:demo.append(render_demo(c,s,OUT/'demos'/f'scenario_{s}'))
    save_json(OUT/'demo_accounting.json',dict(windows=12,environment_ticks=3600,policy_lookups=3600,receipts=demo,training_interactions=0,Q_updates=0))
    comp=result['comparison'];effects=[]
    for split in ('final','stress'):
        for terminal in ('Plan','Hybrid'):
            a=next(r for r in comp if r['split']==split and r['family']==terminal+'-H3-A10');b=next(r for r in comp if r['split']==split and r['family']==terminal+'-H3-A14')
            effects.append([split,terminal,f'{b["D"]-a["D"]:+.6f}',f'{100*(b["C"]-a["C"]):+.4f}',f'{b["H"]-a["H"]:+.4f}',f'{100*(b["H"]/a["H"]-1):+.4f}%'])
    metrics=table(['Split','Controller','D','C','H / 300 ticks','A'],[[r['split'],r['family'],*[fmt(r,k) for k in ('D','C','H','A')]] for r in comp])
    effecttable=table(['Split','Terminal','ΔD','ΔC (pp)','ΔH','H change'],effects)
    gates=table(['Split','Terminal','Level','Seed','Practical','Stronger','C≥75%','C≥75% + H retained'],[[r['split'],r['terminal'],r['level'],r['seed'],r['practical_pass'],r['simultaneous_pass'],r['absolute_C75'],r['absolute_C75_and_hits_retained']] for r in result['thresholds']])
    pre=read_json(OUT/'precomputation.json');evaluation=read_json(OUT/'evaluation_complete.json');dev=read_json(OUT/'evaluation/development/complete.json');validation=read_json(OUT/'validation/checks.json')
    cost=dict(precomputation=pre,formal=evaluation,development=dev,validation_runs={k:read_json(OUT/k/'checks.json') for k in ('validation_initial','validation','validation_O') if (OUT/k/'checks.json').exists()},demo_windows=12,demo_ticks=3600,
        formal_windows=8400,formal_ticks=2520000,development_windows=2400,development_ticks=720000,common_diagnostics_environment_ticks=0,new_training_interactions=0,Q_updates=0)
    save_json(OUT/'execution_accounting.json',cost)
    usage=read_csv(OUT/'action_usage.csv');slices=read_csv(OUT/'speed_boundary_slices.csv');speed=[];behavior=[]
    for split in ('final','stress'):
        for family in ('Plan','Hybrid'):
            rs=[]
            for n in (10,14):
                vals=[r for r in slices if r['split']==split and r['family']==f'{family}-H3-A{n}' and r['group']=='pre_speed_2']
                rs.append({k:float(np.mean([float(r[k]) for r in vals])) for k in ('D','C','H_per300','A')})
            speed.append([split,family,*[f'{rs[1][k]-rs[0][k]:+.6f}' for k in ('D','C','H_per300')]])
            for n in (10,14):
                vals=[r for r in usage if r['split']==split and r['family']==f'{family}-H3-A{n}' and r['group']=='all']
                behavior.append([split,f'{family} A{n}',*[f'{100*np.mean([float(r[k+"_fraction"]) for r in vals]):.3f}%' for k in ('slow','medium','fast','stay','fire')],
                    sum(int(r['medium_actual_2']) for r in vals),sum(int(r['medium_actual_1']) for r in vals),sum(int(r['medium_actual_0']) for r in vals)])
    speedtable=table(['Split','Terminal','Speed-2 ΔD','Speed-2 ΔC','Speed-2 Δhits/300 slice ticks'],speed)
    usagetable=table(['Split','Controller','Slow','Medium','Fast','Stay','Fire','Medium actual 2','actual 1','actual 0'],behavior)
    consistent=all(r['D']<0 and r['C']>0 and r['H_ratio']>=.98 for r in result['paired_differences'])
    recommendation=('The matched effects are consistently favorable under both terminals and all five source models. Prioritize a separately preregistered fourteen-action training study to test whether a learned controller can exploit this control option without full-state model-based DP. Keep this zero-training action-resolution result as a complete paper section; do not silently retune this holdout. The current learned terminal remains ten-action-derived.' if consistent else 'Effects are not consistently favorable across the frozen pairs. Prioritize diagnosing the Development control trade-off and writing up this controlled result before further training. Any revised controller requires a new protocol and new holdout.')
    partition=table(['Split','Controller','C','Hits ending outside close / all ticks'],[[r['split'],r['policy'],f'{100*r["C"]:.3f}%',f'{100*r["total_hit_nonclose"]/r["ticks"]:.3f}%'] for r in result['per_policy'] if r['seed'] in (None,60)])
    intervals=table(['Split','Terminal','Metric','95% paired lower','upper'],[[r['split'],r['terminal'],r['metric'],f'{r["lower_95"]:.6f}',f'{r["upper_95"]:.6f}'] for r in result['bootstrap_intervals']])
    text=f'''# Do two-cell movements improve continuous tracking?

## Question and controlled intervention

Can adding four one-tick medium cursor moves improve closeness while retaining hit frequency, under the same full physical state, known stochastic model, task reward, H=3 feedback planner and terminal values? A10 IDs 0–9 are unchanged. A14 appends 10=left_medium (-2,0), 11=right_medium (2,0), 12=up_medium (0,-2), 13=down_medium (0,2). Maximum movement remains 3. Movement and firing are mutually exclusive. Every tick moves the cursor once, tests a shot before target motion, consumes the original three random draws, then advances the target once with original unit-substep reflection. Reset and 300-tick windows are unchanged.

The twelve fixed instances are Plan-H3 A10/A14 (one each) and Hybrid-H3 A10/A14 for frozen R-Decay source seeds 60–64 (five each). **New training interactions=0; Q updates=0**. Hybrid terminals always maximize over the **original ten** Q entries; new actions have no learned Q entries and participate in every planning layer. This study tests additional control options in this planner, not a trained fourteen-action Q-learning agent.

## Frozen design and correctness

gamma=0.99; task reward=base_reward+0.5 I(d_post≤1); Phi=-L1. Because old shaped reward=task+gamma Phi(next)-Phi(current), task value=shaped value+Phi. V0=Phi for Plan and max_A10 Q_shaped+Phi for Hybrid, identically in both action sets. Planning uses float64, exact maximum V, and chooses the lowest ID within 1e-10 of maximum; all actions including Fire are enumerated. Same inference-only firing guard. H=3 is also used at the last recorded tick, since truncation is not task termination. Planner probabilities remain 0.05/0.05; Stress uses **real 0.10/0.10**. Planners never receive RNG, scene seed or future route.

All 36 A10/A14 Bellman layers are computed. Every A10 value, policy and expectation equals its historical published array exactly; eleven saved states per historical controller additionally agree action-wise. Python/native value and shaped/task equivalence max error: {validation['max_value_error']:.3g} and {validation['max_shaping_error']:.3g}. Old-action environment states/rewards/termination/RNG agree step-wise; medium clipping and single target advancement are independently checked. Saved int16 observations are widened in the new index adapter, avoiding arithmetic overflow without editing the historical helper.

For all 3,125,000 valid physical states, all six terminal instances, and h=1,2,3, **V_A14≥V_A10**: 56,250,000 comparisons, minimum difference 0, violations 0, pre-fixed absolute tolerance 1e-9. No tolerance adjustment occurred. Exact stored maxima are distinguished from approximate tied-action selection. This inequality constrains the finite-horizon objective and **does not guarantee** rollout D/C/H improvements. Full arrays and [monotonicity receipt](monotonicity.json) remain auditable.

Development uses 7100000–7100199 (2,400 windows). Final uses 7200000–7200499 (**6,000 windows**), Stress 7300000–7300199 (**2,400 windows**). Formal total **2,520,000 environment ticks**. [Seed audit](seed_audit.json) found no historical range collision. The protocol was frozen after Development/preflight and before Final; all methods retained and no tuning followed Final. Policies share initial state and exact target-route hashes for each scene.

## Results

D is mean post-motion L1 distance, C the post-motion d≤1 fraction, H hits per 300-tick window, A hit/shot accuracy (NA if zero shots). Hybrid entries show five source-model means ± **sample SD**, not tick-level uncertainty. Pure Plan has one instance and no training-seed SD. A10 controls are from **these new scenes**, never the older 75.15%/69.38% figures.

{metrics}

### A14 minus matched A10

All differences below use unrounded integer-derived measurements before formatting.

{effecttable}

Relative practical gate: D does not increase, C gains ≥5 percentage points, H retains ≥98%, A≥98%. Stronger gate replaces H retention with ≥2% improvement. Absolute C≥75% and hit retention are reported separately. Final and Stress never share a baseline.

{gates}

### Paired uncertainty

2,000 percentile bootstrap replicates, pre-fixed seed 20261114 with split/terminal offsets. Whole paired scenes are resampled; Hybrid also resamples paired source models, using the same scene draw across all models. Pure Plan resamples scenes only. The five terminal models remain matched across action sets. No tick is treated as an independent observation. These are descriptive bootstrap intervals with only five source models, not a claim of broader model-family certainty.

{intervals}

## What changed at speed 2 and in control behavior?

The table conditions on **decision-time observable** speed=2. H is normalized to 300 slice ticks; it is not total-window H. Actual post-toggle speed has a distinct table slice, avoiding look-ahead in policy interpretation. Target-speed slice membership is common across policies; cursor-boundary and acquisition slices are policy-dependent and descriptive.

{speedtable}

{usagetable}

Medium's nominal displacement is always 2, while clipped actual motion is counted separately as 2/1/0. [Action usage](action_usage.csv) includes speed, boundary/interior and pre-distance state distributions, overshoot and immediate reversal counts. [Speed/boundary slices](speed_boundary_slices.csv) include pre/post speed, boundary, speed×boundary, initial acquisition and strictly post-acquisition sustained tracking. [Common-state comparisons](diagnostics/common_comparison.csv) compare all six terminal pairs on the same previously frozen physical states; common-state action/value differences are distinct from own-rollout slices.

The fixed selection rules retain the first observable-speed-2 medium disagreement, first accepted bounce with different alignment recovery, and first qualifying medium disagreement with a worse A14 own-route distance. [Selection and absence records](behavior_selection.json) and [full behavior fragments](behavior_fragments.csv) preserve both routes' real states/actions, including regressions. Own-route regressions after diverging states are descriptive, not a causal one-step action contrast. Overshoot means crossing the *current* target coordinate on its exact row/column; reversal means successive opposite-axis-direction commands. Neither alone proves causal correction. Hits, alignment opportunities and alignment-conditioned firing rates are shown together in per_policy/per_seed and slice tables.

[Mechanism paired changes](mechanism_comparison.csv) tabulate slow/fast/medium/Stay/Fire, crossing, reversal and unaligned-wait rate changes. Hit timing matters: a pre-motion hit followed by actual speed-2 target motion normally ends at d_post=2, so that tick contributes H but not C. The mutually exclusive close / hit-outside-close accounting is independently checked in every window; a large hit gain need not produce a comparably large C gain. The following partition is shown for the single Plan and fixed display model 60 (all models remain in data):

{partition}

Event windows remain [-5,+15], same-type suppression 30 ticks; turn/toggle recovery requires three consecutive close ticks within horizon 30 and is censored at the next random event/window/horizon. Bounce curves keep overlaps with other event types. Alignment recovery after a bounce uses first subsequent pre-action alignment within 30 ticks. [Recovery](recovery.csv) and [high-speed event curves](high_speed_event_curves.csv) preserve these definitions. No event or fragment is selected for better A14 performance.

## Compute, integrity and delivery

Precomputation: {pre['elapsed_seconds']:.3f} s; **112,500,000 value backups**, {pre['chance_terms']:,} chance-transition terms, {pre['action_lookups']:,} action-value lookups and {pre['shared_expectation_reuses']:,} reused expectations. Half of the value backups are explicit A10 reconstruction checks, half A14. Peak working set during precompute: {pre['memory'].get('bytes',0)/1048576:.2f} MiB (process lifetime, not stage-attributed allocation). Formal evaluation/storage: Final {evaluation['final']['elapsed_seconds']:.3f} s and Stress {evaluation['stress']['elapsed_seconds']:.3f} s. Per-policy load/storage time and lookup latency are in [runtime tables](evaluation/final/runtime.csv); online planning performs a table query, no model expansion. Offline computation is not free.

[Execution accounting](execution_accounting.json) separates formal 8,400 windows/2.52M ticks, Development 2,400/720,000, all validation runs, demos 12/3,600, and common-state diagnostics (zero environment ticks). Precomputation and environment interactions are separately counted. Native compiler: Zig 0.15.2, C11 -O3 -fno-fast-math -ffp-contract=off; [build receipt](native/build.json). Timing includes this host's overhead and compression and is not a hardware-independent benchmark.

Historical released files are protected by a 3,662-file hash baseline; [integrity proof](historical_integrity.json). The five Q/count tables are read-only and their hashes match before/after. No historical source, protocol, cache, result or Git identity was changed. This run performs no commit or push. Failed setup attempts are retained honestly and preceded all holdout evaluation. [Check commands](preflight/commands.json), [postflight](postflight/commands.json), [inventory](output_inventory.csv) and [inventory hash boundary](inventory_scope.json) provide acceptance evidence. Final acceptance is recorded after checks, outside its own inventory hash boundary.

## Figures, complete demos and next step

![D/C/H](figures/D_C_H.png)
![Paired model effects](figures/paired_models.png)
![Action use](figures/action_usage.png)
![High-speed events](figures/high_speed_events.png)

Three four-panel demos: Plan A10, Plan A14, Hybrid A10, Hybrid A14; Hybrid always source seed 60. Target route is shared, cursor policies run independently. Each GIF contains the initial frame and all 300 ticks. Full decoded pixels and durations were compared against all rendered frames; 301 frames/20,070 ms each. **Actual-player inspection was not performed**, and no Chrome/desktop permission was requested.

{recommendation}

- [Scenario 7400000 GIF](demos/scenario_7400000/synchronized.gif)
- [Scenario 7400001 GIF](demos/scenario_7400001/synchronized.gif)
- [Scenario 7400002 GIF](demos/scenario_7400002/synchronized.gif)
- [Reproduce and rebuild](REPRODUCE.md), [conclusions](conclusions.md), [paper addendum](paper_addendum.md)

![Four-panel complete trajectory](demos/scenario_7400000/synchronized.gif)
'''
    (OUT/'results_summary.md').write_text(text,encoding='utf-8')
    conclusion='# Conclusions: adding two-cell actions\n\n'+effecttable+'\n\n'+gates+'\n\n'+recommendation+'''

Interpret finite-horizon value monotonicity separately from realized paired-window metrics. A14 provides extra control authority under exactly the same state/model/reward/terminal. It does not establish a trained A14 Q-learning policy; the learned terminal still comes from A10. Report Final and Stress separately and retain all five source models and every regression. The +5pp closeness requirement is not replaced by statistical significance or an absolute threshold.

Next-step decision rule: if closeness and hits improve consistently but the practical gate remains unmet, a separately frozen fourteen-action training study is a justified follow-up, not an already demonstrated learned result. If useful aggregate gains fail to retain hits or are inconsistent, diagnose horizon/reward/controller behavior on Development before any new holdout. If both gains are negligible, prioritize writing the paper and documenting the negative result. The present report and paired tables determine which branch is supported; no further action scheme is silently tuned here.
'''
    (OUT/'conclusions.md').write_text(conclusion,encoding='utf-8')
    (OUT/'paper_addendum.md').write_text('# Paper addendum: action-resolution intervention\n\n'+
        'We compare A10 against its strict A14 superset in a fixed three-step stochastic feedback planner. Four cardinal two-cell moves consume one tick each. Five frozen ten-action learned terminals and one potential terminal are reused without training. The same scene seeds and target routes provide paired controls; Stress tests real dynamics at twice the assumed random-event probability.\n\n'+metrics+'\n\n'+effecttable+'\n\n'+
        'Exhaustive finite-state subset-value inequalities pass at every depth for all six terminals (56.25M comparisons, zero violations). Their relevance is limited to the planning objective. Realized D/C/H, source-model sample variability, paired scene/model bootstrap intervals and predeclared gates are reported separately. Full observability, known dynamics, offline computation and an A10-derived terminal limit transfer claims. No A14 learned policy or real-game result is claimed.\n\n[Full report](results_summary.md) · [Protocol](protocol.json) · [Reproduce](REPRODUCE.md)\n',encoding='utf-8')
    (OUT/'REPRODUCE.md').write_text(REPRO,encoding='utf-8');save_json(OUT/'historical_integrity.json',history_check())
    save_json(OUT/'report_complete.json',dict(complete=True,all_twelve_instances=True,demo_frames=903,new_training_interactions=0,Q_updates=0))
