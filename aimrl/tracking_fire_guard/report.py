"""Generate English inference-study evidence from the held-out replays."""
import math
from pathlib import Path
import subprocess

from ..tracking_research.io import ROOT, read_json, save_json, sha
from .analysis import compile_metrics, diagnostics
from .history import PRIOR, agrees, rows, verify_history
from .replay import ratio, require
from .study import checked
from .visuals import demo, figures


def fmt(value,precision=3,scale=1):
    return 'NA' if value is None else f'{value*scale:.{precision}f}'


def evaluation_audit(output, protocol, within):
    zero={'final':{'original':0,'guard':0},'stress':{'original':0,'guard':0}}
    window_count=0
    for split,(start,n) in protocol['holdout'].items():
        for m in protocol['models']:
            c,s=m['condition'],m['training_seed']
            a=rows(output/'evaluation'/split/f'{c}_seed{s}_original_windows.csv')
            b=rows(output/'evaluation'/split/f'{c}_seed{s}_guard_windows.csv')
            require(len(a)==len(b)==n,'Variant window count mismatch')
            for index,(r,g) in enumerate(zip(a,b)):
                require(int(r['scenario_seed'])==int(g['scenario_seed'])==start+index,'Variant scenario key differs')
                require(all(r[k]==g[k] for k in ('ticks','distance_sum','close_ticks','hits','target_sha256')),
                        'Variant metrics differ in stored evidence')
                require(int(r['shots'])-int(g['shots'])==int(r['misses']) and int(g['misses'])==0
                        and math.isclose(float(g['base_return'])-float(r['base_return']),int(r['misses']),abs_tol=1e-9),
                        'Variant shot or reward accounting differs in stored evidence')
                for variant,row in (('original',r),('guard',g)):
                    hits,shots=int(row['hits']),int(row['shots'])
                    agrees(ratio(hits,shots),row['accuracy'],'stored window accuracy')
                    zero[split][variant]+=shots==0
                window_count+=1
    require(window_count==14700 and len(within)==42,'Paired matrix totals differ')
    return {'learned_paired_windows_rechecked':window_count,'paired_seed_contrasts':len(within),
            'zero_shot_windows':zero,'zero_shot_accuracy':'NA',
            'exact_invariants':['D','C','H','physical state','target trajectory'],
            'training_interactions':0}


def results_table(summary,split):
    original=[r for r in summary if r['split']==split and r['variant']=='original']
    guard={r['condition']:r for r in summary if r['split']==split and r['variant']=='guard'}
    lines=['| Condition | D | C (%) | H / window | A mean, original → guard (%) | A pooled, original → guard (%) | Misses, original → guard | Base return, original → guard |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def estimate(row,key,p,scale=1):
        value=fmt(row[key],p,scale)
        return value if row[key+'_sd'] is None else f'{value} ± {fmt(row[key+"_sd"],p,scale)}'
    for r in original:
        c=r['condition']
        g=guard.get(c)
        def contrast(k,p,scale=1,sd=False):
            show=lambda x:estimate(x,k,p,scale) if sd else fmt(x[k],p,scale)
            return show(r) if g is None else f'{show(r)} → {show(g)}'
        lines.append(f"| {c} | {estimate(r,'D',3)} | {estimate(r,'C',2,100)} | {estimate(r,'H',2)} | "
                     f"{contrast('A',2,100,True)} | {contrast('A_pooled',2,100)} | {contrast('misses',2)} | "
                     f"{contrast('base_return',2)} |")
    return '\n'.join(lines)


def narratives(output,summary,screens,diagnostics,checks):
    new={r['condition']:r for r in screens if r['comparison']=='new_guard'}
    old=[r for r in screens if r['comparison']=='previous_original']
    unguarded=[r for r in screens if r['comparison']=='new_original']
    history=checks['historical_reconciliation']
    e=checks['evaluation']
    g=checks['gate']
    fmt_names=lambda rs:', '.join(f"`{r['condition']}`" for r in rs) or 'none'
    simultaneous=[r for r in new.values() if r['simultaneous']]
    retention=[r for r in new.values() if r['retention']]
    loss=[r for r in retention if r['retention_with_hit_loss']]
    lines=['| Condition | ΔD | ΔC (pp) | ΔH (%) | Simultaneous gate | Retention gate |',
           '| --- | ---: | ---: | ---: | --- | --- |']
    for c,r in new.items():
        if c=='reference':
            continue
        lines.append(f"| {c} | {r['delta_D']:+.3f} | {100*r['delta_C']:+.2f} | {100*r['relative_H']:+.2f} | "
                     f"{'pass' if r['simultaneous'] else 'fail'} | {'pass' if r['retention'] else 'fail'} |")
    final=diagnostics['final']
    stress=diagnostics['stress']
    q_note=(f"All {final['events']+stress['events']:,} filtered proposals had Q(Fire) > Q(Stay), "
            f"with zero exact Fire–Stay ties. There were {final['max_action_ties']+stress['max_action_ties']:,} "
            f"events with a maximum tie involving another action. The final event-weighted median gap was "
            f"{final['q_gap_median']:.3f}, with median Fire and Stay visitation counts "
            f"{final['fire_updates_median']:,.0f} and {final['stay_updates_median']:,.0f}. "
            f"On the final set, {final['either_zero_updates']:,} events had an unvisited Fire or Stay action and "
            f"{final['either_below_10_updates']:,} had fewer than 10 updates on at least one. "
            f"Both actions had at least 100 updates on {final['both_at_least_100_updates']:,} "
            f"events ({final['both_at_least_100_updates']/final['events']:.2%}). "
            f"At pre-distance one there were {final['by_pre_distance'][1]:,} events. "
            "Finite sampling is compatible with the low-count tail. Most errors persisted in frequently updated pairs. "
            "Counts and Q gaps do not identify a single cause, such as transition aliasing, persistent exploration, "
            "or unequal visitation histories. Repeated visits to the same table row are descriptive events.")
    threshold_note=(
        f"The previous and new original policies met the full simultaneous target for "
        f"{fmt_names([r for r in old if r['simultaneous']])} and {fmt_names([r for r in unguarded if r['simultaneous']])}, respectively. "
        f"The guarded simultaneous result was {fmt_names(simultaneous)}. The guarded retention set was {fmt_names(retention)}. "
        f"The cases with an actual hit loss inside the retention gate were {fmt_names(loss)}. "
        "All comparisons use a matched reference on the same split and with the same variant. "
        "The full screen requires at least five percentage points more close ticks, no larger D, and accuracy at least 98%. "
        "The simultaneous hit gate is +2%. The retention gate permits a 2% loss. "
        "A passing guarded method keeps the original reward-conditioned tracking and hit count. Its eligibility changes because the guard removes misses."
    )
    timing=(f"The prior inner matrix timer was {history['inner_training_seconds']:.3f} s. It starts after protocol freeze "
            "and covers the 21 runs, checkpoint writes, development replays, streaming window logs and exact final reloads. "
            f"The prior train phase was {history['phase_seconds']:.3f} s, and the outer command took "
            f"{history['outer_command_seconds']:.3f} s including process overhead. Final and stress evaluation and rendering were separate. "
            "The native C11 DLL was compiled with Zig 0.15.2 with fast-math and FP contraction disabled, and called from Python using ctypes. "
            "The reset path and NumPy PCG64 streams came from Python. "
            f"The count comes from {history['full_windows']:,} complete 300-tick windows and {history['partial_windows']} "
            f"200-tick tails, totaling {history['training_interactions']:,} actual updates. "
            "The archived Python training parity was exact for 1,001 steps at zero added weight and for all seven weights in a greedy-heavy regime. "
            f"This study independently matched {g['native_python_read_only_ticks_matched']:,} native/Python inference ticks with the frozen DLL, "
            "including actions, physical states and event flags. It did not repeat training or claim full-length Python training equivalence.")
    method=("Q-learning + firing guard queries the original greedy policy once per tick. It substitutes Stay only for "
            "Fire with nonzero pre-motion Manhattan distance. Both actions leave the crosshair fixed and consume the same target "
            "randomness. For this event the actual reward rises by exactly one because the miss penalty is removed. "
            "The transition, continuation term and proximity reward are identical for the paired action. The true environment "
            "therefore prefers Stay in this case, while the measured greedy tables preferred Fire. No Q-table or update count changed.")
    table=results_table(summary,'final')
    text=f"""# Inference-only firing guard

{threshold_note}

## Intervention and pairing

{method}

The 21 frozen policies were replayed on the first ten old final scenarios with independent Python environments. The gate compared physical state and target RNG state every tick, plus actual executed actions, distance, hits, misses and base rewards. The guard preserved state and distance at every paired tick. The held-out ranges 4500000–4500499 and 4600000–4600199 were checked for occupancy and frozen before evaluation. The latter uses 0.10 turn and speed-toggle probabilities. Rule is listed once per split and its actions were unaffected.

## Held-out final results

{table}

Values with uncertainty are seed means ± sample SD over seeds 20–22. Rule is a single policy. For each seed accuracy is total hits divided by actual shots over its scenarios. The seed mean gives every defined training seed equal weight, while the pooled ratio sums hits and shots over the three seeds before division. Zero executed shots give NA and any missing seed accuracy fails the learned-method accuracy gate. D, C and H are identical between the two variants.

{chr(10).join(lines)}

The objective coordinates and non-dominated mean sets are unchanged by the guard. The plotted rings overlap the corresponding original mean points exactly. See [objective trade-offs](figures/objective_tradeoffs.png) and [accuracy and misses](figures/accuracy_and_misses.png). The [threshold table](thresholds.csv) also gives the pooled-accuracy sensitivity, with the historical and new original results preserved.

## Held-out stress results

{results_table(summary,'stress')}

## Why the original policies fired while unaligned

{q_note}

The [seed diagnostics](q_diagnostics_per_seed.csv), [distinct indexed rows](q_diagnostics_distinct_states.csv) and per-tick Q event files retain the estimates, counts, ties and repetitions. Stay has the lower action ID, so an exact tie between Fire and Stay would not select Fire under the existing greedy rule. The guard repairs shot selection on already trained policies. It leaves the timing trade-off of an aligned hit unchanged, because a valid Fire still occupies the tick while the target moves.

## Historical reconciliation

The [recomputed historical seed table](historical_per_seed.csv), [comparison](historical_comparison.csv) and [paired differences](historical_paired_differences.csv) agree with the recorded values. For example, historical final Close 5.0 accuracy is 86.70% as an equal-weight seed mean and 85.36% after pooling all shots.

{timing}

## Execution and outputs

The study ran zero training interactions. It verified {g['paired_ticks']:,} paired gate ticks, {e['splits']['final']['windows']:,} held-out final windows and {e['splits']['stress']['windows']:,} stress windows, including Rule. The [execution checks](execution_checks.json) and [protocol](protocol.json) give counts, frozen hashes and commands. The [paired differences](paired_differences.csv) compare guard minus original for each seed, and [paired versus reference](paired_vs_reference.csv) compares weights within the same variant and seed. The [synchronized GIF](demo/synchronized_fire_guard.gif) is a preselected Close 5.0 seed-20 replay, decoded and verified over 301 frames.
"""
    paper=f"""# Paper addendum · Q-learning + firing guard

## Method

{method}

The original frozen policies remain a separate method. The guard was evaluated with the same held-out target trajectories on all 21 learned policies and with the fixed Rule comparator. All learned results preserve the three training seeds and the original reward conditions.

## Held-out final results

{table}

{threshold_note}

{q_note}
"""
    (output/'results_summary.md').write_text(text,encoding='utf-8')
    (output/'paper_addendum.md').write_text(paper,encoding='utf-8')


def report(output):
    protocol=checked(output)
    history=read_json(output/'prior_sha256_before.json')
    verify_history(history)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==protocol['git_head'],
            'Git HEAD changed')
    require(not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT),'Git index changed')
    summary,within,screens=compile_metrics(output,protocol)
    q=diagnostics(output,protocol,within)
    check=evaluation_audit(output,protocol,within)
    frontier=figures(output,summary,screens)
    gif=demo(output,protocol)
    verify_history(history)
    checks={'complete':True,'training_interactions':0,'prior_files_unchanged':len(history),
            'historical_reconciliation':read_json(output/'historical_reconciliation.json'),
            'gate':read_json(output/'gate/checks.json'),'evaluation':read_json(output/'evaluation/checks.json'),
            'saved_evaluation_audit':check,'q_diagnostic_totals':q,
            'objective_frontiers':frontier,'demo':gif,
            'report_sources_sha256':{p:sha(Path(__file__).with_name(p))
                                     for p in ('analysis.py','visuals.py','report.py')},
            'execution_records':{p:read_json(output/'execution'/f'{p}.json') for p in ('setup','gate','evaluate')}}
    narratives(output,summary,screens,q,checks)
    save_json(output/'execution_checks.json',checks)
    print('Checked all paired outputs, Q evidence, plots, GIF frames and historical hashes.',flush=True)
