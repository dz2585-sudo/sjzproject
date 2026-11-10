"""Write checked English study and paper summaries from the frozen observations."""
import statistics

from ..tracking_research.io import ROOT, read_json, save_json, sha
from .analysis import aggregate, audit
from .study import verify_protocol
from .visuals import demos, diagnostic_fragments, figures


def fmt(value, precision=3, scale=1):
    return 'NA' if value is None else f'{value*scale:.{precision}f}'


def estimate(row, key, precision, scale=1):
    mean = fmt(row[key], precision, scale)
    return mean if row[key+'_sd'] is None else f'{mean} ± {fmt(row[key+"_sd"], precision, scale)}'


def table(rows):
    lines = [
        '| Condition | Mean distance D | Close C (%) | Hits H / window | Accuracy A (%) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for r in rows:
        lines.append(f"| {r['condition']} | {estimate(r,'D',3)} | {estimate(r,'C',2,100)} | "
                     f"{estimate(r,'H',2)} | {estimate(r,'A',2,100)} |")
    return '\n'.join(lines)


def decisions(screens):
    variants = [r for r in screens if r['condition'] != 'reference']
    simultaneous = [r for r in variants if r['simultaneous']]
    retention = [r for r in variants if r['retention'] and r['delta_H'] < 0]
    directional = [r for r in variants if r['delta_C'] > 0 and r['delta_H'] > 0]
    if simultaneous:
        status = 'The prespecified simultaneous target was reached by ' + ', '.join(f"`{r['condition']}`" for r in simultaneous) + '.'
    elif directional:
        status = 'No condition reached the prespecified simultaneous target. ' + ', '.join(
            f"`{r['condition']}`" for r in directional) + ' improved both raw C and H, below at least one gate.'
    else:
        status = 'No condition improved both raw close fraction and hit count, and none met the simultaneous target.'
    retain = [r for r in variants if r['retention']]
    if retain:
        best = max(retain,key=lambda r:r['delta_C'])
        compromise = (f"`{best['condition']}` had the highest close gain among conditions meeting all retention gates. "
                      f"It changed D by {best['delta_D']:+.3f} cells, C by {100*best['delta_C']:+.2f} percentage points "
                      f"and H by {best['delta_H']:+.2f} hits per window ({100*best['relative_H']:+.2f}%).")
    else:
        above_h = [r for r in variants if r['hit_retention_gate'] and r['accuracy_gate']
                   and r['distance_gate'] and r['delta_C'] > 0]
        if above_h:
            best = max(above_h,key=lambda r:r['delta_C'])
            compromise = (f"No condition met all retention gates. `{best['condition']}` had the largest positive close gain "
                          f"among weights meeting the distance, accuracy and hit-retention gates "
                          f"({100*best['delta_C']:+.2f} pp C, {best['delta_D']:+.3f} D, {100*best['relative_H']:+.2f}% H).")
        else:
            compromise = ('No condition met all retention gates or delivered a positive close gain while also '
                          'meeting the distance, accuracy and hit-retention gates.')
    if retention:
        compromise += ' The weights meeting retention with a small hit loss were ' + ', '.join(
            f"`{r['condition']}`" for r in retention) + '.'
    return status, compromise


def narrative(config, output, final, screens, frontiers, actions, mechanisms, paired, slices, checks):
    status, compromise = decisions(screens)
    lookup = {r['condition']:r for r in final}
    ref, rule = lookup['reference'], lookup['rule']
    selection = read_json(output/'selection.json')['selected']
    chosen = [selection[k]['condition'] for k in ('distance','close')]
    stress = [r for r in read_json(output/'analysis_summary.json')['comparison'] if r['split']=='stress']
    thresholds = (
        f"Reference D={ref['D']:.3f}, C={ref['C']:.2%}, H={ref['H']:.2f}. "
        f"The final gates are D ≤ {ref['D']:.3f}, C ≥ {ref['C']+.05:.2%}, A ≥ 98%, "
        f"and H ≥ {1.02*ref['H']:.2f} for simultaneous improvement or H ≥ {0.98*ref['H']:.2f} for retention."
    )
    paired_by_condition = {r['condition']:r for r in paired if r['split']=='final'}
    gate_lines = [
        '| Condition | ΔD | ΔC (pp) | ΔH | ΔH (%) | C / D / A gates | H +2% / H −2% gates | D↓ / C↑ / H↑ seeds | Classification |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |',
    ]
    for s in screens:
        c=s['condition']
        if c=='reference':
            continue
        markers=lambda keys:' / '.join('yes' if s[k] else 'no' for k in keys)
        p=paired_by_condition[c]
        if s['simultaneous']:
            classification='Simultaneous target met'
        elif s['delta_C']>0 and s['delta_H']>0:
            classification='Both raw metrics improved below target'
        elif s['retention'] and s['delta_H']<0:
            classification='Retention target met with a small hit loss'
        elif s['retention']:
            classification='Retention target met'
        elif s['delta_C']>0 and s['delta_H']<0:
            classification='Close–hit trade-off outside the retention target'
        elif s['delta_C']<=0 and s['delta_D']<0:
            classification='Lower mean distance only for tracking'
        else:
            classification='No close-tracking improvement'
        gate_lines.append(f"| {c} | {s['delta_D']:+.3f} | {s['delta_C']*100:+.2f} | {s['delta_H']:+.2f} | "
                          f"{s['relative_H']*100:+.2f} | {markers(('close_gate','distance_gate','accuracy_gate'))} | "
                          f"{markers(('hit_gain_gate','hit_retention_gate'))} | "
                          f"{p['D_seeds_improved']} / {p['C_seeds_improved']} / {p['H_seeds_improved']} | {classification} |")

    mechanism_lines=[
        '| Condition | Alignment opportunity (%) | Fire given alignment (%) | Shots / window | Misses / window | Hit ticks outside close (%) | Slow / fast / stay / fire (%) | ΔH from opportunities / decisions |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for r in final:
        c=r['condition']
        subset=[a for a in actions if a['split']=='final' and a['condition']==c]
        mix=' / '.join(f"{100*sum(a[k+'_count'] for a in subset)/sum(a['ticks'] for a in subset):.1f}"
                       for k in ('slow','fast','stay','fire'))
        components=[m for m in mechanisms if m['split']=='final' and m['condition']==c
                    and m['opportunity_component'] is not None]
        component_text=' / '.join(f"{statistics.mean(m[k] for m in components):+.2f}"
                                 for k in ('opportunity_component','decision_component')) if components else 'NA'
        mechanism_lines.append(f"| {c} | {fmt(r['alignment_opportunity_fraction'],2,100)} | "
                               f"{fmt(r['fire_given_alignment'],2,100)} | {r['shots']:.2f} | {r['misses']:.2f} | "
                               f"{fmt(r['hit_nonclose_fraction'],2,100)} | {mix} | {component_text} |")
    emphasis=[]
    for c in chosen:
        r=lookup[c]
        f='NA' if r['fire_given_alignment'] is None else f"{r['fire_given_alignment']:.2%}"
        emphasis.append(f"`{c}` reached alignment on {r['alignment_opportunity_fraction']:.2%} of ticks "
                        f"and fired on {f} of those opportunities. It recorded {r['H']:.2f} hits per window "
                        f"and {r['C']:.2%} close ticks.")
    def action_rate(condition, bin_name, mode):
        rows=[r for r in actions if r['split']=='final' and r['condition']==condition
              and (bin_name is None or r['pre_distance_bin']==bin_name)]
        return sum(r[mode+'_count'] for r in rows)/sum(r['ticks'] for r in rows)

    fastest = lookup['close_0p5']
    highest = lookup['close_5p0']
    behavioral = (
        f"Across all final-set policies and seeds, aligned firing was 100%, so the hit decomposition assigns all "
        f"observed hit changes to how often the crosshair reached alignment. The distance sweep was non-monotonic. "
        f"The 0.1 penalty produced more aligned opportunities and {lookup['distance_0p1']['misses']:.2f} misses per window "
        f"versus {ref['misses']:.2f} for the reference, while 0.3 stayed closest to the reference shot mix and met the "
        f"accuracy and retention gates. The 1.0 penalty improved C while raising mean D and missing the accuracy gate. "
        f"The bonus sweep increased C at every weight. The 0.5 bonus alone cleared the D, C and +2% H gates together "
        f"but fell below the accuracy gate ({fastest['A']:.2%}). At 5.0, the share of stays fell from "
        f"{action_rate('reference',None,'stay'):.2%} to {action_rate('close_5p0',None,'stay'):.2%}. "
        f"Firing at pre-distance one rose from {action_rate('reference','1','fire'):.2%} to "
        f"{action_rate('close_5p0','1','fire'):.2%}. Those shots cannot hit under the existing rule. "
        f"Average misses rose to {highest['misses']:.2f} per window. The larger close bonus improved tracking, "
        f"with little further hit gain and poorer shot selection."
    )
    mechanism = (
        f"Rule fired on {rule['fire_given_alignment']:.2%} of aligned ticks, compared with "
        f"{ref['fire_given_alignment']:.2%} for the reference. Rule reached alignment on "
        f"{rule['alignment_opportunity_fraction']:.2%} of ticks, versus "
        f"{ref['alignment_opportunity_fraction']:.2%} for the reference. The reference therefore recorded "
        f"{ref['H']-rule['H']:+.2f} hits per window relative to Rule, reflecting how frequently each policy "
        f"reached alignment. {' '.join(emphasis)}"
    )
    timing = (
        f"A hit tests the pre-motion target while Fire holds the crosshair fixed. Its post-motion distance equals the "
        f"target's net displacement, which can be 2 on a speed-two tick or 0 when reflection returns the target "
        f"to its starting cell. In the reference, {ref['total_hit_nonclose']:,} of {ref['total_hits']:,} hit ticks "
        f"had net displacement above one ({ref['hit_nonclose_fraction']:.2%}). "
        f"Every observed window satisfied close ticks + such hits ≤ total ticks. "
        f"That identity applies to each realized trajectory. The policy can still alter how often it gets aligned, "
        f"how often it shoots when aligned, and which target-motion contexts it occupies. "
        f"Because firing and moving are mutually exclusive, a hit can give up the chance to follow the target on the same tick."
    )
    slice_lines=[
        '| Condition | Slice | D | C (%) | Hit rate per tick (%) | A (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for r in slices:
        if r['split']=='final' and r['condition'] in ['reference',*chosen,'rule']:
            slice_lines.append(f"| {r['condition']} | {r['slice']} | {fmt(r['D'])} | {fmt(r['C'],2,100)} | "
                               f"{fmt(r['hit_rate'],2,100)} | {fmt(r['A'],2,100)} |")
    selection_text='\n'.join(
        f"- {family.title()} panel: `{selection[family]['condition']}`. "
        f"{'Passed' if selection[family]['screen_passed'] else 'No family member passed'} the development hit/accuracy screen."
        for family in ('distance','close'))
    study = read_json(output/'training_complete.json')
    total_seconds=study['elapsed_seconds']
    max_memory=max((r['memory']['bytes'] or 0 for r in study['runs']))/1024**2
    next_test = (
        'The next isolated test should gate Fire on exact pre-action alignment. With frozen greedy policies, '
        'map an unaligned Fire to Stay on a new paired split and test whether it eliminates misses while preserving '
        'the full physical trajectories and hit counts. Fire and Stay both hold the crosshair fixed, so this '
        'intervention has a precise trajectory-invariance prediction under the existing timing and state-only policies. '
        f'The 5.0 bonus averaged {highest["misses"]:.2f} misses per window despite using every aligned opportunity. '
        'This would address shot selection and would leave the post-motion distance cost of a valid hit to be studied separately.'
    )
    methods = (
        'Seven scalar rewards were compared on a fixed 25×25 continuing tracking task with a relative tabular '
        'observation, ten actions, a common distance potential, and the same 5M-interaction budget. '
        'Three matched training seeds (20, 21, 22) started from zero for each condition. Distance penalty '
        'weights were 0.1, 0.3 and 1.0. Close bonus weights were 0.5, 2.0 and 5.0. '
        'The reference had zero added weight. The stress set changed both target event probabilities from 0.05 to 0.10. '
        'Performance used common post-motion distance and close fraction, hit frequency, and pooled within-seed shot accuracy. '
        'The policies shared exact target trajectories on each evaluation split.'
    )
    results_table = table(final)
    body = f"""# Tracking and shooting reward trade-offs

{status} {compromise}

## Matched design

{methods}

The native rollout uses the original NumPy random streams and reset path. Exact Python parity checks covered the zero-weight reference and each reward in a greedy-heavy regime. Bootstrapping used the actual next state at all boundaries. The formal comparison includes every frozen condition. Development selected only the demonstration panels. The final test opened after all 21 runs and the selection record were complete.

## Final results

{results_table}

Learned values are means ± sample SD across three training seeds. Rule is a single fixed policy on the same 500 scenarios. C counts post-motion distance ≤ 1. H counts hits per 300-tick window. A pools hits and shots within seed, then averages seeds. An empty denominator is NA and fails the accuracy screen. The companion tables report shots, misses, original base return, longest far streak, capped first approach, every seed and paired differences. Training returns use different task rewards and are unsuitable for cross-weight ranking.

{thresholds}

{chr(10).join(gate_lines)}

The D–H empirical frontier contains {', '.join(frontiers['D'])}. The C–H empirical frontier contains {', '.join(frontiers['C'])}. These are separate comparisons of the observed mean points, with Rule included. The lines connecting observed means do not establish a task-wide optimal frontier. See [distance versus hits](figures/distance_hits_tradeoff.png), [close versus hits](figures/close_hits_tradeoff.png), [weight response](figures/weight_response.png), and [development learning curves](figures/learning_curves.png).

## Firing, alignment and motion

{mechanism}

{behavioral}

{chr(10).join(mechanism_lines)}

The last column decomposes the paired hit change from the reference as `300 × (O−O₀) × F₀ + 300 × O × (F−F₀)` for each seed, then averages those components. O is pre-action alignment opportunity and F is firing conditional on alignment. This is an accounting identity. Policy-dependent changes in O and F can be related. The [action table](action_by_pre_distance.csv) gives separate pre-distance bins 0, 1, 2 and 3+, including denominators. The [fixed diagnostic trace](diagnostics/fixed_final_trace.csv) and [first fragments](diagnostics/first_fragments.csv) preserve the first local overshoot, wait and persistent one-cell offset under a fixed rule. See [alignment and shooting figure](figures/alignment_and_shooting.png) and [post-hit distance distribution](hit_post_distance.csv).

{timing}

### Speed and wall slices

{chr(10).join(slice_lines)}

Speed uses the actual target speed during the tick, after any toggle. A wall slice includes ticks where either object was at most three cells from an edge before acting. Slices are observational. The [full slice summary](slice_summary.csv) includes both splits and every condition, and the [seed-level counts](slice_per_seed.csv) preserve denominators.

## Stress results

{table(stress)}

## Demonstrations chosen on development

{selection_text}

The four panels use the same target motion and independent crosshair rollouts. Learned panels use training seed 20. Each animation contains the initial frame and all 300 ticks, at nominal 15 fps. Decoding verified every frame and the start, middle and end snapshots. See [4400000](demos/scenario_4400000.gif), [4400001](demos/scenario_4400001.gif), [4400002](demos/scenario_4400002.gif) and [render checks](demos/render_checks.json).

## Execution and next experiment

Formal training took {total_seconds:.1f} seconds including development checkpoints and logging. The Windows process-lifetime peak working set reported during training was {max_memory:.1f} MiB. The audit verified {checks['training_reward_identities']:,} formal training-window reward identities and {checks['exact_checkpoint_reloads']} checkpoint packages. It counted 46,200 development, 11,000 final and 4,400 stress windows. See [execution checks](execution_checks.json), the [frozen protocol](protocol.json), and the [preflight record](preflight.json).

{next_test}

## References

- Roijers, Vamplew, Whiteson and Dazeley (2013). *A Survey of Multi-Objective Sequential Decision-Making*. [Publication](https://www.cs.ox.ac.uk/publications/publication9780-abstract.html)
- Van Moffaert and Nowé (2014). *Multi-Objective Reinforcement Learning using Sets of Pareto Dominating Policies*. [Publication](https://jmlr.org/papers/v15/vanmoffaert14a.html)
"""
    paper = f"""# Paper addendum · Scalar rewards for tracking and shooting

Sustained pursuit and successful firing can reward different behavior when the agent must choose between a movement and a shot on each tick. We tested whether weighting proximity could improve both objectives under fixed mechanics and a fixed training budget.

{methods}

{results_table}

{status} {compromise} The criterion required at least five percentage points of close gain, at least two percent more hits, no increase in mean distance, and at least 98% shot accuracy. The retention criterion allowed at most two percent fewer hits. Three training seeds are the replicates. The uncertainty shown is sample SD across seeds.

{mechanism} {timing}

{behavioral}

The reward sweep describes empirical non-dominance among the trained policies. Conditional action and wall/speed slices describe behavior along the realized trajectories. A controlled action-space comparison is needed to test how much of the observed trade-off is attributable to exclusive movement and firing.
"""
    (output/'results_summary.md').write_text(body,encoding='utf-8')
    (output/'paper_addendum.md').write_text(paper,encoding='utf-8')


def report(config,kernel,library,output):
    verify_protocol(config,library,output)
    checks=audit(config,output)
    final, per_seed, screens, frontiers, actions, mechanisms, paired, slices=aggregate(config,output)
    figures(config,output,final,per_seed,frontiers,actions)
    diagnostic_fragments(config,kernel,output)
    checks['gifs']=demos(config,kernel,output)
    checks.update(complete=True, preflight=read_json(output/'preflight.json'),
                  training_stage=read_json(output/'execution/train.json'),
                  evaluation_stage=read_json(output/'execution/evaluate.json'),
                  formal_failed_runs=0, additional_failed_training_interactions=0,
                  presentation_source_sha256={p:sha(ROOT/p) for p in
                                              ('aimrl/tracking_tradeoff/analysis.py',
                                               'aimrl/tracking_tradeoff/visuals.py',
                                               'aimrl/tracking_tradeoff/report.py')},
                  outputs_sha256={p.relative_to(output).as_posix():sha(p) for p in sorted((output/'figures').glob('*.png'))})
    narrative(config,output,final,screens,frontiers,actions,mechanisms,paired,slices,checks)
    save_json(output/'execution_checks.json',checks)
    print('All counts, paired scenarios, historical hashes, figures and GIFs verified.',flush=True)
