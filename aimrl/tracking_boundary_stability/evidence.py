"""Post-training interpretation from frozen artifacts; never changes the protocol.

This supplementary analysis was added after formal training. It changes no
learning implementation, model, prespecified endpoint, event definition or gate.
"""
from collections import defaultdict
import json
from datetime import datetime,timezone

import numpy as np

from ..tracking_research.io import ROOT,read_csv,read_json,save_csv,save_json,sha
from .core import CONDITIONS,OUT,config,load_model,require
from .report import md_table


def main():
    from .study import check_protocol
    c=config();check_protocol(c)
    comparison=read_json(OUT/'comparison.json');per_seed=read_json(OUT/'per_seed.json')
    coverage=[]
    for condition in CONDITIONS:
        for seed in c['study']['training_seeds']:
            a,_=load_model(OUT/'runs'/condition/f'seed_{seed}/checkpoints/step_5000000.npz',c)
            visits=a.counts.sum(axis=1,dtype=np.uint64)
            coverage.append(dict(condition=condition,seed=seed,state_rows=len(visits),visited_rows=int((visits>0).sum()),
                         visited_row_fraction=float((visits>0).mean()),visited_pairs=int((a.counts>0).sum()),
                         visited_pair_fraction=float((a.counts>0).mean()),
                         visited_pair_update_median=float(np.median(a.counts[a.counts>0])),
                         unvisited_pairs=int((a.counts==0).sum())))
            del a
    save_csv(OUT/'coverage.csv',coverage)
    scenario_deltas=[];scenario_summary=[]
    for split in ('final','stress'):
        for condition in CONDITIONS[1:]:
            by_scenario=defaultdict(list)
            for seed in c['study']['training_seeds']:
                root=OUT/'evaluation'/split
                base=read_csv(root/f'R-Fixed_seed{seed}_guard_windows.csv')
                rows=read_csv(root/f'{condition}_seed{seed}_guard_windows.csv')
                for b,r in zip(base,rows):
                    require(b['scenario_seed']==r['scenario_seed'],'Scenario pairing mismatch')
                    d=(int(r['distance_sum'])-int(b['distance_sum']))/300
                    close=(int(r['close_ticks'])-int(b['close_ticks']))/300
                    h=int(r['hits'])-int(b['hits'])
                    rec=dict(split=split,condition=condition,seed=seed,scenario_seed=int(r['scenario_seed']),D_delta=d,C_delta=close,H_delta=h)
                    scenario_deltas.append(rec);by_scenario[r['scenario_seed']].append(rec)
            means=[np.mean([r['C_delta'] for r in rows]) for rows in by_scenario.values()]
            scenario_summary.append(dict(split=split,condition=condition,scenarios=len(means),
                         C_improved_scenarios=sum(x>0 for x in means),C_worsened_scenarios=sum(x<0 for x in means),
                         C_tied_scenarios=sum(x==0 for x in means),
                         interpretation='Descriptive scenario averages over the same five paired training seeds; not independent training replicates.'))
    save_csv(OUT/'scenario_differences.csv',scenario_deltas);save_csv(OUT/'scenario_summary.csv',scenario_summary)
    slices=read_csv(OUT/'boundary_speed_slices.csv');slice_table=[]
    for split in ('final','stress'):
        for group in ('boundary','interior','speed_1','speed_2'):
            rows={cond:[r for r in slices if r['split']==split and r['variant']=='guard' and r['condition']==cond and r['group']==group]
                  for cond in ('R-Fixed','R-Decay')}
            close={cond:float(np.mean([float(r['C']) for r in rs])) for cond,rs in rows.items()}
            dist={cond:float(np.mean([float(r['D']) for r in rs])) for cond,rs in rows.items()}
            slice_table.append([split,group,f'{100*close["R-Fixed"]:.2f}%',f'{100*close["R-Decay"]:.2f}%',
                                f'{100*(close["R-Decay"]-close["R-Fixed"]):+.2f} pp',f'{dist["R-Decay"]-dist["R-Fixed"]:+.4f}'])
    coverage_table=[]
    for cond in CONDITIONS:
        rs=[r for r in coverage if r['condition']==cond]
        coverage_table.append([cond,f'{100*np.mean([r["visited_row_fraction"] for r in rs]):.2f}%',
                               f'{100*np.mean([r["visited_pair_fraction"] for r in rs]):.2f}%',
                               f'{np.mean([r["visited_pair_update_median"] for r in rs]):.1f}'])
    raw_table=[]
    for split in ('final','stress'):
        for cond in CONDITIONS:
            r=next(x for x in comparison if (x['split'],x['condition'],x['variant'])==(split,cond,'original'))
            raw_table.append([split,cond,f'{100*r["invalid_fire_tick_fraction"]:.4f}%',
                              f'{100*r["A"]:.3f}%' if r['A'] is not None else 'NA'])
    seed_table=[]
    for split in ('final','stress'):
        for seed in c['study']['training_seeds']:
            b=next(r for r in per_seed if (r['split'],r['condition'],r['variant'],r['seed'])==(split,'R-Fixed','guard',seed))
            a=next(r for r in per_seed if (r['split'],r['condition'],r['variant'],r['seed'])==(split,'R-Decay','guard',seed))
            seed_table.append([split,seed,f'{100*b["C"]:.3f}%',f'{100*a["C"]:.3f}%',
                               f'{100*(a["C"]-b["C"]):+.3f} pp',f'{a["D"]-b["D"]:+.4f}',f'{a["H"]-b["H"]:+.3f}'])
    scenarios=[[r['split'],r['condition'],r['C_improved_scenarios'],r['C_worsened_scenarios'],r['C_tied_scenarios']] for r in scenario_summary]
    text=f'''# Evidence-specific conclusions

This interpretation uses the frozen 5M artifacts. Supplementary coverage and per-scenario summaries are descriptive additions after training; no protocol, learning source, endpoint or threshold was changed.

## Which factor helped?

R-Decay improves Final C by 4.2865 percentage points (68.8989%→73.1855%), D by −0.03409 and H by +0.2744 hits/window (+0.325%). Stress C gains 3.9013 points (62.4473%→66.3487%), D changes −0.02390 and H +0.746 (+1.010%). Guarded A is 100% in all four cells. Neither split reaches the prespecified +5pp mean practical gate; H also misses the stronger +2% gate. No condition or individual training seed reaches absolute C≥75%. Only R-Decay seed 62 passes the practical per-seed gate in both splits; no seed passes the stronger simultaneous gate.

Final C sample SD falls from 1.5612pp to 0.0422pp; Stress C SD falls from 1.2536pp to 0.1967pp. This is clear stabilization of C in these five seeds. It is not uniform stabilization of every metric: Final H SD rises slightly (0.0887→0.1011), and Stress D SD rises (0.0123→0.0205). Stress D worsens for seed 63 despite its C/H gains.

All five relative-representation seeds improve C, so the benefit is not solely rescue of one bad seed. Seed 62 has the largest gain (+7.0593pp Final, +6.0967pp Stress); other Final seeds gain 3.3987–3.7033pp. This explains part of the reduction in worst-seed risk without attributing the full mean gain to one seed.

{md_table(['Split','Seed','R-Fixed C','R-Decay C','C difference','D difference','H difference'],seed_table)}

The boundary augmentation worsens D, C and H against the matching R condition in all five seeds, with both learning-rate settings and both splits. At fixed alpha its Final C effect is −12.5151pp and H −23.6164; under decay C is −15.3604pp and H −14.9376. The counterexample still proves missing boundary context; the fixed-budget result does not support this particular larger tabular encoding as a practical improvement. Sample dilution is a plausible explanation, supported by coverage below, but was not independently randomized and is not proven as the sole cause.

Within B, decay improves Final C in four of five seeds and H in all five; seed 61 loses C and worsens D. Its Final mean C gain is only 1.4412pp, with H +8.9532. The C interaction is negative (−2.8453pp Final, −2.3927pp Stress); the H interaction is positive (+8.6788 Final, +6.586 Stress). Thus the two factors do not yield an additive C improvement.

{md_table(['Condition','Rows visited','Pairs visited','Mean seed median updates / visited pair'],coverage_table)}

## Which scenes improved?

The following slices use each policy's own physical trajectory and preserve the historical slice definitions. They are descriptive; moving into a different state distribution is part of the observed result.

{md_table(['Split','Slice','R-Fixed C','R-Decay C','C difference','D difference'],slice_table)}

Per-scenario C differences below first average over the five paired trained seeds. Counts describe scene breadth; they do not turn scenes into independent trained models. All scene/seed differences are retained in scenario_differences.csv.

{md_table(['Split','Condition','C improved scenes','C worsened scenes','Tied'],scenarios)}

## Did firing ranking and Bellman stability improve?

On distinct common unaligned physical states, Fire>Stay falls from 39.611% to 32.778% with R decay; event-weighted rates fall from 39.176% to 29.884%. Greedy invalid Fire on these same states falls from 1.347% to 0.623% (state weight) and 1.690% to 0.542% (event weight). Mean absolute Q change over 4.5M→5M drops 0.8685→0.7169 with state weighting. This supports a finite-update stability contribution.

It does not establish convergence or make fixed alpha the sole cause: common-state Fire Bellman MAE is 65.892→66.013, and Stay MAE is 65.420→65.801, so those residuals do not improve. The common set includes far and poorly visited states; large residuals and the representation's aliasing remain. The unit Fire/Stay one-step target identity is fully verified despite learned misranking.

B has lower common-state error fractions but about 36.7% of Fire pairs are unvisited versus 2.6% in R (state weighting). Zero-tied rows reduce strict Fire>Stay fractions. B also makes more greedy invalid Fire on common states and performs worse physically. Lower strict misranking fractions or smaller absolute Bellman residuals alone are insufficient evidence that its tracking solution is better.

{md_table(['Split','Condition','Invalid Fire / all ticks, original','Original accuracy'],raw_table)}

Most raw invalid-fire reduction for R occurs in seeds 61/62; seed 60 has a small Final increase. Empty Fire and Stay have identical physical effects. Their ranking improvement cannot by itself explain the tracking gain; other movement-action values must also change. Guard preserves every physical metric and hit, so the tracking improvement above survives removal of empty-fire penalties.

## Recommended next research

Prioritize a controlled planning baseline with the current action set. One-step lookahead is already represented by the historical Rule policy; the next planning study should test a prespecified multi-step horizon with stochastic target outcomes and the same observation/action constraints. It can test whether temporal action coordination raises the C/H frontier before altering environment semantics. The present evidence does not establish that the existing actions impose a 75% ceiling.

Action design is the next separate factor if planning exposes a persistent C/H conflict: valid Fire happens before target motion and consumes a stationary tick, while speed-two motion often moves a hit outside the close threshold. A move-and-fire action or changed timing would alter the task and must not be mixed into the present factorial comparison. Eligibility traces are lower priority from this evidence: decay stabilizes C and ranking, but the remaining boundary/sample-density and action-timing issues are not specifically evidence of delayed credit assignment. This is a research recommendation, not a tested ranking of the three proposed interventions.

## Remaining validation limitation

All three GIFs contain the initial frame plus every one of the 300 ticks. Decoding verified all 903 frame pixel arrays and exact per-frame timings. Static layout was visually inspected. Actual browser/player playback was not verified: the Windows Computer Use tool returned “Computer Use was not approved to use Google Chrome”; the in-app browser then rejected local file URLs by its protocol policy. No security setting or alternate transport was used to bypass those controls. See demos/player_check.json. This is the only requested validation left incomplete.
'''
    (OUT/'conclusions.md').write_text(text,encoding='utf-8')
    for filename in ('results_summary.md','paper_addendum.md'):
        p=OUT/filename;s=p.read_text(encoding='utf-8')
        if '## Evidence-specific supplement' not in s:
            p.write_text(s+'\n## Evidence-specific supplement\n\nSee [the measured interpretation and next research recommendation](conclusions.md) for factor effects, seed/scene attribution, residual caveats and the actual-player limitation.\n',encoding='utf-8')
    save_json(OUT/'post_training_analysis.json',dict(source='aimrl/tracking_boundary_stability/evidence.py',source_sha256=sha(__file__),
              executed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),
              purpose='Interpretation, coverage, scene breadth and transparent remaining validation limitation.',
              training_updates_added=0,frozen_sources_changed=False))
    save_json(OUT/'demos/player_check.json',dict(complete=False,decoded_complete=True,static_layout_inspected=True,
              windows_player_error='Computer Use was not approved to use Google Chrome',
              in_app_browser_error='Local file protocol disallowed by browser URL security policy; allowed protocols http/https.',
              actual_playback_observed=False,workaround_attempted=False,
              user_action='Open the three delivered GIF files in an authorized local player to complete the remaining playback observation.'))
    print('Wrote frozen-artifact evidence supplement; no new training.',flush=True)


if __name__=='__main__':main()
