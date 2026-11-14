"""Measured interpretation supplement added after the fixed formal evaluation.

No controller, model, cache, gate, event rule or primary calculation is changed.
"""
from collections import defaultdict
from datetime import datetime,timezone

import numpy as np

from ..tracking_research.io import ROOT,read_csv,read_json,save_csv,save_json,sha
from .core import OUT,Policy,config,methods
from .report import table


def main():
    from .study import verify_protocol
    c=config();verify_protocol(c)
    comparison=read_json(OUT/'comparison.json');mean={(r['split'],r['family']):r for r in comparison}
    slices=read_csv(OUT/'slice_comparison.csv');behaviors=read_csv(OUT/'behavior_quantities.csv')
    selected=read_json(OUT/'behavior_selection.json');gates=read_json(OUT/'thresholds.json')
    lines=['# Behavioral and threshold interpretation supplement','',
      'This is a descriptive reading of the frozen outputs, added after evaluation. It changes no prespecified method or criterion.','']
    speed_rows=[];movement_rows=[]
    for split in ('final','stress'):
        for group in ('pre_speed_2','post_speed_2'):
            base=next(r for r in slices if r['split']==split and r['family']=='R-Decay' and r['group']==group)
            for family in ('Plan-H1','Plan-H2','Plan-H3','Hybrid-H3'):
                r=next(r for r in slices if r['split']==split and r['family']==family and r['group']==group)
                speed_rows.append([split,group,family,f'{float(r["D"])-float(base["D"]):+.4f}',
                    f'{100*(float(r["C"])-float(base["C"])):+.3f} pp',
                    f'{float(base["hit_rate_per_300_slice_ticks"]):.3f} → {float(r["hit_rate_per_300_slice_ticks"]):.3f}'])
        base=mean[split,'R-Decay'];h1=mean[split,'Plan-H1'];h2=mean[split,'Plan-H2'];h3=mean[split,'Plan-H3'];hy=mean[split,'Hybrid-H3'];zero=mean[split,'Plan-H3-Zero']
        lines += ['## '+split.title(),'',
          f'H1→H2 adds {h2["H"]-h1["H"]:.3f} hits/window, while C changes {100*(h2["C"]-h1["C"]):+.3f}pp. H2→H3 adds another {h3["H"]-h2["H"]:.3f} hits and {100*(h3["C"]-h2["C"]):+.3f}pp C. More horizon does not monotonically increase close time; the large H1 close score comes with much lower hit count.',
          '',
          f'Adding the learned Q terminal to H3 adds {hy["H"]-h3["H"]:.4f} hits and {100*(hy["C"]-h3["C"]):+.4f}pp C beyond the Phi terminal. Relative to R-Decay it changes D {hy["D"]-base["D"]:+.4f}, C {100*(hy["C"]-base["C"]):+.4f}pp and H {(hy["H"]/base["H"]-1)*100:+.3f}%. The extra learned terminal gain is smaller than the H1→H2 hit gain, while both are genuine within-family comparisons.',
          '',
          f'First pre-action alignment averages {base["first_alignment_tick"]:.4f} ticks for R-Decay, {h3["first_alignment_tick"]:.4f} for Plan-H3 and {hy["first_alignment_tick"]:.4f} for Hybrid. H3-Zero first close averages {zero["first_close_tick"]:.3f} ticks versus {h3["first_close_tick"]:.3f} for H3-Phi, consistent with the distance terminal helping acquisition beyond the short reward horizon.',
          '']
        for family in ('R-Decay','Plan-H1','Plan-H2','Plan-H3','Hybrid-H3'):
            rs=[r for r in behaviors if r['split']==split and r['family']==family and int(r['pre_speed'])==2]
            movement_rows.append([split,family,*[f'{100*np.mean([float(r[k]) for r in rs]):.3f}%' for k in ('fast_fraction','slow_fraction','stay_fraction','fire_fraction','target_axis_crossing_fraction')]])
        pairs=[g for g in gates if g['split']==split and g['candidate'].startswith('Hybrid-H3_s')]
        lines.append(f'Hybrid per-model gates: practical {sum(g["practical_pass"] for g in pairs)}/5, stronger simultaneous {sum(g["simultaneous_pass"] for g in pairs)}/5, absolute C75 with hit retention {sum(g["absolute_C75_and_H_retained"] for g in pairs)}/5. All five same-model D/C/H differences are preserved in paired_differences.csv; the absolute and relative tests must not be substituted for one another.')
        lines.append('')
    lines+=['## Speed-two detail','',
      'The before-decision speed is observable. The after-toggle speed is the actual motion speed in that tick and retains the historical slicing definition. These are policy-specific descriptive rates; hits per 300 slice ticks are not the main per-window H.','',
      table(['Split','Speed definition','Method','D difference vs R-Decay','C difference','Slice hit rate, R→method'],speed_rows),'',
      'Both Plan-H3 and Hybrid improve speed-two D/C and hit rates under both speed definitions in both splits. Thus their close gains are not purchased by losing hits in this slice. This is evidence of useful empirical room within the original action set, under the additional full-state/known-model/compute conditions.','',
      '## Movement and firing behavior','',
      table(['Split','Method','Fast','Slow','Stay','Fire','Cross current target axis'],movement_rows),'',
      'The movement table records mixtures, not a causal explanation. Crossing the current target coordinate on the exact same row/column can be deliberate anticipation of target motion, so fewer crossings alone do not prove less harmful overshoot. Common-state actions and paired first-event fragments are provided to inspect that distinction.','']
    absent=[e for e in selected if e['rule']=='aligned_fire_to_move' and not e['found']]
    present=[e for e in selected if e['rule']=='aligned_fire_to_move' and e['found']]
    lines.append(f'There were {len(present)} qualifying aligned R-Fire→planner-movement examples and {len(absent)} explicit absent method/split searches. In this run none occurred for any planned family in either split. Every formal policy had fire_given_alignment=1 and aligned_move_fraction=0. Therefore the observed gains do not come from deliberately postponing a currently valid shot; they come from movement/approach decisions before the next alignment opportunity.')
    lines+=['',
      'For instance, the frozen earliest-event selection can show a worse planning recovery, and those clips are retained. No event is chosen for positive performance. Same-state proposals and each controller’s own evolving physical states remain separately labelled.',
      '',
      '## Next intervention','',
      'A 2-cell movement experiment is worth doing as a separate action-design factor. Hybrid improves the current speed-two frontier but leaves substantially lower close time at speed two than at speed one. Existing moves are one or three cells and all methods still Fire immediately on alignment; a two-cell option could change how the next opportunity is reached without changing shot timing. This remains a hypothesis, not a proven source of all remaining loss. Preserve this stochastic planning baseline and use a controlled action-set comparison. A longer horizon and probability-calibrated planning are also distinct clean follow-ups; none of these measurements establishes a theoretical limit.',
      '',
      'All primary relative +5pp gates fail at the family-mean level. Final Hybrid passes absolute C≥75% while retaining hits, but that is not the +5pp gate. Final H1 also passes C≥75% but retains only about 60% of R-Decay hits, so it is not joint improvement. Stress has no absolute 75% pass. No controller was dropped from reporting.']
    import ctypes
    frequency=ctypes.c_longlong()
    clock_period=None
    if hasattr(ctypes,'windll') and ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(frequency)):
        clock_period=1e9/frequency.value
    startup=read_json(OUT/'precomputation.json');build=read_json(OUT/'native/build.json')
    api=read_csv(OUT/'diagnostics/api_latency.csv')
    latency=[]
    for r in api:
        if r['family'] in ('Plan-H1','Plan-H2','Plan-H3','Plan-H3-Zero','Hybrid-H3'):
            latency.append([r['policy'],f'{float(r["load_seconds"]):.6f}',r['first_api_call_ns'],r['warm_median_ns'],r['warm_p95_ns'],r['warm_max_ns']])
    lines+=['','## Cold preparation and timer resolution','',
      f'Native compilation took {build["elapsed_seconds"]:.4f}s. Full-study precomputation and storage took {startup["elapsed_seconds"]:.4f}s; charged together, these are {build["elapsed_seconds"]+startup["elapsed_seconds"]:.4f}s before loading/querying the delivered cached controllers. This is the complete study build, not a claimed minimum startup time for one controller.',
      '',
      f'The native performance counter period on this host is {clock_period}ns. The recorded native median of 0ns is below measurement resolution, not zero computation. Approximately 100/101ns P95 bins likewise must not be read as nanosecond precision or an end-to-end latency result. Native maxima include scheduling interruptions. Keep raw timings for audit, and use the separately measured Python observation API below for a fuller call cost. Neither first API lookup nor second pass flushes OS caches.',
      '',table(['Policy','Load seconds','First API ns','Warm median ns','Warm P95 ns','Warm max ns'],latency)]
    import time
    with np.load(ROOT/c['planning']['common_diagnostics']['source'],allow_pickle=False) as z:
        benchmark_states=[tuple(map(int,s)) for s in z['states'][:512]]
    paired_latency=[]
    for method in methods(c):
        policy=Policy(c,method);first=time.perf_counter_ns();policy.act(benchmark_states[0]);first=time.perf_counter_ns()-first
        passes=[]
        for _ in range(2):
            samples=[]
            for state in benchmark_states:
                t=time.perf_counter_ns();policy.act(state);samples.append(time.perf_counter_ns()-t)
            passes.append(samples)
        paired_latency.append(dict(policy=method['id'],load_seconds=policy.load_seconds,first_query_ns=first,
              pass1_median_ns=float(np.median(passes[0])),pass1_p95_ns=float(np.quantile(passes[0],.95)),pass1_max_ns=max(passes[0]),
              pass2_median_ns=float(np.median(passes[1])),pass2_p95_ns=float(np.quantile(passes[1],.95)),pass2_max_ns=max(passes[1]),
              states=512,scope='Identical preconverted tuple inputs and timing boundaries in both passes; no OS-cache flush; no environment steps.'))
    save_csv(OUT/'paired_api_latency.csv',paired_latency)
    lines+=['',
      'For equal timing boundaries, paired_api_latency.csv additionally queries the first 512 previously frozen common states twice, with tuple conversion outside both timing intervals. The original diagnostics/api_latency.csv first pass excludes conversion while its second pass includes saved-array-to-tuple conversion, so that difference must not be treated as a cache-speedup estimate. Neither benchmark flushes OS caches or repeats an environment evaluation. Full precomputation remains separately charged.',
      '',
      'An absent pre-action alignment recovery is stored as null and censored at the 30-tick horizon or window end, never scored as a zero-tick successful recovery.']
    (OUT/'behavior_interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name in ('conclusions.md','results_summary.md','paper_addendum.md'):
        path=OUT/name;s=path.read_text(encoding='utf-8')
        if '## Behavior interpretation supplement' not in s:
            path.write_text(s+'\n## Behavior interpretation supplement\n\nSee [the measured speed-two, movement and threshold interpretation](behavior_interpretation.md). This post-evaluation reading changes no frozen method, metric or gate.\n',encoding='utf-8')
    repro=OUT/'REPRODUCE.md';s=repro.read_text(encoding='utf-8')
    if '## Optional measured interpretation supplement' not in s:
        repro.write_text(s+'\n## Optional measured interpretation supplement\n\nAfter report generation, run `python -m aimrl.tracking_planning.post_analysis` to reproduce behavior_interpretation.md and its links. This descriptive post-evaluation module changes no primary result or frozen source. Native zero-nanosecond timing medians are resolution-limited; see its timer-resolution and cold-preparation discussion.\n',encoding='utf-8')
    save_json(OUT/'post_analysis_metadata.json',dict(source='aimrl/tracking_planning/post_analysis.py',source_sha256=sha(__file__),
              added_after_formal_evaluation=True,scope='Descriptive interpretation only; no change to primary calculations, thresholds or policies.',
              protocol_sha256=sha(OUT/'protocol.json'),new_training_interactions=0,formal_replays=0,executed_utc=datetime.now(timezone.utc).isoformat()))
    save_json(OUT/'clock_resolution.json',dict(performance_counter_frequency_hz=frequency.value,period_ns=clock_period,
                native_zero_median='Below clock resolution, not zero execution time',
                compile_seconds=build['elapsed_seconds'],full_study_precompute_and_storage_seconds=startup['elapsed_seconds']))
    print('Wrote measured behavior supplement; no formal replay or training.')


if __name__=='__main__':main()
