# Behavioral and threshold interpretation supplement

This is a descriptive reading of the frozen outputs, added after evaluation. It changes no prespecified method or criterion.

## Final

H1→H2 adds 36.938 hits/window, while C changes -2.099pp. H2→H3 adds another 0.648 hits and +0.237pp C. More horizon does not monotonically increase close time; the large H1 close score comes with much lower hit count.

Adding the learned Q terminal to H3 adds 0.4416 hits and +0.3933pp C beyond the Phi terminal. Relative to R-Decay it changes D -0.1332, C +1.8352pp and H +5.035%. The extra learned terminal gain is smaller than the H1→H2 hit gain, while both are genuine within-family comparisons.

First pre-action alignment averages 10.2648 ticks for R-Decay, 9.4640 for Plan-H3 and 9.1960 for Hybrid. H3-Zero first close averages 21.464 ticks versus 7.326 for H3-Phi, consistent with the distance terminal helping acquisition beyond the short reward horizon.

Hybrid per-model gates: practical 0/5, stronger simultaneous 0/5, absolute C75 with hit retention 5/5. All five same-model D/C/H differences are preserved in paired_differences.csv; the absolute and relative tests must not be substituted for one another.

## Stress

H1→H2 adds 30.510 hits/window, while C changes -3.330pp. H2→H3 adds another 0.805 hits and +0.202pp C. More horizon does not monotonically increase close time; the large H1 close score comes with much lower hit count.

Adding the learned Q terminal to H3 adds 0.5170 hits and +0.5433pp C beyond the Phi terminal. Relative to R-Decay it changes D -0.1914, C +3.0403pp and H +7.031%. The extra learned terminal gain is smaller than the H1→H2 hit gain, while both are genuine within-family comparisons.

First pre-action alignment averages 11.0950 ticks for R-Decay, 9.9250 for Plan-H3 and 9.7120 for Hybrid. H3-Zero first close averages 20.215 ticks versus 7.390 for H3-Phi, consistent with the distance terminal helping acquisition beyond the short reward horizon.

Hybrid per-model gates: practical 0/5, stronger simultaneous 0/5, absolute C75 with hit retention 0/5. All five same-model D/C/H differences are preserved in paired_differences.csv; the absolute and relative tests must not be substituted for one another.

## Speed-two detail

The before-decision speed is observable. The after-toggle speed is the actual motion speed in that tick and retains the historical slicing definition. These are policy-specific descriptive rates; hits per 300 slice ticks are not the main per-window H.

| Split | Speed definition | Method | D difference vs R-Decay | C difference | Slice hit rate, R→method |
| --- | --- | --- | --- | --- | --- |
| final | pre_speed_2 | Plan-H1 | -0.2296 | +3.617 pp | 81.387 → 80.200 |
| final | pre_speed_2 | Plan-H2 | -0.1707 | +1.581 pp | 81.387 → 84.984 |
| final | pre_speed_2 | Plan-H3 | -0.1819 | +1.966 pp | 81.387 → 85.850 |
| final | pre_speed_2 | Hybrid-H3 | -0.1973 | +2.634 pp | 81.387 → 86.371 |
| final | post_speed_2 | Plan-H1 | -0.1866 | +0.954 pp | 81.704 → 77.309 |
| final | post_speed_2 | Plan-H2 | -0.1675 | +1.577 pp | 81.704 → 85.410 |
| final | post_speed_2 | Plan-H3 | -0.1766 | +1.851 pp | 81.704 → 86.236 |
| final | post_speed_2 | Hybrid-H3 | -0.1915 | +2.453 pp | 81.704 → 86.819 |
| stress | pre_speed_2 | Plan-H1 | -0.3652 | +6.633 pp | 69.962 → 67.222 |
| stress | pre_speed_2 | Plan-H2 | -0.2799 | +3.422 pp | 69.962 → 75.252 |
| stress | pre_speed_2 | Plan-H3 | -0.2867 | +3.711 pp | 69.962 → 75.909 |
| stress | pre_speed_2 | Hybrid-H3 | -0.3025 | +4.546 pp | 69.962 → 76.409 |
| stress | post_speed_2 | Plan-H1 | -0.2914 | +1.872 pp | 70.727 → 63.319 |
| stress | post_speed_2 | Plan-H2 | -0.2705 | +3.395 pp | 70.727 → 75.891 |
| stress | post_speed_2 | Plan-H3 | -0.2717 | +3.325 pp | 70.727 → 76.608 |
| stress | post_speed_2 | Hybrid-H3 | -0.2861 | +4.053 pp | 70.727 → 77.153 |

Both Plan-H3 and Hybrid improve speed-two D/C and hit rates under both speed definitions in both splits. Thus their close gains are not purchased by losing hits in this slice. This is evidence of useful empirical room within the original action set, under the additional full-state/known-model/compute conditions.

## Movement and firing behavior

| Split | Method | Fast | Slow | Stay | Fire | Cross current target axis |
| --- | --- | --- | --- | --- | --- | --- |
| final | R-Decay | 68.019% | 4.616% | 0.235% | 27.129% | 57.277% |
| final | Plan-H1 | 65.418% | 7.006% | 0.842% | 26.733% | 55.545% |
| final | Plan-H2 | 62.378% | 8.389% | 0.905% | 28.328% | 53.782% |
| final | Plan-H3 | 62.547% | 7.777% | 1.059% | 28.617% | 53.863% |
| final | Hybrid-H3 | 63.585% | 6.466% | 1.158% | 28.790% | 54.476% |
| stress | R-Decay | 68.270% | 8.092% | 0.317% | 23.321% | 51.629% |
| stress | Plan-H1 | 68.016% | 8.501% | 1.076% | 22.407% | 52.891% |
| stress | Plan-H2 | 62.530% | 11.164% | 1.222% | 25.084% | 48.803% |
| stress | Plan-H3 | 62.899% | 10.413% | 1.385% | 25.303% | 48.843% |
| stress | Hybrid-H3 | 64.463% | 8.617% | 1.450% | 25.470% | 49.630% |

The movement table records mixtures, not a causal explanation. Crossing the current target coordinate on the exact same row/column can be deliberate anticipation of target motion, so fewer crossings alone do not prove less harmful overshoot. Common-state actions and paired first-event fragments are provided to inspect that distinction.

There were 0 qualifying aligned R-Fire→planner-movement examples and 10 explicit absent method/split searches. In this run none occurred for any planned family in either split. Every formal policy had fire_given_alignment=1 and aligned_move_fraction=0. Therefore the observed gains do not come from deliberately postponing a currently valid shot; they come from movement/approach decisions before the next alignment opportunity.

For instance, the frozen earliest-event selection can show a worse planning recovery, and those clips are retained. No event is chosen for positive performance. Same-state proposals and each controller’s own evolving physical states remain separately labelled.

## Next intervention

A 2-cell movement experiment is worth doing as a separate action-design factor. Hybrid improves the current speed-two frontier but leaves substantially lower close time at speed two than at speed one. Existing moves are one or three cells and all methods still Fire immediately on alignment; a two-cell option could change how the next opportunity is reached without changing shot timing. This remains a hypothesis, not a proven source of all remaining loss. Preserve this stochastic planning baseline and use a controlled action-set comparison. A longer horizon and probability-calibrated planning are also distinct clean follow-ups; none of these measurements establishes a theoretical limit.

All primary relative +5pp gates fail at the family-mean level. Final Hybrid passes absolute C≥75% while retaining hits, but that is not the +5pp gate. Final H1 also passes C≥75% but retains only about 60% of R-Decay hits, so it is not joint improvement. Stress has no absolute 75% pass. No controller was dropped from reporting.

## Cold preparation and timer resolution

Native compilation took 20.8836s. Full-study precomputation and storage took 26.2726s; charged together, these are 47.1562s before loading/querying the delivered cached controllers. This is the complete study build, not a claimed minimum startup time for one controller.

The native performance counter period on this host is 100.0ns. The recorded native median of 0ns is below measurement resolution, not zero computation. Approximately 100/101ns P95 bins likewise must not be read as nanosecond precision or an end-to-end latency result. Native maxima include scheduling interruptions. Keep raw timings for audit, and use the separately measured Python observation API below for a fuller call cost. Neither first API lookup nor second pass flushes OS caches.

| Policy | Load seconds | First API ns | Warm median ns | Warm P95 ns | Warm max ns |
| --- | --- | --- | --- | --- | --- |
| Plan-H1 | 0.003883 | 8000 | 2300.0 | 2400.0 | 37500 |
| Plan-H2 | 0.003951 | 7900 | 2300.0 | 2400.0 | 6600 |
| Plan-H3 | 0.004333 | 6300 | 2300.0 | 2400.0 | 33300 |
| Plan-H3-Zero | 0.004185 | 6200 | 2300.0 | 2400.0 | 5400 |
| Hybrid-H3_s60 | 0.016443 | 8400 | 2300.0 | 2400.0 | 59200 |
| Hybrid-H3_s61 | 0.015747 | 6500 | 2300.0 | 2400.0 | 27100 |
| Hybrid-H3_s62 | 0.015793 | 7100 | 2300.0 | 2400.0 | 5500 |
| Hybrid-H3_s63 | 0.015938 | 6900 | 2300.0 | 2400.0 | 148900 |
| Hybrid-H3_s64 | 0.015966 | 7300 | 2300.0 | 2400.0 | 37400 |

For equal timing boundaries, paired_api_latency.csv additionally queries the first 512 previously frozen common states twice, with tuple conversion outside both timing intervals. The original diagnostics/api_latency.csv first pass excludes conversion while its second pass includes saved-array-to-tuple conversion, so that difference must not be treated as a cache-speedup estimate. Neither benchmark flushes OS caches or repeats an environment evaluation. Full precomputation remains separately charged.

An absent pre-action alignment recovery is stored as null and censored at the 30-tick horizon or window end, never scored as a zero-tick successful recovery.
