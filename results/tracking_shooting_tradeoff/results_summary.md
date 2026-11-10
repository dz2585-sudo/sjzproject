# Tracking and shooting reward trade-offs

No condition reached the prespecified simultaneous target. `distance_0p1`, `close_0p5`, `close_2p0`, `close_5p0` improved both raw C and H, below at least one gate. No condition met all retention gates. `distance_0p3` had the largest positive close gain among weights meeting the distance, accuracy and hit-retention gates (+1.19 pp C, -0.043 D, -0.45% H).

## Matched design

Seven scalar rewards were compared on a fixed 25×25 continuing tracking task with a relative tabular observation, ten actions, a common distance potential, and the same 5M-interaction budget. Three matched training seeds (20, 21, 22) started from zero for each condition. Distance penalty weights were 0.1, 0.3 and 1.0. Close bonus weights were 0.5, 2.0 and 5.0. The reference had zero added weight. The stress set changed both target event probabilities from 0.05 to 0.10. Performance used common post-motion distance and close fraction, hit frequency, and pooled within-seed shot accuracy. The policies shared exact target trajectories on each evaluation split.

The native rollout uses the original NumPy random streams and reset path. Exact Python parity checks covered the zero-weight reference and each reward in a greedy-heavy regime. Bootstrapping used the actual next state at all boundaries. The formal comparison includes every frozen condition. Development selected only the demonstration panels. The final test opened after all 21 runs and the selection record were complete.

## Final results

| Condition | Mean distance D | Close C (%) | Hits H / window | Accuracy A (%) |
| --- | ---: | ---: | ---: | ---: |
| reference | 1.517 ± 0.142 | 58.50 ± 4.73 | 81.33 ± 4.95 | 99.36 ± 0.78 |
| distance_0p1 | 1.423 ± 0.015 | 60.56 ± 1.99 | 84.07 ± 0.10 | 95.61 ± 6.88 |
| distance_0p3 | 1.474 ± 0.066 | 59.69 ± 2.97 | 80.97 ± 5.02 | 99.03 ± 0.43 |
| distance_1p0 | 1.523 ± 0.064 | 65.57 ± 6.62 | 80.62 ± 4.97 | 95.70 ± 6.26 |
| close_0p5 | 1.398 ± 0.021 | 65.12 ± 2.18 | 84.44 ± 0.15 | 95.69 ± 6.03 |
| close_2p0 | 1.355 ± 0.015 | 69.39 ± 0.74 | 81.73 ± 4.98 | 90.61 ± 15.49 |
| close_5p0 | 1.336 ± 0.058 | 72.77 ± 1.90 | 81.52 ± 5.07 | 86.70 ± 11.30 |
| rule | 1.375 | 72.42 | 44.59 | 100.00 |

Learned values are means ± sample SD across three training seeds. Rule is a single fixed policy on the same 500 scenarios. C counts post-motion distance ≤ 1. H counts hits per 300-tick window. A pools hits and shots within seed, then averages seeds. An empty denominator is NA and fails the accuracy screen. The companion tables report shots, misses, original base return, longest far streak, capped first approach, every seed and paired differences. Training returns use different task rewards and are unsuitable for cross-weight ranking.

Reference D=1.517, C=58.50%, H=81.33. The final gates are D ≤ 1.517, C ≥ 63.50%, A ≥ 98%, and H ≥ 82.96 for simultaneous improvement or H ≥ 79.70 for retention.

| Condition | ΔD | ΔC (pp) | ΔH | ΔH (%) | C / D / A gates | H +2% / H −2% gates | D↓ / C↑ / H↑ seeds | Classification |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| distance_0p1 | -0.093 | +2.06 | +2.74 | +3.37 | no / yes / no | yes / yes | 2 / 2 / 1 | Both raw metrics improved below target |
| distance_0p3 | -0.043 | +1.19 | -0.36 | -0.45 | no / yes / yes | no / yes | 2 / 2 / 0 | Close–hit trade-off outside the retention target |
| distance_1p0 | +0.006 | +7.08 | -0.71 | -0.87 | yes / no / no | no / yes | 2 / 2 / 0 | Close–hit trade-off outside the retention target |
| close_0p5 | -0.119 | +6.62 | +3.11 | +3.83 | yes / yes / no | yes / yes | 2 / 2 / 3 | Both raw metrics improved below target |
| close_2p0 | -0.162 | +10.89 | +0.39 | +0.49 | yes / yes / no | no / yes | 3 / 3 / 3 | Both raw metrics improved below target |
| close_5p0 | -0.181 | +14.28 | +0.19 | +0.23 | yes / yes / no | no / yes | 3 / 3 / 3 | Both raw metrics improved below target |

The D–H empirical frontier contains close_5p0, close_2p0, close_0p5. The C–H empirical frontier contains close_0p5, close_2p0, close_5p0. These are separate comparisons of the observed mean points, with Rule included. The lines connecting observed means do not establish a task-wide optimal frontier. See [distance versus hits](figures/distance_hits_tradeoff.png), [close versus hits](figures/close_hits_tradeoff.png), [weight response](figures/weight_response.png), and [development learning curves](figures/learning_curves.png).

## Firing, alignment and motion

Rule fired on 100.00% of aligned ticks, compared with 100.00% for the reference. Rule reached alignment on 14.86% of ticks, versus 27.11% for the reference. The reference therefore recorded +36.74 hits per window relative to Rule, reflecting how frequently each policy reached alignment. `distance_0p3` reached alignment on 26.99% of ticks and fired on 100.00% of those opportunities. It recorded 80.97 hits per window and 59.69% close ticks. `close_5p0` reached alignment on 27.17% of ticks and fired on 100.00% of those opportunities. It recorded 81.52 hits per window and 72.77% close ticks.

Across all final-set policies and seeds, aligned firing was 100%, so the hit decomposition assigns all observed hit changes to how often the crosshair reached alignment. The distance sweep was non-monotonic. The 0.1 penalty produced more aligned opportunities and 4.19 misses per window versus 0.50 for the reference, while 0.3 stayed closest to the reference shot mix and met the accuracy and retention gates. The 1.0 penalty improved C while raising mean D and missing the accuracy gate. The bonus sweep increased C at every weight. The 0.5 bonus alone cleared the D, C and +2% H gates together but fell below the accuracy gate (95.69%). At 5.0, the share of stays fell from 16.00% to 10.54%. Firing at pre-distance one rose from 0.22% to 10.17%. Those shots cannot hit under the existing rule. Average misses rose to 13.98 per window. The larger close bonus improved tracking, with little further hit gain and poorer shot selection.

| Condition | Alignment opportunity (%) | Fire given alignment (%) | Shots / window | Misses / window | Hit ticks outside close (%) | Slow / fast / stay / fire (%) | ΔH from opportunities / decisions |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| reference | 27.11 | 100.00 | 81.83 | 0.50 | 45.34 | 5.3 / 51.4 / 16.0 / 27.3 | +0.00 / +0.00 |
| distance_0p1 | 28.02 | 100.00 | 88.26 | 4.19 | 46.91 | 4.1 / 51.9 / 14.6 / 29.4 | +2.74 / +0.00 |
| distance_0p3 | 26.99 | 100.00 | 81.75 | 0.78 | 45.15 | 5.7 / 51.3 / 15.7 / 27.2 | -0.36 / +0.00 |
| distance_1p0 | 26.87 | 100.00 | 84.59 | 3.97 | 45.59 | 6.0 / 52.0 / 13.8 / 28.2 | -0.71 / +0.00 |
| close_0p5 | 28.15 | 100.00 | 88.49 | 4.05 | 47.55 | 4.5 / 52.1 / 13.9 / 29.5 | +3.11 / +0.00 |
| close_2p0 | 27.24 | 100.00 | 92.55 | 10.83 | 46.23 | 5.9 / 52.3 / 10.9 / 30.9 | +0.39 / +0.00 |
| close_5p0 | 27.17 | 100.00 | 95.50 | 13.98 | 46.59 | 4.2 / 53.4 / 10.5 / 31.8 | +0.19 / +0.00 |
| rule | 14.86 | 100.00 | 44.59 | 0.00 | 74.24 | 50.3 / 33.8 / 1.1 / 14.9 | NA |

The last column decomposes the paired hit change from the reference as `300 × (O−O₀) × F₀ + 300 × O × (F−F₀)` for each seed, then averages those components. O is pre-action alignment opportunity and F is firing conditional on alignment. This is an accounting identity. Policy-dependent changes in O and F can be related. The [action table](action_by_pre_distance.csv) gives separate pre-distance bins 0, 1, 2 and 3+, including denominators. The [fixed diagnostic trace](diagnostics/fixed_final_trace.csv) and [first fragments](diagnostics/first_fragments.csv) preserve the first local overshoot, wait and persistent one-cell offset under a fixed rule. See [alignment and shooting figure](figures/alignment_and_shooting.png) and [post-hit distance distribution](hit_post_distance.csv).

A hit tests the pre-motion target while Fire holds the crosshair fixed. Its post-motion distance equals the target's net displacement, which can be 2 on a speed-two tick or 0 when reflection returns the target to its starting cell. In the reference, 55,462 of 121,996 hit ticks had net displacement above one (45.34%). Every observed window satisfied close ticks + such hits ≤ total ticks. That identity applies to each realized trajectory. The policy can still alter how often it gets aligned, how often it shoots when aligned, and which target-motion contexts it occupies. Because firing and moving are mutually exclusive, a hit can give up the chance to follow the target on the same tick.

### Speed and wall slices

| Condition | Slice | D | C (%) | Hit rate per tick (%) | A (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| reference | speed_1 | 1.300 | 65.04 | 29.51 | 99.50 |
| reference | speed_2 | 1.727 | 52.15 | 24.78 | 99.19 |
| reference | boundary | 1.610 | 58.55 | 25.62 | 99.11 |
| reference | interior | 1.398 | 58.42 | 29.02 | 99.65 |
| distance_0p3 | speed_1 | 1.305 | 67.51 | 29.48 | 98.42 |
| distance_0p3 | speed_2 | 1.637 | 52.10 | 24.57 | 99.78 |
| distance_0p3 | boundary | 1.599 | 59.66 | 25.52 | 98.99 |
| distance_0p3 | interior | 1.314 | 59.72 | 28.85 | 99.07 |
| close_5p0 | speed_1 | 1.074 | 90.46 | 28.91 | 79.04 |
| close_5p0 | speed_2 | 1.590 | 55.62 | 25.48 | 97.86 |
| close_5p0 | boundary | 1.440 | 72.38 | 26.15 | 86.50 |
| close_5p0 | interior | 1.202 | 73.29 | 28.48 | 86.94 |
| rule | speed_1 | 1.223 | 92.13 | 7.25 | 100.00 |
| rule | speed_2 | 1.523 | 53.28 | 22.25 | 100.00 |
| rule | boundary | 1.454 | 72.40 | 16.29 | 100.00 |
| rule | interior | 1.280 | 72.43 | 13.14 | 100.00 |

Speed uses the actual target speed during the tick, after any toggle. A wall slice includes ticks where either object was at most three cells from an edge before acting. Slices are observational. The [full slice summary](slice_summary.csv) includes both splits and every condition, and the [seed-level counts](slice_per_seed.csv) preserve denominators.

## Stress results

| Condition | Mean distance D | Close C (%) | Hits H / window | Accuracy A (%) |
| --- | ---: | ---: | ---: | ---: |
| reference | 1.688 ± 0.120 | 53.33 ± 3.89 | 71.25 ± 4.00 | 99.08 ± 1.00 |
| distance_0p1 | 1.603 ± 0.007 | 55.24 ± 1.60 | 73.36 ± 0.18 | 94.72 ± 7.87 |
| distance_0p3 | 1.618 ± 0.065 | 54.73 ± 2.67 | 70.86 ± 4.18 | 98.28 ± 0.82 |
| distance_1p0 | 1.644 ± 0.086 | 60.25 ± 6.04 | 70.79 ± 3.95 | 95.10 ± 6.75 |
| close_0p5 | 1.587 ± 0.024 | 59.24 ± 1.80 | 73.81 ± 0.37 | 95.37 ± 5.35 |
| close_2p0 | 1.547 ± 0.011 | 63.14 ± 0.75 | 71.65 ± 4.00 | 90.13 ± 15.70 |
| close_5p0 | 1.531 ± 0.057 | 66.42 ± 1.73 | 71.22 ± 3.91 | 86.16 ± 11.71 |
| rule | 1.469 | 68.36 | 42.45 | 100.00 |

## Demonstrations chosen on development

- Distance panel: `distance_0p3`. Passed the development hit/accuracy screen.
- Close panel: `close_5p0`. No family member passed the development hit/accuracy screen.

The four panels use the same target motion and independent crosshair rollouts. Learned panels use training seed 20. Each animation contains the initial frame and all 300 ticks, at nominal 15 fps. Decoding verified every frame and the start, middle and end snapshots. See [4400000](demos/scenario_4400000.gif), [4400001](demos/scenario_4400001.gif), [4400002](demos/scenario_4400002.gif) and [render checks](demos/render_checks.json).

## Execution and next experiment

Formal training took 115.3 seconds including development checkpoints and logging. The Windows process-lifetime peak working set reported during training was 50.4 MiB. The audit verified 350,007 formal training-window reward identities and 231 checkpoint packages. It counted 46,200 development, 11,000 final and 4,400 stress windows. See [execution checks](execution_checks.json), the [frozen protocol](protocol.json), and the [preflight record](preflight.json).

The next isolated test should gate Fire on exact pre-action alignment. With frozen greedy policies, map an unaligned Fire to Stay on a new paired split and test whether it eliminates misses while preserving the full physical trajectories and hit counts. Fire and Stay both hold the crosshair fixed, so this intervention has a precise trajectory-invariance prediction under the existing timing and state-only policies. The 5.0 bonus averaged 13.98 misses per window despite using every aligned opportunity. This would address shot selection and would leave the post-motion distance cost of a valid hit to be studied separately.

## References

- Roijers, Vamplew, Whiteson and Dazeley (2013). *A Survey of Multi-Objective Sequential Decision-Making*. [Publication](https://www.cs.ox.ac.uk/publications/publication9780-abstract.html)
- Van Moffaert and Nowé (2014). *Multi-Objective Reinforcement Learning using Sets of Pareto Dominating Policies*. [Publication](https://jmlr.org/papers/v15/vanmoffaert14a.html)
