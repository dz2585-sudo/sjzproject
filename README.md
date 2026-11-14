# AimRL: Grid Aiming and Continuous Tracking

A CPU-based reinforcement-learning research project for ELEN E6885. The project studies how reward shaping and state representation affect learning to aim at and continuously follow a moving target. The simulator supplies coordinates directly. The original learned controller is evaluated separately from the inference-only firing guard.

**Current status:** the boundary-stability study completed 100M formal + 4M pilot interactions. The zero-training short-horizon planning study completed 20 policy instances on new Final/Stress scenarios; Hybrid reaches the 75% Final close target while retaining hits, but the relative +5pp joint gate remains unmet. The completed two-cell action ablation improves D/C/H without training; Final exceeds 75% close, Stress and the relative +5pp gate remain unmet. Fourteen-action training is future work.

[Research index](docs/research_index.md) · [Two-cell actions](#two-cell-action-resolution-ablation) · [Boundary stability](#boundary-information-and-learning-rate-stability) · [Short-horizon planning](#short-horizon-planning-with-learned-terminal-values) · [Representation benchmark](#representation-benchmark) · [Reward trade-offs](#trackingshooting-reward-trade-offs) · [Firing guard](#q-learning-with-a-firing-guard) · [Progress, step by step](#progress-step-by-step) · [Demonstrations](#demonstrations) · [Artifact locations and path changes](#artifact-locations-and-path-changes) · [Installation and verification](#installation-and-verification) · [Limitations and next work](#limitations-and-next-work)

The [detailed research history](docs/project_history.md) records the reasoning behind each change. The [artifact guide](docs/artifact_paths.md) explains exact file locations, historical-to-current path mappings, and how to open the demonstrations.

Earlier full representation-study rollout (historical reference; its different budget is described below):

![Full continuous-tracking comparison: archived Potential, Relative Base, Relative Potential, and predictive rule](results/tracking_improvement/render/seed3601000_main_comparison.gif)

## Boundary information and learning-rate stability

**Question.** Does omitted boundary context or constant update size limit sustained tracking? Four conditions cross the original relative observation (already including heading and speed) or boundary-group augmentation with fixed alpha=0.15 or per-pair decay `0.15*(1+n/1000)^(-0.6)`. Rewards, actions, exploration and 5M budget are held fixed. Five paired training seeds (60–64) give **100,000,000 formal interactions**, plus an independent **4,000,000-interaction pilot**.

R-Decay improves close tracking and cross-seed C stability. The much larger boundary table performs worse at the same budget; those negative results remain included. Neither the relative joint improvement gate nor the absolute C≥75% target is reached in this study. Guarded results below are recalculated from raw windows; entries are mean ± sample SD across five trained seeds.

| Split | Method | Instances | D ↓ | C (%) ↑ | H / 300 ticks ↑ | A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Final | R-Fixed | 5 | 1.3774 ± 0.0131 | 68.899 ± 1.561 | 84.4564 ± 0.0887 | 100.00 ± 0.00 |
| Final | R-Decay | 5 | 1.3433 ± 0.0099 | 73.185 ± 0.042 | 84.7308 ± 0.1011 | 100.00 ± 0.00 |
| Final | B-Fixed | 5 | 2.0311 ± 0.0767 | 56.384 ± 1.824 | 60.8400 ± 2.5377 | 100.00 ± 0.00 |
| Final | B-Decay | 5 | 1.9200 ± 0.0165 | 57.825 ± 0.482 | 69.7932 ± 0.4381 | 100.00 ± 0.00 |
| Stress | R-Fixed | 5 | 1.5680 ± 0.0123 | 62.447 ± 1.254 | 73.8980 ± 0.2721 | 100.00 ± 0.00 |
| Stress | R-Decay | 5 | 1.5441 ± 0.0205 | 66.349 ± 0.197 | 74.6440 ± 0.1263 | 100.00 ± 0.00 |
| Stress | B-Fixed | 5 | 2.2666 ± 0.0609 | 50.280 ± 1.881 | 52.2820 ± 2.1788 | 100.00 ± 0.00 |
| Stress | B-Decay | 5 | 2.1812 ± 0.0628 | 51.788 ± 0.351 | 59.6140 ± 0.3557 | 100.00 ± 0.00 |

![Boundary stability: paired training seeds](results/tracking_boundary_stability/figures/paired_seeds.png)

![Boundary stability: all 301 frames, fixed scenario 5400000](results/tracking_boundary_stability/demos/scenario_5400000.gif)

Each GIF has initial state plus all 300 ticks (301 frames, 20.07 s). Programmatic decoding is verified; actual-player inspection was not performed. [Study navigation](docs/tracking_boundary_stability.md) · [report](results/tracking_boundary_stability/results_summary.md) · [all data](results/tracking_boundary_stability/) · [other GIFs](results/tracking_boundary_stability/demos/) · [publication reproduction commands](docs/reproduce_latest_studies.md).

## Short-horizon planning with learned terminal values

**Question.** How much can stochastic feedback planning improve the existing ten-action task, and what comes from horizon versus terminal value? The study reuses frozen models: **zero new training interactions and zero Q updates**. The **20 instances** are Rule (1), R-Fixed (5), R-Decay (5), Phi-terminal Plan-H1/H2/H3 (3), zero-terminal Plan-H3-Zero (1), and learned-terminal Hybrid-H3 (5). Final has **10,000 windows**; Stress has **4,000**, totaling **4,200,000 environment ticks**.

Planners observe full coordinates and a known stochastic model; R uses relative state. These information and computation advantages accompany cross-family differences. H1→H2→H3 within the same Phi family isolates horizon. Zero removes the distance terminal; Hybrid uses `max(Q_shaped)+Phi`, where `Phi=-distance` and task value = shaped value + Phi. In Stress the real turn/toggle probabilities are **0.10**, while every planning model remains **0.05**.

| Split | Method | Instances | D ↓ | C (%) ↑ | H / 300 ticks ↑ | A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Final | Rule | 1 | 1.3849 | 72.741 | 43.4080 | 100.00 |
| Final | R-Fixed | 5 | 1.3848 ± 0.0137 | 68.898 ± 1.538 | 84.2240 ± 0.2003 | 100.00 ± 0.00 |
| Final | R-Decay | 5 | 1.3465 ± 0.0091 | 73.311 ± 0.078 | 84.5276 ± 0.0611 | 100.00 ± 0.00 |
| Final | Plan-H1 | 1 | 1.2990 | 76.615 | 50.7560 | 100.00 |
| Final | Plan-H2 | 1 | 1.2285 | 74.516 | 87.6940 | 100.00 |
| Final | Plan-H3 | 1 | 1.2219 | 74.753 | 88.3420 | 100.00 |
| Final | Plan-H3-Zero | 1 | 2.4685 | 70.961 | 83.8020 | 100.00 |
| Final | Hybrid-H3 | 5 | 1.2134 ± 0.0025 | 75.147 ± 0.026 | 88.7836 ± 0.0158 | 100.00 ± 0.00 |
| Stress | Rule | 1 | 1.4827 | 67.910 | 41.6050 | 100.00 |
| Stress | R-Fixed | 5 | 1.5752 ± 0.0103 | 62.481 ± 1.275 | 73.9950 ± 0.2589 | 100.00 ± 0.00 |
| Stress | R-Decay | 5 | 1.5527 ± 0.0207 | 66.335 ± 0.216 | 74.4570 ± 0.0391 | 100.00 ± 0.00 |
| Stress | Plan-H1 | 1 | 1.3930 | 71.960 | 47.8600 | 100.00 |
| Stress | Plan-H2 | 1 | 1.3777 | 68.630 | 78.3700 | 100.00 |
| Stress | Plan-H3 | 1 | 1.3709 | 68.832 | 79.1750 | 100.00 |
| Stress | Plan-H3-Zero | 1 | 2.4651 | 65.675 | 75.2500 | 100.00 |
| Stress | Hybrid-H3 | 5 | 1.3613 ± 0.0053 | 69.375 ± 0.076 | 79.6920 ± 0.0445 | 100.00 ± 0.00 |

On this new Final, Hybrid-H3 has D=1.2134, C=75.15%, H=88.784; H is **5.04%** above the same-scene R-Decay mean. All five Hybrid instances reach C≥75% and retain at least 98% of their paired baseline hits. Stress Hybrid C is 69.38%, below 75%. The prespecified relative gate requiring **C gain ≥5 percentage points** still fails. H1's high close score comes with substantial hit loss and is not joint improvement. Pure Rule/planners are single instances and have no invented training-seed SD.

**Different holdouts:** boundary-study R-Decay Final C is 73.19%; planning-study R-Decay Final C is 73.31%. Differences are computed within each study from unrounded counts, never across these tables.

Planning preparation is separately charged in the precomputation record: state backups, chance terms and action-value lookups are distinct from real environment ticks. Formal online use is 1,890,000 cached planning queries, not training updates. Full measured preparation/evaluation costs and timer-resolution caveats are in the report.

![Planning: horizon response with fixed state, model and terminal](results/tracking_planning/figures/horizon_response.png)

![Planning: all 301 frames, fixed scenario 6400000](results/tracking_planning/demos/scenario_6400000/synchronized.gif)

Programmatic decoding verified all 301 frames and their timing; actual-player inspection was not performed and no desktop/browser permission was requested. [Study navigation](docs/tracking_planning.md) · [report](results/tracking_planning/results_summary.md) · [paired results](results/tracking_planning/paired_differences.csv) · [other GIFs](results/tracking_planning/demos/) · [reproduce](docs/reproduce_latest_studies.md).

**Follow-up completed:** see the [two-cell action ablation](#two-cell-action-resolution-ablation) on independent scenes. A separate fourteen-action learning study remains future work; these results do not establish a theoretical ceiling.

## Two-cell action resolution ablation

**Question.** With the same full physical state, known model, task reward, three-step feedback planning and terminal value, do four additional one-tick two-cell moves improve tracking while retaining hits? A14 preserves original action IDs 0–9 and appends left/right/up/down medium (IDs 10–13). New actions are allowed at all three depths. Hybrid uses the same five frozen ten-action R-Decay models (seeds 60–64), without inventing Q entries for new actions. **New research training interactions=0; frozen Q updates=0.**

All controls below use this study's new, paired scenes, not the earlier planning holdout. Pure Plan has one controller per action set and **no training-seed SD**; Hybrid reports five source-model means ± **sample SD (ddof=1)**. C/A and their SDs are expressed in percent/percentage-point units. A is pooled hits/shots within each source model, then averaged across models.

| Split | Controller | Instances | D | C (%) | H / 300 ticks | A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Final | Plan-H3-A10 | 1 | 1.2120 | 74.6447 | 88.3700 | 100.0000 |
| Final | Plan-H3-A14 | 1 | 1.1164 | 75.5280 | 111.8980 | 100.0000 |
| Final | Hybrid-H3-A10 | 5 | 1.2023 ± 0.0034 | 75.0547 ± 0.0492 | 88.8128 ± 0.0227 | 100.0000 ± 0.0000 |
| Final | Hybrid-H3-A14 | 5 | 1.0745 ± 0.0004 | 76.2769 ± 0.0037 | 112.7332 ± 0.0041 | 100.0000 ± 0.0000 |
| Stress | Plan-H3-A10 | 1 | 1.3487 | 69.6750 | 79.6100 | 100.0000 |
| Stress | Plan-H3-A14 | 1 | 1.2240 | 71.7367 | 103.4400 | 100.0000 |
| Stress | Hybrid-H3-A10 | 5 | 1.3422 ± 0.0052 | 70.1420 ± 0.0494 | 79.9540 ± 0.0778 | 100.0000 ± 0.0000 |
| Stress | Hybrid-H3-A14 | 5 | 1.1794 ± 0.0009 | 72.3987 ± 0.0147 | 104.2820 ± 0.0135 | 100.0000 ± 0.0000 |

| Split | Terminal | ΔD | ΔC (pp) | ΔH | H change (%) | A14 C (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Final | Plan | -0.095613 | +0.8833 | +23.5280 | +26.6244 | 75.5280 |
| Final | Hybrid | -0.127820 | +1.2223 | +23.9204 | +26.9335 | 76.2769 |
| Stress | Plan | -0.124733 | +2.0617 | +23.8300 | +29.9334 | 71.7367 |
| Stress | Hybrid | -0.162763 | +2.2567 | +24.3280 | +30.4275 | 72.3987 |

**Aggregation:** Hybrid H change is `100*(mean_m H_A14,m / mean_m H_A10,m - 1)`, a **ratio of means**. Per-model ratios are separate paired results; their arithmetic mean is not substituted. All differences use unrounded raw counts. The [publication recomputation](provenance/action_resolution_release/recalculated/raw_recomputed.json) records both formulas explicitly.

D, C and H improve under both terminals; hits increase about **27%–30%**, with A=100%. All five Hybrid model pairs improve D/C/H in both splits. Final Plan-A14 and all five Hybrid-A14 instances attain absolute C≥75% and retain hits. Stress remains below 75%. Both relative joint gates still **fail ΔC≥5 percentage points**, despite passing the hit requirements. Value monotonicity passes all 56,250,000 state/terminal/depth comparisons (fixed tolerance 1e-9, minimum difference 0, zero violations); it does not guarantee each rolling trajectory or metric improves at every tick.

The largest hit gain is associated with **speed-1 alignment control**: Hybrid Final medium use is 41.90% of observable-speed-1 decisions versus 2.80% at speed 2; overall use is 22.24%. Stay decreases from 13.65% to 2.10%. Speed-2 slices also improve. Crossing the *current* target coordinate increases (about +6.80pp), while immediate reversals decrease; this is not a general reduction of overshoot. All 284,127 formal medium commands move an actual two cells; clipped 1/0 cases remain supported and tested. [Fixed-rule regressions and both real routes](results/tracking_action_resolution/behavior_fragments.csv) are retained.

Final: **6,000 windows**; Stress: **2,400**, totaling **2,520,000 environment ticks**. Development is separately 2,400 windows/720,000 ticks; demos 12/3,600. Offline preparation is separately 112,500,000 backups, 657,000,000 chance terms and 1,350,000,000 action-value lookups; none are training interactions. Full-state information, known dynamics and extra planning computation are part of this result. Stress real probabilities are 0.10 while the planner stays at 0.05.

![Two-cell action ablation: D/C/H](results/tracking_action_resolution/figures/D_C_H.png)

![Four panels, initial state and all 300 ticks](results/tracking_action_resolution/demos/scenario_7400000/synchronized.gif)

Each GIF has 301 frames and 20,070 ms. Full pixel/timing decoding is verified; **actual-player inspection was not performed**. [Study guide](docs/tracking_action_resolution.md) · [frozen English report](results/tracking_action_resolution/results_summary.md) · [behavior interpretation](results/tracking_action_resolution/behavior_interpretation.md) · [paired models figure](results/tracking_action_resolution/figures/paired_models.png) · [action-use figure](results/tracking_action_resolution/figures/action_usage.png) · [high-speed events](results/tracking_action_resolution/figures/high_speed_events.png) · [all data](results/tracking_action_resolution/) · [other GIFs](results/tracking_action_resolution/demos/) · [release-safe reproduction](docs/tracking_action_resolution.md#reproduction-and-publication-audits).

**Next work:** a separately frozen fourteen-action training study is justified; **no fourteen-action Q-learning has been trained here**. The present evidence concerns extra control options under full-state model-based planning with a ten-action learned terminal. These holdouts and thresholds will not be retuned.

## Representation benchmark

### Matched representation study

All four methods use the same 25 by 25 continuous-tracking environment, 300-tick windows, ten actions, five-million-interaction budget, and exploration schedule. Training seeds are 10, 11, and 12. Each final model is evaluated on the same 500 independent final-test scenarios. Values below are **mean ± sample standard deviation across three training seeds**, not confidence intervals.

Distance is Manhattan distance, `|xc-xt| + |yc-yt|`, measured after target motion. Close means distance <=1; far means distance >5. The initial animation frame is excluded from tick-weighted metrics. Hits are counted per 300-tick window; shot accuracy is pooled as total hits / total shots within each training seed before averaging across seeds.

| Method | Mean L1 distance (cells) | Close fraction (%) | Far fraction (%) | Hits / 300 ticks | Shot accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| Absolute Base | 16.469 ± 0.105 | 0.848 ± 0.037 | 91.502 ± 0.384 | 0.177 ± 0.012 | 1.200 ± 0.132 |
| Absolute Potential | 7.531 ± 0.064 | 25.808 ± 0.351 | 52.509 ± 0.618 | 21.873 ± 0.235 | 74.913 ± 0.447 |
| Relative Base | 1.502 ± 0.034 | 61.655 ± 4.022 | 2.521 ± 0.269 | 84.442 ± 0.130 | 99.396 ± 0.431 |
| Relative Potential | 1.440 ± 0.022 | 60.540 ± 1.951 | 2.338 ± 0.147 | 84.489 ± 0.116 | 99.493 ± 0.304 |

**Interpretation.** Relative position sharing improved both reward modes compared with their matched absolute-state controls. Relative Potential has lower mean distance and far fraction than Relative Base, but a lower close fraction; their hit counts are similar. This is evidence for a useful state-representation choice in this experiment, not a claim that shaping wins on every metric.

The relative table contains 192,080 Q entries and 768,320 float32 element bytes, compared with 31,250,000 entries and 125,000,000 bytes for the absolute table. The approximately 163-fold reduction describes table storage only. Relative offsets omit absolute wall information, so this controller uses an approximate observation aggregation rather than an exact Markov state.

**Data:** [cross-seed statistics and target checks](results/tracking_improvement/evaluation/cross_seed_summary.json), [all final-test records](results/tracking_improvement/evaluation/final/), and [all per-seed results and interpretation](results/tracking_improvement/results_summary.md).

![Final-test comparison for the representation study](results/tracking_improvement/render/final_comparison.png)

### See the current behavior


This actual rollout contains the initial frame and all 300 ticks: **301 frames, 20.07 seconds**. Red circles are targets and blue crosses are crosshairs. All panels see the same target trajectory; hits do not stop the animation. The new learned panels use the predeclared training seed 10. The archived Potential panel uses seed 0 and 20 million training interactions, while the new models use five million; this mixed-budget display illustrates behavior. The table above is the equal-budget comparison.

[Scenario 3601000](results/tracking_improvement/render/seed3601000_main_comparison.gif) · [Scenario 3601001](results/tracking_improvement/render/seed3601001_main_comparison.gif) · [Scenario 3601002](results/tracking_improvement/render/seed3601002_main_comparison.gif) · [Six-policy comparison](results/tracking_improvement/render/seed3601000_six_policy_comparison.gif)

### References on the same final-test scenarios

These references were evaluated once each and have no training-seed standard deviation. The archived model has a different training budget and schedule.

| Reference | Mean distance | Close (%) | Far (%) | Hits/window | Accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| Archived Absolute Base: 20M, seed 0 | 16.277 | 2.707 | 89.373 | 2.722 | 7.334 |
| Archived Absolute Potential: 20M, seed 0 | 1.651 | 68.169 | 3.847 | 70.770 | 90.575 |
| Predictive rule: untrained | 1.371 | 73.041 | 1.649 | 43.730 | 100.000 |
| Random: untrained negative control | 16.977 | 0.648 | 92.327 | 0.042 | 0.140 |

The predictive rule stays closer to the target, while the relative learned policies obtain more hits. Firing uses an action tick and is checked before target motion; the target then moves while the crosshair stays in place. Accurate shooting and post-motion proximity therefore measure different aspects of control.

### Stress results and practical targets

The independent stress split doubles target turn and speed-toggle probabilities from 0.05 to 0.10, keeping the other mechanics unchanged. Each final model is evaluated on 200 shared scenarios. Values are again mean ± sample SD across three training seeds.

| Method | Mean L1 distance (cells) | Close fraction (%) | Far fraction (%) | Hits / 300 ticks | Shot accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| Absolute Base | 16.494 ± 0.058 | 0.722 ± 0.033 | 92.048 ± 0.071 | 0.118 ± 0.039 | 0.772 ± 0.253 |
| Absolute Potential | 8.401 ± 0.039 | 18.448 ± 0.144 | 59.113 ± 0.090 | 15.003 ± 0.340 | 66.249 ± 0.747 |
| Relative Base | 1.673 ± 0.028 | 55.963 ± 3.481 | 2.901 ± 0.370 | 73.648 ± 0.331 | 99.067 ± 0.702 |
| Relative Potential | 1.622 ± 0.019 | 54.999 ± 1.571 | 2.746 ± 0.158 | 73.710 ± 0.111 | 99.100 ± 0.638 |

Relative Potential passed the predeclared practical comparison against matched Absolute Potential: at least 10% lower mean distance, at least five percentage points more close tracking, no higher far fraction, at least 90% of control hits, and at least 85% accuracy. The separately assessed absolute targets are:

| Predeclared absolute target | Measured mean | Status |
| --- | --- | --- |
| Mean distance <=1.5 cells | 1.440 | Achieved |
| Close fraction >=75% | 60.540 | **Not achieved** |
| Far fraction <=5% | 2.338 | Achieved |
| Mean capped acquisition <=15 ticks | 8.287 | Achieved |

The first-acquisition metric uses the first post-motion distance <=1; failures are capped at 300 with a separate acquired flag. The missed close-tracking target remains visible rather than being replaced after evaluation. Stress records are in [evaluation/stress](results/tracking_improvement/evaluation/stress/).

## Progress, step by step

Each item states the motivation, implementation, observed result, and resulting research decision. These are completed research milestones, not claims that each experiment measured the same task.

| Milestone | Main question | Completed output | Result and next decision |
|---|---|---|---|
| 1. Feasibility and specification | Can a small aiming study run reproducibly on a local CPU? | Configuration, package structure, setup checks, and explicit environment/learning rules | A 30,000-byte small-grid Q table made a bounded first experiment practical |
| 2. Environment and baselines | Do movement, firing, termination, and random seeds work correctly? | Grid environment, rule/random policies, automated rendering, and environment tests | Rule success 100%; random success 25% in the initial diagnostic |
| 3. Basic Q-learning | Can an agent learn aiming through interaction? | Training, isolated validation, saved models, and reload checks | Moving-target validation reached 100%; the limited static diagnostic remained weak |
| 4. Reward shaping | Can distance information accelerate learning? | Potential-based reward transformation and a matched development comparison | Shaping reached the success threshold earlier in the single-seed comparison |
| 5. Small-grid benchmark | Does the observation persist across training seeds? | Six formal runs and independent tests at three movement probabilities | Both methods reached 100% final success; shaping usually used fewer steps |
| 6. Larger-grid development | Does learning remain effective when the table grows? | Two 25 by 25 trials and a real rollout GIF | Base success 6%; Potential 100%; one-hit success did not measure sustained tracking |
| 7. Continuous tracking | Can the agent follow wider motion, turning, and speed changes? | Continuing windows, two target speeds, slow/fast actions, and full-length GIFs | Potential tracked effectively; the absolute Base policy remained weak |
| 8. Representation study | Can shared relative-position values improve finite-budget learning? | Four methods by three seeds, coverage diagnostics, final/stress tests, and four GIFs | Both relative methods improved sharply; close tracking and boundary behavior still need work |
| 9. Publication and evidence organization | Can every displayed result be found and checked? | Descriptive archives, compressed models, exact provenance, and publication verification | Code, reports, numerical records, and animations are available in the repository |
| 10. Progress and presentation index | Can a reader follow the reasoning and find every result from one page? | This overview, complete decision history, and path/GIF guide | Each study is linked to its metrics, figures, models, and limitations |
| [Tracking–shooting reward trade-offs](docs/tracking_shooting_tradeoff.md) | Can proximity weights improve tracking and hits under the same mechanics? | 21 matched runs, 105M updates, two held-out splits, and three demonstrations | Some weights improve proximity and hits. Original shot accuracy prevents a full simultaneous pass. |
| [Q-learning with a firing guard](docs/tracking_fire_guard.md) | Can a deterministic guard remove misses without altering tracking? | Paired replay of 21 frozen models on new Final and Stress scenarios, with zero training | The guard preserves motion and hits. Guarded close bonus 0.5 passes the full mean Final screen. |

### 1. Feasibility and specification

**Reasoning.** Start with a transparent, inexpensive task whose transitions, rewards, and resource use can be checked before increasing difficulty. Direct coordinate observations keep the initial research question focused on control and learning.

**Implemented.** A configurable 5 by 5 environment specification, six actions, a 50-step cap, Q-learning settings, dependency checks, and memory-budget checks. The table has 1,250 state combinations and 7,500 float32 Q entries. The initial setup created the project structure; it did not train a policy.

**Evidence:** [specification](spec.md), [base configuration](configs/base.json), [setup checker](scripts/check_setup.py), and [dependencies](requirements.txt).

### 2. Environment and baselines

**Reasoning.** Establish observable behavior before attributing any outcome to learning. A transparent rule policy checks task feasibility; Random supplies an untrained negative control.

**Implemented.** Explicit reset/step lifecycle, movement clipping, pre-target-motion firing, horizontal target movement, hit termination, timeout truncation, rendering, and independent random-policy sampling. Initial diagnostics used 100 scenarios, seeds 300000 through 300099.

| Policy | Success within 50 steps | Mean steps | Mean misses |
| --- | --- | --- | --- |
| Rule | 100.0% | 5.07 | 0.00 |
| Random | 25.0% | 43.77 | 6.89 |

These are environment diagnostics, not the later final test. [Rule records](results/environment_baselines/rule_policy_diagnostic.csv), [random records](results/environment_baselines/random_policy_diagnostic.csv), and the [environment preview](results/environment_baselines/rule_policy_preview.png) preserve the evidence.

### 3. Basic Q-learning

**Reasoning.** Let the agent discover useful actions through reward and state transitions. Save the learned table and isolate validation so evaluating progress cannot alter training.

**Implemented.** Epsilon-greedy training, Q updates, checkpoint evaluation, model persistence, and deterministic evaluation. A true hit stops bootstrap in this hit-once task; external truncation uses the actual next state.

| Interactions | Validation success | Mean steps | Mean misses |
| --- | --- | --- | --- |
| 0 | 0.0% | 50.000 | 0.000 |
| 10,000 | 70.0% | 26.605 | 1.300 |
| 30,000 | 99.0% | 7.490 | 0.115 |
| 100,000 | 100.0% | 5.055 | 0.000 |

The moving-target run used training seed 0 and 100,000 interactions. A separate static-target diagnostic used seed 42 and only 10,000 interactions, reaching 13.5% validation success. Subsequent replay found repeated-state/time-out behavior; it did not establish a single cause of that diagnostic's low score.

**Next decision.** High success in a small arena can saturate quickly. Compare learning speed and interaction cost as well as final success.

**Evidence:** [validation CSV](results/q_learning_development/base_seed0/validation.csv), [learning curve](results/q_learning_development/base_seed0/learning_curve.png), [model](results/q_learning_development/base_seed0/q_table.npy), and [static diagnostic note](results/reward_shaping_development/static_diagnostic_note.md).

### 4. Reward-shaping development

**Reasoning.** Sparse hit rewards give little direct guidance when the crosshair is far away. Add a distance-based potential difference while leaving the environment's base reward and action rules intact.

The transformation is `r_train = r_base + gamma * Phi(next_state) - Phi(state)`, where `Phi = -L1_distance`. It is applied once before each update. Both methods start with zero Q values.

| Interactions | Base: success / steps / misses | Potential: success / steps / misses |
| --- | --- | --- |
| 0 | 0.0% / 50.000 / 0.000 | 0.0% / 50.000 / 0.000 |
| 10,000 | 70.0% / 26.605 / 1.300 | 100.0% / 6.990 / 0.015 |
| 30,000 | 99.0% / 7.490 / 0.115 | 100.0% / 4.935 / 0.100 |
| 100,000 | 100.0% / 5.055 / 0.000 | 100.0% / 4.670 / 0.000 |

Each method used 100,000 interactions with seed 0. The first checkpoint in two consecutive validations at or above 90% success was 20,000 for Base and 10,000 for Potential. Checkpoints are 10,000 steps apart, so this does not measure an exact speedup factor.

**Next decision.** Freeze the protocol and repeat across multiple training seeds before treating this development observation as a stable comparison.

**Evidence:** [comparison data](results/reward_shaping_development/comparison/comparison.csv), [comparison figure](results/reward_shaping_development/comparison/comparison.png), and [frozen benchmark protocol](results/reward_shaping_development/protocol_frozen.json).

### 5. Small-grid formal benchmark

**Reasoning.** Separate development from final evaluation and expose between-run variation. The rule baseline remains informative even if a learned method reaches perfect success.

**Implemented.** Base and Potential, each with training seeds 0, 1, and 2 and 100,000 interactions per run: 600,000 total. Final evaluation used 500 scenarios per model and movement probability, totaling 9,000 learned-policy episodes and 3,000 baseline episodes. Models were fixed before final testing.

| Movement probability | Policy | Success | Mean steps | Mean misses |
| --- | --- | --- | --- | --- |
| 0.10 | Base | 100.0% ± 0.0% | 5.973 ± 0.414 | 0.000 ± 0.000 |
| 0.10 | Potential | 100.0% ± 0.0% | 5.243 ± 0.089 | 0.018 ± 0.025 |
| 0.10 | Rule | 100.0% | 4.576 | 0.000 |
| 0.10 | Random | 23.2% | 44.012 | 7.416 |
| 0.25 | Base | 100.0% ± 0.0% | 5.234 ± 0.113 | 0.000 ± 0.000 |
| 0.25 | Potential | 100.0% ± 0.0% | 4.885 ± 0.061 | 0.011 ± 0.012 |
| 0.25 | Rule | 100.0% | 4.972 | 0.000 |
| 0.25 | Random | 23.8% | 43.654 | 7.358 |
| 0.50 | Base | 100.0% ± 0.0% | 5.207 ± 0.112 | 0.000 ± 0.000 |
| 0.50 | Potential | 100.0% ± 0.0% | 5.039 ± 0.012 | 0.000 ± 0.000 |
| 0.50 | Rule | 100.0% | 6.408 | 0.000 |
| 0.50 | Random | 25.2% | 43.102 | 7.178 |

Learned entries are mean ± sample SD across three seeds; baselines were evaluated once per condition. Both learned methods solved the tested scenarios, but the rule used fewer steps at movement probability 0.10. All three Potential runs first reached the confirmed 90% validation threshold at 10,000 steps; Base first reached it at 20,000, 30,000, and 20,000 steps.

![Three-seed small-grid learning curves](results/grid5_benchmark/summary/learning_curves.png)

**Next decision.** Increase the arena size to reduce the success-rate ceiling, while keeping this completed benchmark as a separate archive.

**Data:** [learned final summaries](results/grid5_benchmark/test/learned_summary.csv), [baseline summaries](results/grid5_benchmark/test/baseline_summary.csv), [paired differences](results/grid5_benchmark/summary/paired_differences.csv), and [full report](results/grid5_benchmark/summary/results_summary.md).

### 6. Larger-grid development

**Reasoning.** A 50 by 50 extension was considered, but the implemented next step was 25 by 25 to keep the first scaling experiment bounded. No 50 by 50 experiment is reported.

**Implemented.** Two five-million-interaction, seed-0 runs with a 100-step cap and gamma 0.99. The float32 Q table grew to 18,750,000 element bytes. The target still moved horizontally and a hit still ended an episode.

| Policy | Success within 100 steps | Mean steps | Mean misses | Mean base return |
| --- | --- | --- | --- | --- |
| Base | 6.0% | 96.455 | 4.470 | -8.69275 |
| Potential | 100.0% | 24.675 | 0.180 | 8.58625 |
| Rule | 100.0% | 20.835 | 0.000 | 8.95825 |
| Random | 2.0% | 99.450 | 16.510 | -21.28250 |

These results use 200 development scenarios, not a new independent final test. The cap and gamma also changed from the small-grid study, so the difference is not attributable to grid size alone.

**Observed limitation.** On demo seed 3501000, Potential and the rule hit at tick 14, and their panels then held the completed state. Base timed out at tick 100. This truthful hit-once animation could not demonstrate sustained pursuit or continuous speed adaptation.

<details>
<summary>Open the 25 by 25 hit-once demonstration and learning curve</summary>

![Hit-once larger-grid demonstration](results/grid25_development/demo/demo_comparison.gif)

![Larger-grid development learning curves](results/grid25_development/summary/learning_curves.png)

</details>

**Next decision.** Change the mechanics to continuing tracking, rather than treating a first hit as evidence of sustained control.

**Evidence:** [trial configuration](configs/trial25.json), [comparison CSV](results/grid25_development/summary/comparison.csv), [full report](results/grid25_development/summary/results_summary.md), and [demo trace](results/grid25_development/demo/trajectories.csv).

### 7. Continuous tracking with turning and speed changes

**Reasoning.** A wider arena alone does not create prolonged tracking. The environment needs motion on both axes, turns, variable speed, and an outcome that continues after a hit.

**Implemented.** Fixed 300-tick windows; cardinal target headings; target speeds of one or two cells per tick; perpendicular turns and speed toggles with probability 0.05 each; unit-substep wall reflection; one-cell and three-cell crosshair moves. Each learned policy chooses its own movement speed. Shots are checked before target motion, and hits do not terminate or reset the window. Truncation and hits retain bootstrap and the actual next potential.

Two absolute-state methods trained for 20 million interactions each with seed 0. The same 200 development scenarios were used for the following comparison:

| Policy | Mean distance | Close (%) | Hits/window | Misses/window | Accuracy (%) |
| --- | --- | --- | --- | --- | --- |
| Base | 16.254 | 2.623 | 2.640 | 35.405 | 6.939 |
| Potential | 1.702 | 67.662 | 70.535 | 7.310 | 90.610 |
| Rule | 1.382 | 72.512 | 44.525 | 0.000 | 100.000 |
| Random | 17.161 | 0.633 | 0.050 | 29.380 | 0.170 |

The rule maintained better proximity and accuracy; Potential fired more often and obtained more hits. Base remained weak despite completing its training budget. Passing mechanism tests therefore did not establish good learned control. These are single-seed development results, distinct from the later independent final test.

<details>
<summary>Open the original continuous-tracking demonstration</summary>

![Historical four-policy continuous-tracking demonstration](results/continuous_tracking/render/seed3601000_comparison.gif)

</details>

**Next decision.** Diagnose the poor absolute Base policy, then test a representation change using matched budgets and multiple seeds.

**Evidence:** [mechanics specification](docs/tracking25_v2.md), [comparison CSV](results/continuous_tracking/evaluation/comparison.csv), [learning curves](results/continuous_tracking/render/learning_curves.png), [speed choices](results/continuous_tracking/render/speed_choices.png), and [full report](results/continuous_tracking/results_summary.md).

### 8. Diagnostics and relative-state study

**Reasoning.** The absolute representation assigns different rows to translated arrangements with the same relative target offset. Sharing those values could improve finite-budget learning, while discarding wall information could introduce new errors. Both possibilities require measurement.

**Diagnosis.** Replaying the archived Base model found distance above five cells on about 89.4% of ticks and exact maximum-Q ties on 89.77% of visited evaluation rows. A zero Q row was not treated as proof of an unvisited state. The new study collected genuine state-action update counts to measure coverage directly.

**Implemented.** Relative state `(xt-xc+24, yt-yc+24, heading, speed-1)`; Absolute/Relative crossed with Base/Potential; three seeds per method; five million interactions per run. Physics, action IDs, rewards, tie handling, and the four-million-step epsilon-decay schedule were matched within this new study. Four disposable pilots checked execution; their Q values were not transferred into formal runs.

**Results.** The [representation benchmark tables](#representation-benchmark) give the final and stress findings. Relative Potential achieved mean distance 1.440 cells, 2.338% far ticks, approximately 84.489 hits/window, and 99.493% accuracy. It did not meet the 75% close-tracking target. Near-boundary mean distance was 1.531 cells versus 1.325 in the interior. These policy-dependent slices describe an association and do not isolate the causal effect of aliasing.

![Matched representation-study learning curves](results/tracking_improvement/render/learning_curves.png)

[Boundary and speed diagnostics](results/tracking_improvement/render/boundary_speed_diagnostics.png) · [Distance tails](results/tracking_improvement/render/distance_tails.png) · [Update coverage](results/tracking_improvement/render/coverage.png) · [Movement-speed choices](results/tracking_improvement/render/speed_choices.png)

**Verification recorded for the study:** 132 development checkpoints, 26,400 development windows, 8,000 final windows, and 3,200 stress windows. Formal training plus checkpoint validation and saving took 22.10 minutes. Peak observed process memory was about 1.07 GiB, a process-lifetime high-water mark. Every run completed 16,666 full windows plus a 200-tick budget-stop partial window.

**Next decision.** Treat boundary-aware representation and the shooting/proximity tradeoff as separate prospective studies. Keep final-test results fixed rather than extending the budget until targets pass.

**Evidence:** [frozen protocol](results/tracking_improvement/protocol.json), [measured report](results/tracking_improvement/results_summary.md), [execution audit](results/tracking_improvement/execution_checks.json), and [models with visitation counters](results/tracking_improvement/runs/).

### 9. Publication and evidence organization

**Reasoning.** Results should be discoverable by purpose, while experimental records still identify the exact configurations and source bytes that produced them.

**Implemented.** Descriptive result directories, English explanatory material, actual GIF publication, exact source-compatibility mappings, and lossless model packages. The retired unrelated explanatory directory is absent from the current tree. Historical path strings inside frozen evidence remain as recorded and are resolved through the archive index.

The two historical continuous-tracking NPZ packages contain the exact original `q_table.npy` bytes. The new representation-study packages additionally identify representation/indexing metadata. All twelve formal model packages and twelve visitation packages were published. Eight disposable pilot binary packages were excluded from publication and retained locally; their recorded hashes and pilot text records were kept.

Publication checks recorded setup, 63 passing tests, model/counter integrity, paired records, document links, and GIF timing. Those are the recorded publication checks; they are separate from the study's original execution audit. Original statements that no upload occurred describe the experiment execution stage, before publication.

**Evidence:** [archive provenance](provenance/README.md), [publication explanation](provenance/tracking_improvement_publication.md), and [publication verification record](provenance/tracking_improvement_publication_checks.json).

### 10. Progress and presentation index

**Reasoning.** A results repository should explain how each observed limitation led to the next design decision, and let a reader reach the underlying data from the displayed table or figure.

**Implemented in this documentation revision.** A chronological progress index, all principal result tables on this page, captions explaining animation behavior and budgets, a detailed research history, and a path-migration/GIF guide. Existing artifact locations remain the canonical locations listed below. This revision presents recorded evidence; it does not constitute a new experiment.

## Demonstrations

Every link below points to an actual stored animation. Click a GIF link to open it; if a viewer displays only its first frame, download the GIF and open the local file.

| Study | Scenario / contents | Animation | Supporting trace |
|---|---|---|---|
| Current representation study | 3601000; archived Potential, Relative Base, Relative Potential, rule | [Main comparison](results/tracking_improvement/render/seed3601000_main_comparison.gif) | [CSV](results/tracking_improvement/render/seed3601000_main_comparison_traces.csv) |
| Current representation study | 3601001; same policy roles | [Main comparison](results/tracking_improvement/render/seed3601001_main_comparison.gif) | [CSV](results/tracking_improvement/render/seed3601001_main_comparison_traces.csv) |
| Current representation study | 3601002; same policy roles | [Main comparison](results/tracking_improvement/render/seed3601002_main_comparison.gif) | [CSV](results/tracking_improvement/render/seed3601002_main_comparison_traces.csv) |
| Current representation study | 3601000; four new learned methods, rule, Random | [Six-policy comparison](results/tracking_improvement/render/seed3601000_six_policy_comparison.gif) | [CSV](results/tracking_improvement/render/seed3601000_six_policy_comparison_traces.csv) |
| Historical continuous tracking | 3601000; original absolute Base/Potential, rule, Random | [Four-policy comparison](results/continuous_tracking/render/seed3601000_comparison.gif) | [Result directory](results/continuous_tracking/render/) |
| Historical continuous tracking | 3601001; same policy roles | [Four-policy comparison](results/continuous_tracking/render/seed3601001_comparison.gif) | [Result directory](results/continuous_tracking/render/) |
| Historical continuous tracking | 3601002; same policy roles | [Four-policy comparison](results/continuous_tracking/render/seed3601002_comparison.gif) | [Result directory](results/continuous_tracking/render/) |
| Larger-grid hit-once trial | 3501000; Base, Potential, rule; completed panels hold their final state | [Hit-once comparison](results/grid25_development/demo/demo_comparison.gif) | [CSV](results/grid25_development/demo/trajectories.csv) |

The seven continuous-tracking GIFs each contain 301 frames and 20.07 seconds. The hit-once GIF contains 101 frames. The main tracking scenarios were selected using target-motion coverage, not learned-policy scores. Random is an untrained negative control and poor outcomes remain visible. Recorded frame decoding and visual checks do not establish interactive GUI playback on every computer.

## Artifact locations and path changes

All paths are relative to the repository root. Numerical tables above link to their source records; this index groups the supporting outputs by purpose.

| Current directory | Contents |
|---|---|
| [results/environment_baselines](results/environment_baselines/) | Initial rule/random diagnostic CSVs and environment preview |
| [results/q_learning_development](results/q_learning_development/) | Static diagnostic, moving-target training, validation, and models |
| [results/reward_shaping_development](results/reward_shaping_development/) | Matched shaping comparison, diagnostic note, and frozen small-grid protocol |
| [results/grid5_benchmark](results/grid5_benchmark/) | Six formal runs, final-test data, paired differences, and small-grid figures |
| [results/grid25_development](results/grid25_development/) | Two larger-grid runs, four-policy summaries, and hit-once animation |
| [results/continuous_tracking](results/continuous_tracking/) | Original 20M continuous-tracking models, development metrics, and three GIFs |
| [results/tracking_improvement](results/tracking_improvement/) | Twelve 5M runs, diagnostics, independent final/stress data, figures, and four GIFs |

Historical references map to the published locations as follows. The old names in this table document migration only; they are not current result directories.

| Historical reference | Current published location |
| --- | --- |
| `outputs/day2_rule_smoke.csv` | [results/environment_baselines/rule_policy_diagnostic.csv](results/environment_baselines/rule_policy_diagnostic.csv) |
| `outputs/day2_random_smoke.csv` | [results/environment_baselines/random_policy_diagnostic.csv](results/environment_baselines/random_policy_diagnostic.csv) |
| `outputs/day2_rule_preview.png` | [results/environment_baselines/rule_policy_preview.png](results/environment_baselines/rule_policy_preview.png) |
| `outputs/day3` | [results/q_learning_development](results/q_learning_development) |
| `outputs/day4` | [results/reward_shaping_development](results/reward_shaping_development) |
| `outputs/day5` | [results/grid5_benchmark](results/grid5_benchmark) |
| `outputs/trial25` | [results/grid25_development](results/grid25_development) |
| `outputs/tracking25_v2` | [results/continuous_tracking](results/continuous_tracking) |

The representation study was created directly under `results/tracking_improvement/`; it has no earlier numbered archive path. Its location has not changed in this documentation revision. A historical continuous model reference ending in `q_table.npy` resolves to the corresponding published `q_table.npz` package under `results/continuous_tracking/base/` or `potential/`.

For example, the current saved Relative Potential model for training seed 10 is [results/tracking_improvement/runs/relative_potential/seed_10/q_table.npz](results/tracking_improvement/runs/relative_potential/seed_10/q_table.npz), with genuine update counts in [visitation.npz](results/tracking_improvement/runs/relative_potential/seed_10/visitation.npz). The [artifact guide](docs/artifact_paths.md) lists exact report, figure, GIF, model, trace, and verification paths and explains which files are historical evidence.

## Installation and verification

**Fresh-checkout reproduction:** use the [supported reproduction guide](docs/reproduction.md) for verified saved-model evaluation and independent fresh training. The new `aimrl.reproduce` entry point records its own session and avoids original-workspace-only startup files. The historical commands remain execution records.

The recorded Windows environment used Python 3.13.9, NumPy 2.5.4, Matplotlib 3.11.2, and Pillow 13.0.0. No GPU or deep-learning framework is required. Use the existing working environment when available. From a normal clone of the repository root, a new Windows PowerShell environment can be prepared with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m aimrl.reproduce --mode verify --output .tmp/reproduction/publication_check
```

The publication verifier checks the published models, records, source bundle, figures/GIFs, and links without training. Some historical retired-evidence checks require full Git history; a source ZIP alone cannot satisfy those checks. The original research audit also records its original workspace, pilot arrays, and startup snapshot; the [publication guide](provenance/tracking_improvement_publication.md) explains the appropriate verifier for a published checkout.

Open a current demonstration from the repository root:

```powershell
Start-Process ".\results\tracking_improvement\render\seed3601000_main_comparison.gif"
```

Run a bounded trace of the historical saved Potential model without training:

```powershell
.\.venv\Scripts\python.exe -m aimrl.tracking25.demo --model results/continuous_tracking/potential/q_table.npz --ticks 30 --seed 3601000
```

This command writes a short trace; it does not open an interactive game window. For complete training protocols, use the [small-grid specification](spec.md), [larger-grid instructions](docs/trial25.md), [continuous-tracking specification](docs/tracking25_v2.md), and [representation-study execution record](docs/tracking_improvement.md). Completed outputs are protected against overwrite. Fresh experiments require a distinct output session and their own frozen protocol.

## Limitations and next work

- The arena is a structured-state simulation. No real-game input control, screen recognition, or game-performance claim is included.
- Relative-state aliasing at boundaries is established mechanically. Observational boundary slices do not determine its isolated causal effect on performance.
- The 75% close-tracking target is still unmet. Higher hit counts, higher accuracy, and closer tracking are separate objectives under the current firing rule.
- The current formal comparison uses three training seeds and finite scenario sets. Results are descriptive; independent scenario seeds need not generate states never seen in training.
- Early hit-once success, original continuous-tracking development, and the latest final/stress tests use different tasks, splits, or budgets. Their tables should not be pooled into one leaderboard.
- Remaining work is the course-format final report and presentation, followed, if pursued, by a separately specified boundary-aware or tracking-objective study. A 50 by 50 extension has not been executed. Interactive GUI playback remains unverified.

Potential-based shaping follows Ng, Harada, and Russell (1999), [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). The potential form does not provide a finite-budget or optimality guarantee for a lossy relative observation. Full technical references are in the [specification](spec.md).

## Reproduction workflow update

A separate checkout runner now supports `verify`, `evaluate`, and `train` modes with protected fresh output directories. It preserves the published learning implementation and historical evidence while recording new source/configuration provenance. Memory telemetry is platform-specific and best effort. New replay summaries clarify that the historical clipping diagnostic measures fully blocked moves.

Windows acceptance on 2026-11-03 passed all 74 tests. Complete publication verification passed in standard Python and with `python -O`. A separate fresh Windows virtual environment passed the pinned install, dependency check, setup check, and a limited development replay. The full final-split replay covered Relative Base and Relative Potential (seed 10), Rule, and Random, for 2,000 windows and 36 exact metric matches. The full twelve-model final/stress replay, formal 60M retraining, other platforms for this update, and native GUI playback remain unverified in this acceptance. The course-format paper and member-contribution statement are still pending.

See the [reproduction guide and acceptance evidence](docs/reproduction.md).

## Tracking–shooting reward trade-offs

The reward study changed the proximity component of the training reward under fixed dynamics, action space, relative state and Q-learning settings. It completed **21 runs × 5,000,000 interactions = 105,000,000 training updates**, with seeds 20, 21 and 22 for each condition. Nearness bonuses changed both the tracking and shooting behavior learned by the original policies.

D is mean post-motion Manhattan distance, C is the fraction of post-motion ticks within one cell, and H is hits per 300-tick window. Learned entries show **mean ± sample SD across the three training seeds**. Rule is a fixed policy on the same paired scenarios. A seed mean averages seed-level hits/actual shots with equal seed weights. A pooled sums hits and actual shots across seeds before division. Zero actual shots yield NA.

### Reward study Final (500 scenarios, 4200000–4200499)

| Condition | D (cells) | C (%) | H / 300 ticks | A seed mean (%) | A pooled (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reference` | 1.517 ± 0.142 | 58.50 ± 4.73 | 81.33 ± 4.95 | 99.36 ± 0.78 | 99.39 |
| `distance_0p1` | 1.423 ± 0.015 | 60.56 ± 1.99 | 84.07 ± 0.10 | 95.61 ± 6.88 | 95.26 |
| `distance_0p3` | 1.474 ± 0.066 | 59.69 ± 2.97 | 80.97 ± 5.02 | 99.03 ± 0.43 | 99.05 |
| `distance_1p0` | 1.523 ± 0.064 | 65.57 ± 6.62 | 80.62 ± 4.97 | 95.70 ± 6.26 | 95.31 |
| `close_0p5` | 1.398 ± 0.021 | 65.12 ± 2.18 | 84.44 ± 0.15 | 95.69 ± 6.03 | 95.42 |
| `close_2p0` | 1.355 ± 0.015 | 69.39 ± 0.74 | 81.73 ± 4.98 | 90.61 ± 15.49 | 88.30 |
| `close_5p0` | 1.336 ± 0.058 | 72.77 ± 1.90 | 81.52 ± 5.07 | 86.70 ± 11.30 | 85.36 |
| `rule` | 1.375 | 72.42 | 44.59 | 100.00 | 100.00 |

Close bonus 0.5 improved D by 0.119 cells, C by 6.62 percentage points and H by 3.83% versus the matched reference. Its 95.69% mean accuracy missed the 98% gate. No original condition met the full simultaneous or retention screen. Distance penalty 0.3 kept accuracy above 98%, lowered D and lost 0.45% of hits, with a 1.19-point C gain below the +5-point gate.

### Reward study Stress (200 scenarios, 4300000–4300199)

| Condition | D (cells) | C (%) | H / 300 ticks | A seed mean (%) | A pooled (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reference` | 1.688 ± 0.120 | 53.33 ± 3.89 | 71.25 ± 4.00 | 99.08 ± 1.00 | 99.11 |
| `distance_0p1` | 1.603 ± 0.007 | 55.24 ± 1.60 | 73.36 ± 0.18 | 94.72 ± 7.87 | 94.28 |
| `distance_0p3` | 1.618 ± 0.065 | 54.73 ± 2.67 | 70.86 ± 4.18 | 98.28 ± 0.82 | 98.31 |
| `distance_1p0` | 1.644 ± 0.086 | 60.25 ± 6.04 | 70.79 ± 3.95 | 95.10 ± 6.75 | 94.67 |
| `close_0p5` | 1.587 ± 0.024 | 59.24 ± 1.80 | 73.81 ± 0.37 | 95.37 ± 5.35 | 95.15 |
| `close_2p0` | 1.547 ± 0.011 | 63.14 ± 0.75 | 71.65 ± 4.00 | 90.13 ± 15.70 | 87.76 |
| `close_5p0` | 1.531 ± 0.057 | 66.42 ± 1.73 | 71.22 ± 3.91 | 86.16 ± 11.71 | 84.79 |
| `rule` | 1.469 | 68.36 | 42.45 | 100.00 | 100.00 |

Stress uses 0.10 turn and speed-toggle probabilities. The prespecified pass decisions apply to Final.

![Reward study close fraction versus hits](results/tracking_shooting_tradeoff/figures/close_hits_tradeoff.png)

![Reward study four-policy demonstration on the same target route](results/tracking_shooting_tradeoff/demos/scenario_4400000.gif)

[English report](results/tracking_shooting_tradeoff/results_summary.md) · [Paper addendum](results/tracking_shooting_tradeoff/paper_addendum.md) · [Models and checkpoints](results/tracking_shooting_tradeoff/runs) · [Per-seed data](results/tracking_shooting_tradeoff/per_seed.csv) · [Paired differences](results/tracking_shooting_tradeoff/paired_differences.csv) · [Final and Stress windows](results/tracking_shooting_tradeoff/evaluation) · [Distance–hits plot](results/tracking_shooting_tradeoff/figures/distance_hits_tradeoff.png) · [Learning curves](results/tracking_shooting_tradeoff/figures/learning_curves.png) · [Demo 4400001](results/tracking_shooting_tradeoff/demos/scenario_4400001.gif) · [Demo 4400002](results/tracking_shooting_tradeoff/demos/scenario_4400002.gif) · [Protocol](results/tracking_shooting_tradeoff/protocol.json) · [Execution checks](results/tracking_shooting_tradeoff/execution_checks.json) · [Build and reproduction commands](docs/tracking_shooting_tradeoff.md)

## Q-learning with a firing guard

The inference study reused **21 frozen reward-study models with zero new training updates**. The original variant executes the greedy action. **Q-learning + firing guard** queries the same policy once and executes Stay only when it proposes Fire while unaligned. Both variants produced exactly the same physical trajectories, D, C and H. The guard removed unaligned shots and the corresponding miss penalties. Rule remains the same single fixed comparator.

### Firing guard Final (500 new scenarios, 4500000–4500499)

| Condition | D (cells) | C (%) | H / 300 ticks | A seed mean original → guard (%) | A pooled original → guard (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reference` | 1.526 ± 0.135 | 58.41 ± 4.69 | 81.22 ± 4.78 | 99.32 ± 0.86 → 100.00 ± 0.00 | 99.35 → 100.00 |
| `distance_0p1` | 1.434 ± 0.013 | 60.47 ± 2.03 | 83.90 ± 0.05 | 95.43 ± 7.17 → 100.00 ± 0.00 | 95.06 → 100.00 |
| `distance_0p3` | 1.484 ± 0.076 | 59.67 ± 3.05 | 80.85 ± 5.00 | 99.00 ± 0.43 → 100.00 ± 0.00 | 99.02 → 100.00 |
| `distance_1p0` | 1.520 ± 0.073 | 65.73 ± 6.84 | 80.41 ± 4.92 | 95.53 ± 6.50 → 100.00 ± 0.00 | 95.11 → 100.00 |
| `close_0p5` | 1.404 ± 0.021 | 65.20 ± 2.17 | 84.21 ± 0.09 | 95.59 ± 6.21 → 100.00 ± 0.00 | 95.31 → 100.00 |
| `close_2p0` | 1.357 ± 0.011 | 69.71 ± 0.75 | 81.58 ± 4.88 | 90.36 ± 15.88 → 100.00 ± 0.00 | 87.94 → 100.00 |
| `close_5p0` | 1.340 ± 0.053 | 73.19 ± 1.81 | 81.29 ± 4.97 | 86.20 ± 11.70 → 100.00 ± 0.00 | 84.79 → 100.00 |
| `rule` | 1.381 | 73.01 | 43.48 | 100.00 | 100.00 |

Against the **guarded** reference, `close_0p5` with the guard lowered D by 0.122 cells, increased C by 6.79 percentage points and increased H by 3.69%, with 100% observed shot accuracy. It was the only condition meeting all four simultaneous gates on the **cross-seed Final means**. The other guarded retention passes were `distance_1p0`, `close_2p0` and `close_5p0`. The distance-1.0 condition lost 0.99% of hits. Close-5.0 had the largest close gain among the retention passes, at 14.77 points and +0.10% hits. Neither the previous Final nor the new Final yielded a full pass for an original policy.

The thresholds require D no larger than reference, C at least 5 percentage points higher and A at least 98%. The simultaneous screen requires at least 2% more hits. The retention screen permits a hit loss of at most 2%. These are group-level Final decisions. The table does not assert a pass for each seed or for Stress.

### Firing guard Stress (200 new scenarios, 4600000–4600199)

| Condition | D (cells) | C (%) | H / 300 ticks | A seed mean original → guard (%) | A pooled original → guard (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reference` | 1.683 ± 0.123 | 53.35 ± 4.10 | 71.23 ± 4.16 | 99.06 ± 1.00 → 100.00 ± 0.00 | 99.10 → 100.00 |
| `distance_0p1` | 1.582 ± 0.008 | 55.22 ± 1.54 | 73.44 ± 0.17 | 94.66 ± 7.94 → 100.00 ± 0.00 | 94.20 → 100.00 |
| `distance_0p3` | 1.603 ± 0.060 | 54.67 ± 2.90 | 70.90 ± 4.28 | 98.25 ± 0.77 → 100.00 ± 0.00 | 98.28 → 100.00 |
| `distance_1p0` | 1.603 ± 0.075 | 60.35 ± 6.22 | 70.85 ± 4.06 | 95.07 ± 6.82 → 100.00 ± 0.00 | 94.64 → 100.00 |
| `close_0p5` | 1.575 ± 0.021 | 59.25 ± 1.79 | 73.78 ± 0.33 | 95.22 ± 5.86 → 100.00 ± 0.00 | 94.95 → 100.00 |
| `close_2p0` | 1.536 ± 0.015 | 63.28 ± 0.79 | 71.60 ± 4.14 | 89.93 ± 16.11 → 100.00 ± 0.00 | 87.43 → 100.00 |
| `close_5p0` | 1.517 ± 0.054 | 66.58 ± 1.75 | 71.27 ± 4.04 | 85.72 ± 12.08 → 100.00 ± 0.00 | 84.26 → 100.00 |
| `rule` | 1.457 | 68.72 | 41.70 | 100.00 | 100.00 |

Stress uses the same target event probabilities as the earlier stress condition. The guard's 100% observed accuracy is conditional on actually firing. The two objective projections are unchanged because D, C and H are invariant.

![Original and guarded objective points coincide](results/tracking_fire_guard/figures/objective_tradeoffs.png)

![Synchronized original and firing guard replay](results/tracking_fire_guard/demo/synchronized_fire_guard.gif)

[English report](results/tracking_fire_guard/results_summary.md) · [Paper addendum](results/tracking_fire_guard/paper_addendum.md) · [Frozen models](results/tracking_shooting_tradeoff/runs) · [Full comparison](results/tracking_fire_guard/comparison.csv) · [Per-seed data](results/tracking_fire_guard/per_seed.csv) · [Paired guard effects](results/tracking_fire_guard/paired_differences.csv) · [Final and Stress windows](results/tracking_fire_guard/evaluation) · [Threshold decisions](results/tracking_fire_guard/thresholds.csv) · [Accuracy and misses](results/tracking_fire_guard/figures/accuracy_and_misses.png) · [Q diagnostics](results/tracking_fire_guard/q_diagnostics_by_condition.csv) · [Protocol](results/tracking_fire_guard/protocol.json) · [Execution checks](results/tracking_fire_guard/execution_checks.json) · [Reproduction commands](docs/tracking_fire_guard.md)
