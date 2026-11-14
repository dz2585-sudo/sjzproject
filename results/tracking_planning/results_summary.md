# Short-horizon planning with the existing actions

This study reuses frozen R-Fixed and R-Decay step-5M models, seeds 60–64. New training interactions and Q updates are exactly zero. The 20 policy instances comprise Rule (1), R-Fixed (5), R-Decay (5), Plan-H1/H2/H3 with Phi terminal (3), Plan-H3-Zero (1), and Hybrid-H3 using five different frozen R-Decay Q terminals (5). Each pure planner is evaluated once per scenario, never copied into five fictitious training seeds.

## Information and task

All methods retain the ten original actions, speeds 1/2, move/fire exclusivity, pre-motion hit test, post-motion close reward, substep wall reflection and 300-tick continuing windows. Task reward is base+0.5 I(d_post≤1), gamma=0.99. Planners observe full current physical state and a known fixed stochastic motion model (turn/toggle probabilities 0.05/0.05). R policies observe relative offsets, heading and speed. Full-state knowledge and additional computation accompany the cross-family comparison; it is not a depth-only causal contrast. Only the Plan-H1→H2→H3 comparison holds representation, model and terminal fixed.

Stress changes real turn/toggle probabilities to 0.10/0.10; planning remains at 0.05/0.05. No planner receives scenario seeds, RNG states, pre-generated real future trajectories or actual future random events. Every executed action passes the same firing guard. All ten actions are evaluated in each DP backup; strictly dominated unaligned Fire is never proposed by any planned controller.

## Feedback recurrence and terminal values

V_h(s)=max_a sum_s' P(s'|s,a)[r_task+gamma V_(h−1)(s')]. Each successor has its own feedback action; only the root action is executed, then the observed new state is queried. Remaining horizon never shrinks near tick 300. Target branches enumerate no/left/right turn crossed with keep/toggle speed; duplicate complete target states are merged by summed probability.

Phi(s)=−L1 distance. The three V0 choices are Phi, zero, and max(Q_shaped)+Phi. Previous reward was task+gamma Phi(next)−Phi(current), with shaping scale one. Therefore the Q conversion has a plus Phi sign. Task+Phi terminal equals shaped+zero terminal up to root constant Phi(s); task+[max Q+Phi] equals shaped+max Q up to the same constant. Numerical/action equivalence tests passed. Planning uses float64 values and a fixed lowest-ID tie within 1e−10 of the exact maximum. R uses its original exact lowest-ID argmax; Rule preserves the original distance/displacement/action ordering.

## Main held-out results

Means ± sample SD are shown only where five distinct frozen learning models exist. Single pure controllers have no training-seed SD. A pools actual hits/shots within a policy and is NA if no shots. Metrics retain the previous definitions; all detailed seed/window records are saved.

| Split | Family | Instances | D | C | H / 300 ticks | A |
| --- | --- | --- | --- | --- | --- | --- |
| final | Rule | 1 | 1.3849 | 72.74% | 43.4080 | 100.00% |
| final | R-Fixed | 5 | 1.3848 ± 0.0137 | 68.90 ± 1.5378% | 84.2240 ± 0.2003 | 100.00 ± 0.0000% |
| final | R-Decay | 5 | 1.3465 ± 0.0091 | 73.31 ± 0.0785% | 84.5276 ± 0.0611 | 100.00 ± 0.0000% |
| final | Plan-H1 | 1 | 1.2990 | 76.61% | 50.7560 | 100.00% |
| final | Plan-H2 | 1 | 1.2285 | 74.52% | 87.6940 | 100.00% |
| final | Plan-H3 | 1 | 1.2219 | 74.75% | 88.3420 | 100.00% |
| final | Plan-H3-Zero | 1 | 2.4685 | 70.96% | 83.8020 | 100.00% |
| final | Hybrid-H3 | 5 | 1.2134 ± 0.0025 | 75.15 ± 0.0264% | 88.7836 ± 0.0158 | 100.00 ± 0.0000% |
| stress | Rule | 1 | 1.4827 | 67.91% | 41.6050 | 100.00% |
| stress | R-Fixed | 5 | 1.5752 ± 0.0103 | 62.48 ± 1.2751% | 73.9950 ± 0.2589 | 100.00 ± 0.0000% |
| stress | R-Decay | 5 | 1.5527 ± 0.0207 | 66.33 ± 0.2157% | 74.4570 ± 0.0391 | 100.00 ± 0.0000% |
| stress | Plan-H1 | 1 | 1.3930 | 71.96% | 47.8600 | 100.00% |
| stress | Plan-H2 | 1 | 1.3777 | 68.63% | 78.3700 | 100.00% |
| stress | Plan-H3 | 1 | 1.3709 | 68.83% | 79.1750 | 100.00% |
| stress | Plan-H3-Zero | 1 | 2.4651 | 65.67% | 75.2500 | 100.00% |
| stress | Hybrid-H3 | 5 | 1.3613 ± 0.0053 | 69.38 ± 0.0762% | 79.6920 ± 0.0445 | 100.00 ± 0.0000% |

## Prespecified gates relative to R-Decay

Practical: D nonincrease, C≥baseline+5pp, H≥98% baseline, A≥98%. Strong substitutes H≥102%. Absolute C≥75% is reported independently, along with hit retention. Hybrid is paired to its own model; pure methods are compared with each of five baselines and with their mean. Family mean and each-pair results are both retained in thresholds.csv.

| Split | Family | Practical | Strong | C≥75% | C≥75% and H retained | C difference | H ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | Rule | False | False | False | False | -0.570 pp | 0.5135 |
| final | R-Fixed | False | False | False | False | -4.413 pp | 0.9964 |
| final | R-Decay | False | False | False | False | +0.000 pp | 1.0000 |
| final | Plan-H1 | False | False | True | False | +3.303 pp | 0.6005 |
| final | Plan-H2 | False | False | False | False | +1.205 pp | 1.0375 |
| final | Plan-H3 | False | False | False | False | +1.442 pp | 1.0451 |
| final | Plan-H3-Zero | False | False | False | False | -2.350 pp | 0.9914 |
| final | Hybrid-H3 | False | False | True | True | +1.835 pp | 1.0504 |
| stress | Rule | False | False | False | False | +1.575 pp | 0.5588 |
| stress | R-Fixed | False | False | False | False | -3.853 pp | 0.9938 |
| stress | R-Decay | False | False | False | False | +0.000 pp | 1.0000 |
| stress | Plan-H1 | False | False | False | False | +5.625 pp | 0.6428 |
| stress | Plan-H2 | False | False | False | False | +2.295 pp | 1.0526 |
| stress | Plan-H3 | False | False | False | False | +2.497 pp | 1.0634 |
| stress | Plan-H3-Zero | False | False | False | False | -0.660 pp | 1.0107 |
| stress | Hybrid-H3 | False | False | False | False | +3.040 pp | 1.0703 |

## Horizon and terminal contrasts

Differences are candidate minus baseline. These are deterministic empirical controller contrasts or means over the five learned terminal tables, not five independent pure-planner training runs.

| Split | Contrast | D difference | C difference | H difference |
| --- | --- | --- | --- | --- |
| final | horizon_2_minus_1 | -0.0705 | -2.0987 pp | 36.9380 |
| final | horizon_3_minus_2 | -0.0066 | +0.2373 pp | 0.6480 |
| final | Phi_terminal_minus_zero | -1.2466 | +3.7920 pp | 4.5400 |
| final | learned_terminal_minus_Phi | -0.0085 | +0.3933 pp | 0.4416 |
| stress | horizon_2_minus_1 | -0.0154 | -3.3300 pp | 30.5100 |
| stress | horizon_3_minus_2 | -0.0067 | +0.2017 pp | 0.8050 |
| stress | Phi_terminal_minus_zero | -1.0941 | +3.1567 pp | 3.9250 |
| stress | learned_terminal_minus_Phi | -0.0096 | +0.5433 pp | 0.5170 |

## Complete policy-instance table

| Split | Policy | D | C | H | A | First close | Longest far |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | Rule | 1.3849 | 72.741% | 43.408 | 100.00% | 9.64 | 5.01 |
| final | R-Fixed_s60 | 1.3708 | 69.617% | 84.070 | 100.00% | 8.44 | 5.46 |
| final | R-Fixed_s61 | 1.3895 | 69.525% | 83.978 | 100.00% | 8.29 | 5.57 |
| final | R-Fixed_s62 | 1.4059 | 66.155% | 84.234 | 100.00% | 8.49 | 5.52 |
| final | R-Fixed_s63 | 1.3762 | 69.755% | 84.412 | 100.00% | 8.58 | 5.57 |
| final | R-Fixed_s64 | 1.3817 | 69.439% | 84.426 | 100.00% | 8.61 | 5.65 |
| final | R-Decay_s60 | 1.3475 | 73.302% | 84.522 | 100.00% | 8.49 | 5.51 |
| final | R-Decay_s61 | 1.3430 | 73.343% | 84.432 | 100.00% | 8.39 | 5.55 |
| final | R-Decay_s62 | 1.3499 | 73.308% | 84.530 | 100.00% | 8.51 | 5.65 |
| final | R-Decay_s63 | 1.3584 | 73.194% | 84.598 | 100.00% | 8.36 | 5.42 |
| final | R-Decay_s64 | 1.3338 | 73.410% | 84.556 | 100.00% | 8.36 | 5.43 |
| final | Plan-H1 | 1.2990 | 76.615% | 50.756 | 100.00% | 8.42 | 4.83 |
| final | Plan-H2 | 1.2285 | 74.516% | 87.694 | 100.00% | 7.62 | 4.60 |
| final | Plan-H3 | 1.2219 | 74.753% | 88.342 | 100.00% | 7.33 | 4.51 |
| final | Plan-H3-Zero | 2.4685 | 70.961% | 83.802 | 100.00% | 21.46 | 18.54 |
| final | Hybrid-H3_s60 | 1.2164 | 75.111% | 88.782 | 100.00% | 7.33 | 4.65 |
| final | Hybrid-H3_s61 | 1.2151 | 75.163% | 88.804 | 100.00% | 7.32 | 4.68 |
| final | Hybrid-H3_s62 | 1.2138 | 75.129% | 88.784 | 100.00% | 7.33 | 4.70 |
| final | Hybrid-H3_s63 | 1.2108 | 75.175% | 88.788 | 100.00% | 7.38 | 4.68 |
| final | Hybrid-H3_s64 | 1.2107 | 75.156% | 88.760 | 100.00% | 7.32 | 4.67 |
| stress | Rule | 1.4827 | 67.910% | 41.605 | 100.00% | 9.40 | 5.03 |
| stress | R-Fixed_s60 | 1.5642 | 63.147% | 73.755 | 100.00% | 8.76 | 5.37 |
| stress | R-Fixed_s61 | 1.5872 | 62.968% | 73.685 | 100.00% | 8.79 | 5.73 |
| stress | R-Fixed_s62 | 1.5845 | 60.210% | 74.085 | 100.00% | 8.65 | 5.53 |
| stress | R-Fixed_s63 | 1.5670 | 63.180% | 74.210 | 100.00% | 8.71 | 5.53 |
| stress | R-Fixed_s64 | 1.5733 | 62.902% | 74.240 | 100.00% | 8.73 | 5.42 |
| stress | R-Decay_s60 | 1.5559 | 66.303% | 74.500 | 100.00% | 8.97 | 5.57 |
| stress | R-Decay_s61 | 1.5359 | 66.513% | 74.405 | 100.00% | 8.71 | 5.54 |
| stress | R-Decay_s62 | 1.5520 | 66.312% | 74.435 | 100.00% | 8.80 | 5.57 |
| stress | R-Decay_s63 | 1.5856 | 66.003% | 74.490 | 100.00% | 8.81 | 5.54 |
| stress | R-Decay_s64 | 1.5343 | 66.542% | 74.455 | 100.00% | 8.84 | 5.45 |
| stress | Plan-H1 | 1.3930 | 71.960% | 47.860 | 100.00% | 8.39 | 4.79 |
| stress | Plan-H2 | 1.3777 | 68.630% | 78.370 | 100.00% | 7.59 | 4.59 |
| stress | Plan-H3 | 1.3709 | 68.832% | 79.175 | 100.00% | 7.39 | 4.46 |
| stress | Plan-H3-Zero | 2.4651 | 65.675% | 75.250 | 100.00% | 20.21 | 17.20 |
| stress | Hybrid-H3_s60 | 1.3664 | 69.307% | 79.670 | 100.00% | 7.50 | 4.63 |
| stress | Hybrid-H3_s61 | 1.3665 | 69.308% | 79.695 | 100.00% | 7.50 | 4.63 |
| stress | Hybrid-H3_s62 | 1.3618 | 69.358% | 79.720 | 100.00% | 7.41 | 4.64 |
| stress | Hybrid-H3_s63 | 1.3568 | 69.485% | 79.745 | 100.00% | 7.42 | 4.62 |
| stress | Hybrid-H3_s64 | 1.3551 | 69.417% | 79.630 | 100.00% | 7.41 | 4.59 |

## Speed, boundaries and behavior

Historical speed slices use actual post-toggle motion speed. New pre_speed slices use the observed speed before the action. Boundary means either object within three cells of an edge before action. Slice hit rates below normalize by 300 slice ticks, unlike H in the main table, which is per full window. All policy-trajectory slices are descriptive, not a matched-state causal decomposition. Common-state action comparisons provide a separate matched complement with distinct-state and event-frequency weights.

| Split | Family | Slice | D | C | Hits / 300 slice ticks | Fire when aligned |
| --- | --- | --- | --- | --- | --- | --- |
| final | Rule | pre_speed_1 | 1.2583 | 89.07% | 19.08 | 1.0000 |
| final | R-Fixed | pre_speed_1 | 1.1765 | 79.91% | 88.48 | 1.0000 |
| final | R-Decay | pre_speed_1 | 1.0873 | 88.84% | 87.65 | 1.0000 |
| final | Plan-H1 | pre_speed_1 | 1.2209 | 91.83% | 21.45 | 1.0000 |
| final | Plan-H2 | pre_speed_1 | 1.0217 | 89.67% | 90.39 | 1.0000 |
| final | Plan-H3 | pre_speed_1 | 1.0197 | 89.76% | 90.82 | 1.0000 |
| final | Plan-H3-Zero | pre_speed_1 | 2.4043 | 85.07% | 85.96 | 1.0000 |
| final | Hybrid-H3 | pre_speed_1 | 1.0180 | 89.88% | 91.18 | 1.0000 |
| final | Rule | pre_speed_2 | 1.5121 | 56.33% | 67.85 | 1.0000 |
| final | R-Fixed | pre_speed_2 | 1.5942 | 57.84% | 79.95 | 1.0000 |
| final | R-Decay | pre_speed_2 | 1.6070 | 57.70% | 81.39 | 1.0000 |
| final | Plan-H1 | pre_speed_2 | 1.3773 | 61.32% | 80.20 | 1.0000 |
| final | Plan-H2 | pre_speed_2 | 1.4362 | 59.29% | 84.98 | 1.0000 |
| final | Plan-H3 | pre_speed_2 | 1.4250 | 59.67% | 85.85 | 1.0000 |
| final | Plan-H3-Zero | pre_speed_2 | 2.5329 | 56.79% | 81.63 | 1.0000 |
| final | Hybrid-H3 | pre_speed_2 | 1.4097 | 60.34% | 86.37 | 1.0000 |
| final | Rule | post_speed_1 | 1.2120 | 92.61% | 21.52 | 1.0000 |
| final | R-Fixed | post_speed_1 | 1.1523 | 81.69% | 88.04 | 1.0000 |
| final | R-Decay | post_speed_1 | 1.0752 | 90.24% | 87.34 | 1.0000 |
| final | Plan-H1 | post_speed_1 | 1.1660 | 95.88% | 24.34 | 1.0000 |
| final | Plan-H2 | post_speed_1 | 1.0064 | 91.07% | 89.97 | 1.0000 |
| final | Plan-H3 | post_speed_1 | 1.0022 | 91.27% | 90.44 | 1.0000 |
| final | Plan-H3-Zero | post_speed_1 | 2.3773 | 86.43% | 85.58 | 1.0000 |
| final | Hybrid-H3 | post_speed_1 | 1.0001 | 91.46% | 90.74 | 1.0000 |
| final | Rule | post_speed_2 | 1.5588 | 52.77% | 65.41 | 1.0000 |
| final | R-Fixed | post_speed_2 | 1.6185 | 56.04% | 80.39 | 1.0000 |
| final | R-Decay | post_speed_2 | 1.6192 | 56.30% | 81.70 | 1.0000 |
| final | Plan-H1 | post_speed_2 | 1.4326 | 57.25% | 77.31 | 1.0000 |
| final | Plan-H2 | post_speed_2 | 1.4517 | 57.87% | 85.41 | 1.0000 |
| final | Plan-H3 | post_speed_2 | 1.4426 | 58.15% | 86.24 | 1.0000 |
| final | Plan-H3-Zero | post_speed_2 | 2.5602 | 55.41% | 82.01 | 1.0000 |
| final | Hybrid-H3 | post_speed_2 | 1.4278 | 58.75% | 86.82 | 1.0000 |
| final | Rule | boundary | 1.4737 | 73.09% | 47.83 | 1.0000 |
| final | R-Fixed | boundary | 1.4920 | 68.86% | 81.12 | 1.0000 |
| final | R-Decay | boundary | 1.4548 | 73.30% | 82.09 | 1.0000 |
| final | Plan-H1 | boundary | 1.3910 | 76.47% | 55.04 | 1.0000 |
| final | Plan-H2 | boundary | 1.3185 | 74.54% | 86.49 | 1.0000 |
| final | Plan-H3 | boundary | 1.3082 | 74.92% | 87.53 | 1.0000 |
| final | Plan-H3-Zero | boundary | 3.5085 | 68.22% | 79.53 | 1.0000 |
| final | Hybrid-H3 | boundary | 1.3049 | 75.13% | 88.11 | 1.0000 |
| final | Rule | interior | 1.2796 | 72.32% | 38.16 | 1.0000 |
| final | R-Fixed | interior | 1.2495 | 68.94% | 88.14 | 1.0000 |
| final | R-Decay | interior | 1.2099 | 73.33% | 87.60 | 1.0000 |
| final | Plan-H1 | interior | 1.1896 | 76.79% | 45.67 | 1.0000 |
| final | Plan-H2 | interior | 1.1209 | 74.49% | 89.14 | 1.0000 |
| final | Plan-H3 | interior | 1.1197 | 74.56% | 89.30 | 1.0000 |
| final | Plan-H3-Zero | interior | 1.1430 | 74.46% | 89.25 | 1.0000 |
| final | Hybrid-H3 | interior | 1.1046 | 75.16% | 89.58 | 1.0000 |
| stress | Rule | pre_speed_1 | 1.3625 | 81.67% | 25.22 | 1.0000 |
| stress | R-Fixed | pre_speed_1 | 1.3381 | 72.72% | 80.56 | 1.0000 |
| stress | R-Decay | pre_speed_1 | 1.2597 | 80.74% | 78.99 | 1.0000 |
| stress | Plan-H1 | pre_speed_1 | 1.3071 | 85.35% | 28.35 | 1.0000 |
| stress | Plan-H2 | pre_speed_1 | 1.1903 | 81.90% | 81.51 | 1.0000 |
| stress | Plan-H3 | pre_speed_1 | 1.1837 | 82.01% | 82.47 | 1.0000 |
| stress | Plan-H3-Zero | pre_speed_1 | 2.2389 | 78.54% | 78.56 | 1.0000 |
| stress | Hybrid-H3 | pre_speed_1 | 1.1802 | 82.26% | 83.00 | 1.0000 |
| stress | Rule | pre_speed_2 | 1.6019 | 54.26% | 57.87 | 1.0000 |
| stress | R-Fixed | pre_speed_2 | 1.8105 | 52.32% | 67.48 | 1.0000 |
| stress | R-Decay | pre_speed_2 | 1.8435 | 52.04% | 69.96 | 1.0000 |
| stress | Plan-H1 | pre_speed_2 | 1.4783 | 58.67% | 67.22 | 1.0000 |
| stress | Plan-H2 | pre_speed_2 | 1.5635 | 55.46% | 75.25 | 1.0000 |
| stress | Plan-H3 | pre_speed_2 | 1.5568 | 55.75% | 75.91 | 1.0000 |
| stress | Plan-H3-Zero | pre_speed_2 | 2.6895 | 52.90% | 71.96 | 1.0000 |
| stress | Hybrid-H3 | pre_speed_2 | 1.5410 | 56.58% | 76.41 | 1.0000 |
| stress | Rule | post_speed_1 | 1.2703 | 88.15% | 28.67 | 1.0000 |
| stress | R-Fixed | post_speed_1 | 1.2862 | 75.92% | 79.31 | 1.0000 |
| stress | R-Decay | post_speed_1 | 1.2294 | 83.28% | 78.22 | 1.0000 |
| stress | Plan-H1 | post_speed_1 | 1.2027 | 92.69% | 32.26 | 1.0000 |
| stress | Plan-H2 | post_speed_1 | 1.1506 | 84.47% | 80.87 | 1.0000 |
| stress | Plan-H3 | post_speed_1 | 1.1383 | 84.94% | 81.77 | 1.0000 |
| stress | Plan-H3-Zero | post_speed_1 | 2.1828 | 81.08% | 77.91 | 1.0000 |
| stress | Hybrid-H3 | post_speed_1 | 1.1336 | 85.30% | 82.25 | 1.0000 |
| stress | Rule | post_speed_2 | 1.6931 | 47.85% | 54.42 | 1.0000 |
| stress | R-Fixed | post_speed_2 | 1.8616 | 49.16% | 68.73 | 1.0000 |
| stress | R-Decay | post_speed_2 | 1.8731 | 49.54% | 70.73 | 1.0000 |
| stress | Plan-H1 | post_speed_2 | 1.5817 | 51.42% | 63.32 | 1.0000 |
| stress | Plan-H2 | post_speed_2 | 1.6026 | 52.94% | 75.89 | 1.0000 |
| stress | Plan-H3 | post_speed_2 | 1.6014 | 52.87% | 76.61 | 1.0000 |
| stress | Plan-H3-Zero | post_speed_2 | 2.7447 | 50.41% | 72.62 | 1.0000 |
| stress | Hybrid-H3 | post_speed_2 | 1.5870 | 53.60% | 77.15 | 1.0000 |
| stress | Rule | boundary | 1.5789 | 67.62% | 45.90 | 1.0000 |
| stress | R-Fixed | boundary | 1.6841 | 62.38% | 69.87 | 1.0000 |
| stress | R-Decay | boundary | 1.6618 | 66.19% | 70.95 | 1.0000 |
| stress | Plan-H1 | boundary | 1.4844 | 71.73% | 52.48 | 1.0000 |
| stress | Plan-H2 | boundary | 1.4682 | 68.77% | 77.66 | 1.0000 |
| stress | Plan-H3 | boundary | 1.4588 | 69.12% | 78.72 | 1.0000 |
| stress | Plan-H3-Zero | boundary | 3.4112 | 63.46% | 72.14 | 1.0000 |
| stress | Hybrid-H3 | boundary | 1.4601 | 69.34% | 79.27 | 1.0000 |
| stress | Rule | interior | 1.3700 | 68.25% | 36.58 | 1.0000 |
| stress | R-Fixed | interior | 1.4422 | 62.61% | 79.03 | 1.0000 |
| stress | R-Decay | interior | 1.4193 | 66.51% | 78.74 | 1.0000 |
| stress | Plan-H1 | interior | 1.2872 | 72.22% | 42.51 | 1.0000 |
| stress | Plan-H2 | interior | 1.2727 | 68.47% | 79.20 | 1.0000 |
| stress | Plan-H3 | interior | 1.2702 | 68.50% | 79.70 | 1.0000 |
| stress | Plan-H3-Zero | interior | 1.2935 | 68.42% | 79.10 | 1.0000 |
| stress | Hybrid-H3 | interior | 1.2477 | 69.41% | 80.18 | 1.0000 |

The frozen −5…+15 event windows, 30-tick same-type overlap exclusion, and three-consecutive-close recovery definition are reused. Recovery times among recovered events are conditional and accompanied by censor counts. Pre-action alignment recovery after bounce is separately defined as first aligned decision within 30 ticks; no recovery is censored, not scored as zero. See evaluation/*/recovery.csv and behavior_selection.json.

Behavior clips use the earliest event under prespecified rules: speed-2 same-state disagreement; bounce with different subsequent alignment recovery; aligned R-Decay Fire versus planned movement. Absence is explicitly recorded. behavior_fragments.csv preserves every tick in each −5…+15 fragment. It distinguishes planner decisions on the R state from the planner's own evolving rollout; they must not be conflated. No clip is selected for better planner performance.

## Runtime and computation accounting

Complete DP precomputation (including saved bundles) took 26.2726 s. It performed 65,625,000 actual state value backups, 383,250,000 chance transition terms and 656,250,000 action-value lookups. Shared chance-expectation cache reuse was 590,625,000 lookups; this is reuse of exact successor expectations, not sampling or deterministic target approximation. DP working arrays use 78,125,000 bytes and transition/movement maps 377,500 bytes. Peak precomputation-process working set was 142.51 MiB (process-lifetime metric). Build time, per-pass core compute, cache size/hash and Python API first/warm timings are recorded separately.

Formal evaluation used 10,000 Final plus 4,000 Stress windows, exactly 4,200,000 environmental ticks, with 1,890,000 planned-controller table queries and no online tree expansions or value backups. Formal evaluation time (including saved complete rollouts and behavior selection) was 35.7993 s. Development: 4,000 windows / 1,200,000 ticks. Validation and demonstration counts are separate. A cached lookup does not erase the above startup cost.

Native step timing includes proposal plus firing guard and timer overhead; excludes environment stepping, Python orchestration, loading and precomputation. Python API first-pass and warm latency is separately provided in diagnostics/api_latency.csv; the first pass is not OS-cache flushed. Neither timing measure should be sold as end-to-end cold planning latency.

| Split | Policy | Load seconds | Median ns | P95 ns | Maximum ns | Eval/storage seconds |
| --- | --- | --- | --- | --- | --- | --- |
| final | Rule | 0.0000 | 0.0 | 101.0 | 11700 | 0.4601 |
| final | R-Fixed_s60 | 0.0073 | 0.0 | 101.0 | 11300 | 0.4438 |
| final | R-Fixed_s61 | 0.0058 | 0.0 | 101.0 | 40800 | 0.4455 |
| final | R-Fixed_s62 | 0.0059 | 0.0 | 101.0 | 9801 | 0.4413 |
| final | R-Fixed_s63 | 0.0060 | 0.0 | 101.0 | 8500 | 0.4413 |
| final | R-Fixed_s64 | 0.0057 | 0.0 | 101.0 | 14400 | 0.4341 |
| final | R-Decay_s60 | 0.0055 | 0.0 | 101.0 | 14500 | 0.4588 |
| final | R-Decay_s61 | 0.0054 | 0.0 | 101.0 | 33000 | 0.4285 |
| final | R-Decay_s62 | 0.0055 | 0.0 | 101.0 | 6600 | 0.4302 |
| final | R-Decay_s63 | 0.0064 | 0.0 | 101.0 | 6500 | 0.4343 |
| final | R-Decay_s64 | 0.0058 | 0.0 | 101.0 | 9500 | 0.4315 |
| final | Plan-H1 | 0.0041 | 0.0 | 101.0 | 10100 | 0.4309 |
| final | Plan-H2 | 0.0041 | 0.0 | 101.0 | 227800 | 0.4453 |
| final | Plan-H3 | 0.0044 | 0.0 | 101.0 | 7800 | 0.4417 |
| final | Plan-H3-Zero | 0.0049 | 0.0 | 101.0 | 24300 | 0.4576 |
| final | Hybrid-H3_s60 | 0.0167 | 0.0 | 101.0 | 6200 | 0.4496 |
| final | Hybrid-H3_s61 | 0.0166 | 0.0 | 101.0 | 6800 | 0.4374 |
| final | Hybrid-H3_s62 | 0.0166 | 0.0 | 101.0 | 5400 | 0.4361 |
| final | Hybrid-H3_s63 | 0.0163 | 0.0 | 101.0 | 43100 | 0.4388 |
| final | Hybrid-H3_s64 | 0.0164 | 0.0 | 101.0 | 5300 | 0.4404 |
| stress | Rule | 0.0000 | 0.0 | 101.0 | 6000 | 0.2047 |
| stress | R-Fixed_s60 | 0.0060 | 0.0 | 101.0 | 8800 | 0.1837 |
| stress | R-Fixed_s61 | 0.0059 | 0.0 | 101.0 | 7100 | 0.1928 |
| stress | R-Fixed_s62 | 0.0059 | 0.0 | 101.0 | 6500 | 0.2095 |
| stress | R-Fixed_s63 | 0.0060 | 0.0 | 101.0 | 5699 | 0.1829 |
| stress | R-Fixed_s64 | 0.0059 | 0.0 | 101.0 | 6200 | 0.1847 |
| stress | R-Decay_s60 | 0.0060 | 0.0 | 101.0 | 8100 | 0.1831 |
| stress | R-Decay_s61 | 0.0057 | 0.0 | 101.0 | 1299 | 0.1853 |
| stress | R-Decay_s62 | 0.0059 | 0.0 | 101.0 | 6100 | 0.1811 |
| stress | R-Decay_s63 | 0.0059 | 0.0 | 101.0 | 5500 | 0.1813 |
| stress | R-Decay_s64 | 0.0067 | 0.0 | 101.0 | 40500 | 0.1834 |
| stress | Plan-H1 | 0.0038 | 0.0 | 101.0 | 24700 | 0.2293 |
| stress | Plan-H2 | 0.0058 | 0.0 | 101.0 | 6600 | 0.1875 |
| stress | Plan-H3 | 0.0045 | 0.0 | 101.0 | 6000 | 0.1841 |
| stress | Plan-H3-Zero | 0.0041 | 0.0 | 101.0 | 6299 | 0.1866 |
| stress | Hybrid-H3_s60 | 0.0169 | 0.0 | 101.0 | 401 | 0.1829 |
| stress | Hybrid-H3_s61 | 0.0190 | 0.0 | 101.0 | 599 | 0.1967 |
| stress | Hybrid-H3_s62 | 0.0166 | 0.0 | 101.0 | 56700 | 0.1878 |
| stress | Hybrid-H3_s63 | 0.0166 | 0.0 | 101.0 | 401 | 0.1840 |
| stress | Hybrid-H3_s64 | 0.0167 | 0.0 | 101.0 | 5501 | 0.2056 |

## Artifacts and validation

- [Measured interpretation](conclusions.md), [paper addendum](paper_addendum.md), [reproduction](REPRODUCE.md).
- [Horizon response](figures/horizon_response.png), [terminal comparison](figures/terminal_values.png), [D/C/H](figures/D_C_H.png), [paired Hybrid](figures/paired_hybrid.png), [dynamic events](figures/dynamic_events.png), [behavior fragments](figures/behavior_fragments.png).
- [Demo 6400000](demos/scenario_6400000/synchronized.gif), [demo 6400001](demos/scenario_6400001/synchronized.gif), [demo 6400002](demos/scenario_6400002/synchronized.gif).
- Every GIF contains initial state plus all 300 ticks. All 301 pixel arrays and timing entries are decoded/compared; total duration is 20,070 ms. Actual player inspection is deliberately not performed, as requested; no browser/desktop permission was requested.
- Source/model/cache/scenario freeze is in protocol.json. Q and counts are read-only and protected inputs are hashed. Historical receipt writes are redirected to this study; all previous uncommitted work remains unchanged. Tests and verification commands record actual exits/timings in execution/. No commit or push is performed.

## Behavior interpretation supplement

See [the measured speed-two, movement and threshold interpretation](behavior_interpretation.md). This post-evaluation reading changes no frozen method, metric or gate.

## Final acceptance

All 96 complete-suite tests and all seven new optimized-mode tests passed. Historical publication, reward/guard-study and previous-boundary canonical verifiers passed in ordinary and optimized Python, with old receipt writes redirected outside protected files. Both new saved-artifact verifiers passed, including exhaustive reconstruction of all 21 DP layers and comparison of 5.4 million stored policy proposals. Formal environment evaluations were not replayed. All 2,972 protected historical and previously uncommitted files remain unchanged. See final_acceptance.json and execution/. The delivery manifest is checked separately after creation; receipts are excluded from their own hashes. Actual-player inspection was intentionally not executed by this protocol.
