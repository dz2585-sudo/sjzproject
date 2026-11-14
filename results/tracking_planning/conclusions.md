# Measured conclusions

All results below come from the saved held-out records. No method, reward, terminal coefficient or model was selected after Final. Pure planners are single controllers; five Hybrid rows refer to five different frozen terminal Q tables.

## Final

- Plan-H1: D=1.2990, C=76.615%, H=50.756, A=100.00%; versus R-Decay mean, C +3.303pp, H ratio 0.6005. Practical=False, strong=False, absolute C75=True, C75 with retained hits=False.
- Plan-H2: D=1.2285, C=74.516%, H=87.694, A=100.00%; versus R-Decay mean, C +1.205pp, H ratio 1.0375. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Plan-H3: D=1.2219, C=74.753%, H=88.342, A=100.00%; versus R-Decay mean, C +1.442pp, H ratio 1.0451. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Plan-H3-Zero: D=2.4685, C=70.961%, H=83.802, A=100.00%; versus R-Decay mean, C -2.350pp, H ratio 0.9914. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Hybrid-H3: D=1.2134, C=75.147%, H=88.784, A=100.00%; versus R-Decay mean, C +1.835pp, H ratio 1.0504. Practical=False, strong=False, absolute C75=True, C75 with retained hits=True.
- Plan-H2 minus Plan-H1: D -0.07047, C -2.0987pp, H +36.9380.
- Plan-H3 minus Plan-H2: D -0.00663, C +0.2373pp, H +0.6480.
- Plan-H3 minus Plan-H3-Zero: D -1.24663, C +3.7920pp, H +4.5400.
- Hybrid-H3 minus Plan-H3: D -0.00851, C +0.3933pp, H +0.4416.
- Hybrid improves D in 5/5 models, C in 5/5, H in 5/5. Paired values (seed, D, C fraction, H): `[[60, -0.13114000000000003, 0.018086666666666695, 4.259999999999991], [61, -0.12793333333333345, 0.018193333333333284, 4.372], [62, -0.1360866666666667, 0.018206666666666704, 4.254000000000005], [63, -0.14761333333333315, 0.01981333333333335, 4.189999999999998], [64, -0.12308666666666679, 0.01746000000000003, 4.204000000000008]]`.
- Descriptive pre_speed_2, Plan-H3 versus R-Decay: D -0.1819, C +1.966pp, hits/300 slice ticks 85.850 versus 81.387.
- Descriptive pre_speed_2, Hybrid-H3 versus R-Decay: D -0.1973, C +2.634pp, hits/300 slice ticks 86.371 versus 81.387.
- Descriptive post_speed_2, Plan-H3 versus R-Decay: D -0.1766, C +1.851pp, hits/300 slice ticks 86.236 versus 81.704.
- Descriptive post_speed_2, Hybrid-H3 versus R-Decay: D -0.1915, C +2.453pp, hits/300 slice ticks 86.819 versus 81.704.
- Descriptive boundary, Plan-H3 versus R-Decay: D -0.1466, C +1.626pp, hits/300 slice ticks 87.533 versus 82.093.
- Descriptive boundary, Hybrid-H3 versus R-Decay: D -0.1499, C +1.839pp, hits/300 slice ticks 88.115 versus 82.093.
- Descriptive interior, Plan-H3 versus R-Decay: D -0.0903, C +1.223pp, hits/300 slice ticks 89.300 versus 87.600.
- Descriptive interior, Hybrid-H3 versus R-Decay: D -0.1053, C +1.829pp, hits/300 slice ticks 89.578 versus 87.600.

## Stress

- Plan-H1: D=1.3930, C=71.960%, H=47.860, A=100.00%; versus R-Decay mean, C +5.625pp, H ratio 0.6428. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Plan-H2: D=1.3777, C=68.630%, H=78.370, A=100.00%; versus R-Decay mean, C +2.295pp, H ratio 1.0526. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Plan-H3: D=1.3709, C=68.832%, H=79.175, A=100.00%; versus R-Decay mean, C +2.497pp, H ratio 1.0634. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Plan-H3-Zero: D=2.4651, C=65.675%, H=75.250, A=100.00%; versus R-Decay mean, C -0.660pp, H ratio 1.0107. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Hybrid-H3: D=1.3613, C=69.375%, H=79.692, A=100.00%; versus R-Decay mean, C +3.040pp, H ratio 1.0703. Practical=False, strong=False, absolute C75=False, C75 with retained hits=False.
- Plan-H2 minus Plan-H1: D -0.01540, C -3.3300pp, H +30.5100.
- Plan-H3 minus Plan-H2: D -0.00672, C +0.2017pp, H +0.8050.
- Plan-H3 minus Plan-H3-Zero: D -1.09412, C +3.1567pp, H +3.9250.
- Hybrid-H3 minus Plan-H3: D -0.00962, C +0.5433pp, H +0.5170.
- Hybrid improves D in 5/5 models, C in 5/5, H in 5/5. Paired values (seed, D, C fraction, H): `[[60, -0.1895500000000001, 0.030033333333333356, 5.170000000000002], [61, -0.16938333333333344, 0.02795000000000003, 5.289999999999992], [62, -0.1902166666666667, 0.030466666666666642, 5.284999999999997], [63, -0.22871666666666668, 0.03481666666666661, 5.25500000000001], [64, -0.17918333333333325, 0.028750000000000053, 5.174999999999997]]`.
- Descriptive pre_speed_2, Plan-H3 versus R-Decay: D -0.2867, C +3.711pp, hits/300 slice ticks 75.909 versus 69.962.
- Descriptive pre_speed_2, Hybrid-H3 versus R-Decay: D -0.3025, C +4.546pp, hits/300 slice ticks 76.409 versus 69.962.
- Descriptive post_speed_2, Plan-H3 versus R-Decay: D -0.2717, C +3.325pp, hits/300 slice ticks 76.608 versus 70.727.
- Descriptive post_speed_2, Hybrid-H3 versus R-Decay: D -0.2861, C +4.053pp, hits/300 slice ticks 77.153 versus 70.727.
- Descriptive boundary, Plan-H3 versus R-Decay: D -0.2031, C +2.931pp, hits/300 slice ticks 78.719 versus 70.952.
- Descriptive boundary, Hybrid-H3 versus R-Decay: D -0.2017, C +3.150pp, hits/300 slice ticks 79.268 versus 70.952.
- Descriptive interior, Plan-H3 versus R-Decay: D -0.1491, C +1.989pp, hits/300 slice ticks 79.697 versus 78.742.
- Descriptive interior, Hybrid-H3 versus R-Decay: D -0.1716, C +2.904pp, hits/300 slice ticks 80.180 versus 78.742.

## Interpretation and limits

Read the horizon and terminal contrasts separately. H1/H2/H3 share the same full state, fixed stochastic model, tie rule and Phi terminal, isolating the chosen finite-horizon change. H3-Zero can be poor far from the target because a short horizon sees no differentiating near-term reward; this is an acquisition/terminal issue, not proof that planning itself fails. Hybrid tests the value of a learned terminal on top of the same H3 information/model. Its effect is not a new training effect.

Any observed gain over R-Decay establishes empirical room within the original action set for those measured objectives, but combines full-state access, model knowledge and extra offline computation. A C gain with a large H loss is not joint improvement. Stress is a deliberate model-mismatch test: no planner receives the true 0.10 transition parameters. No result is a theoretical upper bound.

Behavior evidence is in behavior_selection.json and behavior_fragments.csv. The frozen first-event rules may select an unfavorable planner clip and explicitly record an absent event. Common-state actions distinguish a decision difference from on-policy distribution changes. Speed before decision and speed after actual toggle are separate measurements; post-toggle slice hit rates are descriptive, not treatment effects.

The next useful action-design experiment is a controlled 2-cell movement factor: speed-two targets and the existing 1/3-cell cursor moves create a coordination problem that finite-horizon planning can partly manage but does not remove. First preserve this planning controller as a reference, then add 2-cell moves in a separate environment intervention with the same scenarios and report both C and H. Do not claim a movement mismatch is proven as the sole remaining limit. A longer planning horizon or better calibrated learned terminal is another clean follow-up if the horizon/terminal contrasts warrant it; eligibility traces are not specifically supported by this zero-training experiment.

Computation is part of the comparison: the full physical DP sweeps and cache generation are charged before reporting O(1) online queries. There is no hidden free search. Original model counters and hashes remain unchanged. Programmatic GIF decoding is complete; actual player inspection is intentionally not executed, per request.

## Behavior interpretation supplement

See [the measured speed-two, movement and threshold interpretation](behavior_interpretation.md). This post-evaluation reading changes no frozen method, metric or gate.
