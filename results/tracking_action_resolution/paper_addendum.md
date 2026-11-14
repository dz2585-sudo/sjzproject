# Paper addendum: action-resolution intervention

We compare A10 against its strict A14 superset in a fixed three-step stochastic feedback planner. Four cardinal two-cell moves consume one tick each. Five frozen ten-action learned terminals and one potential terminal are reused without training. The same scene seeds and target routes provide paired controls; Stress tests real dynamics at twice the assumed random-event probability.

| Split | Controller | D | C | H / 300 ticks | A |
| --- | --- | --- | --- | --- | --- |
| final | Plan-H3-A10 | 1.2120 | 74.6447% | 88.3700 | 100.0000% |
| final | Plan-H3-A14 | 1.1164 | 75.5280% | 111.8980 | 100.0000% |
| final | Hybrid-H3-A10 | 1.2023 ± 0.0034 | 75.0547% ± 0.0492 | 88.8128 ± 0.0227 | 100.0000% ± 0.0000 |
| final | Hybrid-H3-A14 | 1.0745 ± 0.0004 | 76.2769% ± 0.0037 | 112.7332 ± 0.0041 | 100.0000% ± 0.0000 |
| stress | Plan-H3-A10 | 1.3487 | 69.6750% | 79.6100 | 100.0000% |
| stress | Plan-H3-A14 | 1.2240 | 71.7367% | 103.4400 | 100.0000% |
| stress | Hybrid-H3-A10 | 1.3422 ± 0.0052 | 70.1420% ± 0.0494 | 79.9540 ± 0.0778 | 100.0000% ± 0.0000 |
| stress | Hybrid-H3-A14 | 1.1794 ± 0.0009 | 72.3987% ± 0.0147 | 104.2820 ± 0.0135 | 100.0000% ± 0.0000 |

| Split | Terminal | ΔD | ΔC (pp) | ΔH | H change |
| --- | --- | --- | --- | --- | --- |
| final | Plan | -0.095613 | +0.8833 | +23.5280 | +26.6244% |
| final | Hybrid | -0.127820 | +1.2223 | +23.9204 | +26.9335% |
| stress | Plan | -0.124733 | +2.0617 | +23.8300 | +29.9334% |
| stress | Hybrid | -0.162763 | +2.2567 | +24.3280 | +30.4275% |

Exhaustive finite-state subset-value inequalities pass at every depth for all six terminals (56.25M comparisons, zero violations). Their relevance is limited to the planning objective. Realized D/C/H, source-model sample variability, paired scene/model bootstrap intervals and predeclared gates are reported separately. Full observability, known dynamics, offline computation and an A10-derived terminal limit transfer claims. No A14 learned policy or real-game result is claimed.

[Full report](results_summary.md) · [Protocol](protocol.json) · [Reproduce](REPRODUCE.md)


[Detailed behavioral interpretation](behavior_interpretation.md) (post-analysis narrative from the frozen tables; no experiment change).
