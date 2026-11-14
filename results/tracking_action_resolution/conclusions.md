# Conclusions: adding two-cell actions

| Split | Terminal | ΔD | ΔC (pp) | ΔH | H change |
| --- | --- | --- | --- | --- | --- |
| final | Plan | -0.095613 | +0.8833 | +23.5280 | +26.6244% |
| final | Hybrid | -0.127820 | +1.2223 | +23.9204 | +26.9335% |
| stress | Plan | -0.124733 | +2.0617 | +23.8300 | +29.9334% |
| stress | Hybrid | -0.162763 | +2.2567 | +24.3280 | +30.4275% |

| Split | Terminal | Level | Seed | Practical | Stronger | C≥75% | C≥75% + H retained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | Plan | instance | None | False | False | True | True |
| final | Hybrid | instance | 60 | False | False | True | True |
| final | Hybrid | instance | 61 | False | False | True | True |
| final | Hybrid | instance | 62 | False | False | True | True |
| final | Hybrid | instance | 63 | False | False | True | True |
| final | Hybrid | instance | 64 | False | False | True | True |
| stress | Plan | instance | None | False | False | False | False |
| stress | Hybrid | instance | 60 | False | False | False | False |
| stress | Hybrid | instance | 61 | False | False | False | False |
| stress | Hybrid | instance | 62 | False | False | False | False |
| stress | Hybrid | instance | 63 | False | False | False | False |
| stress | Hybrid | instance | 64 | False | False | False | False |
| final | Plan | family_mean | None | False | False | True | True |
| final | Hybrid | family_mean | None | False | False | True | True |
| stress | Plan | family_mean | None | False | False | False | False |
| stress | Hybrid | family_mean | None | False | False | False | False |

The matched effects are consistently favorable under both terminals and all five source models. Prioritize a separately preregistered fourteen-action training study to test whether a learned controller can exploit this control option without full-state model-based DP. Keep this zero-training action-resolution result as a complete paper section; do not silently retune this holdout. The current learned terminal remains ten-action-derived.

Interpret finite-horizon value monotonicity separately from realized paired-window metrics. A14 provides extra control authority under exactly the same state/model/reward/terminal. It does not establish a trained A14 Q-learning policy; the learned terminal still comes from A10. Report Final and Stress separately and retain all five source models and every regression. The +5pp closeness requirement is not replaced by statistical significance or an absolute threshold.

Next-step decision rule: if closeness and hits improve consistently but the practical gate remains unmet, a separately frozen fourteen-action training study is a justified follow-up, not an already demonstrated learned result. If useful aggregate gains fail to retain hits or are inconsistent, diagnose horizon/reward/controller behavior on Development before any new holdout. If both gains are negligible, prioritize writing the paper and documenting the negative result. The present report and paired tables determine which branch is supported; no further action scheme is silently tuned here.


[Detailed behavioral interpretation](behavior_interpretation.md) (post-analysis narrative from the frozen tables; no experiment change).
