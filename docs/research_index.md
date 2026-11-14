# AimRL research index

| Research | Question and outcome | Evidence |
| --- | --- | --- |
| [Boundary information and learning-rate stability](tracking_boundary_stability.md) | Does boundary context or update decay help? R decay improves tracking/stability; the larger B table is worse at fixed budget. The declared gates remain unmet. | [Report](../results/tracking_boundary_stability/results_summary.md), [data](../results/tracking_boundary_stability/), [GIFs](../results/tracking_boundary_stability/demos/) |
| [Short-horizon planning with learned terminal values](tracking_planning.md) | Is there room with the existing actions? Hybrid improves all five model pairs and reaches C≥75% with retained hits on Final. Relative +5pp and Stress 75% remain unmet. | [Report](../results/tracking_planning/results_summary.md), [data](../results/tracking_planning/), [GIFs](../results/tracking_planning/demos/) |
| [Tracking–shooting reward trade-offs](tracking_shooting_tradeoff.md) | Same-action reward comparison across 21 trained models. | [Report](../results/tracking_shooting_tradeoff/results_summary.md) |
| [Firing guard](tracking_fire_guard.md) | Remove empty shots without changing physical tracking. | [Report](../results/tracking_fire_guard/results_summary.md) |
| [Representation study](tracking_improvement.md) | Compare absolute and relative observations under matched budgets. | [Report](../results/tracking_improvement/results_summary.md) |

[Reproduction entry points](reproduce_latest_studies.md) distinguish saved-record analysis, frozen-model reevaluation, native rebuilding and optional exact training replication. [Publication evidence](../provenance/latest_studies_release/README.md) records all actual payload hashes, permitted documentation changes, omissions and checks. The 2-cell action study is future work.
