# Paper addendum · Q-learning + firing guard

## Method

Q-learning + firing guard queries the original greedy policy once per tick. It substitutes Stay only for Fire with nonzero pre-motion Manhattan distance. Both actions leave the crosshair fixed and consume the same target randomness. For this event the actual reward rises by exactly one because the miss penalty is removed. The transition, continuation term and proximity reward are identical for the paired action. The true environment therefore prefers Stay in this case, while the measured greedy tables preferred Fire. No Q-table or update count changed.

The original frozen policies remain a separate method. The guard was evaluated with the same held-out target trajectories on all 21 learned policies and with the fixed Rule comparator. All learned results preserve the three training seeds and the original reward conditions.

## Held-out final results

| Condition | D | C (%) | H / window | A mean, original → guard (%) | A pooled, original → guard (%) | Misses, original → guard | Base return, original → guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 1.526 ± 0.135 | 58.41 ± 4.69 | 81.22 ± 4.78 | 99.32 ± 0.86 → 100.00 ± 0.00 | 99.35 → 100.00 | 0.53 → 0.00 | 796.63 → 797.16 |
| distance_0p1 | 1.434 ± 0.013 | 60.47 ± 2.03 | 83.90 ± 0.05 | 95.43 ± 7.17 → 100.00 ± 0.00 | 95.06 → 100.00 | 4.36 → 0.00 | 819.60 → 823.97 |
| distance_0p3 | 1.484 ± 0.076 | 59.67 ± 3.05 | 80.85 ± 5.00 | 99.00 ± 0.43 → 100.00 ± 0.00 | 99.02 → 100.00 | 0.80 → 0.00 | 792.68 → 793.48 |
| distance_1p0 | 1.520 ± 0.073 | 65.73 ± 6.84 | 80.41 ± 4.92 | 95.53 ± 6.50 → 100.00 ± 0.00 | 95.11 → 100.00 | 4.14 → 0.00 | 784.98 → 789.11 |
| close_0p5 | 1.404 ± 0.021 | 65.20 ± 2.17 | 84.21 ± 0.09 | 95.59 ± 6.21 → 100.00 ± 0.00 | 95.31 → 100.00 | 4.14 → 0.00 | 823.01 → 827.15 |
| close_2p0 | 1.357 ± 0.011 | 69.71 ± 0.75 | 81.58 ± 4.88 | 90.36 ± 15.88 → 100.00 ± 0.00 | 87.94 → 100.00 | 11.19 → 0.00 | 789.58 → 800.77 |
| close_5p0 | 1.340 ± 0.053 | 73.19 ± 1.81 | 81.29 ± 4.97 | 86.20 ± 11.70 → 100.00 ± 0.00 | 84.79 → 100.00 | 14.58 → 0.00 | 783.36 → 797.95 |
| rule | 1.381 | 73.01 | 43.48 | 100.00 | 100.00 | 0.00 | 419.82 |

The previous and new original policies met the full simultaneous target for none and none, respectively. The guarded simultaneous result was `close_0p5`. The guarded retention set was `distance_1p0`, `close_0p5`, `close_2p0`, `close_5p0`. The cases with an actual hit loss inside the retention gate were `distance_1p0`. All comparisons use a matched reference on the same split and with the same variant. The full screen requires at least five percentage points more close ticks, no larger D, and accuracy at least 98%. The simultaneous hit gate is +2%. The retention gate permits a 2% loss. A passing guarded method keeps the original reward-conditioned tracking and hit count. Its eligibility changes because the guard removes misses.

All 82,391 filtered proposals had Q(Fire) > Q(Stay), with zero exact Fire–Stay ties. There were 115 events with a maximum tie involving another action. The final event-weighted median gap was 1.105, with median Fire and Stay visitation counts 12,364 and 40,318. On the final set, 114 events had an unvisited Fire or Stay action and 581 had fewer than 10 updates on at least one. Both actions had at least 100 updates on 58,809 events (98.64%). At pre-distance one there were 58,057 events. Finite sampling is compatible with the low-count tail. Most errors persisted in frequently updated pairs. Counts and Q gaps do not identify a single cause, such as transition aliasing, persistent exploration, or unequal visitation histories. Repeated visits to the same table row are descriptive events.
