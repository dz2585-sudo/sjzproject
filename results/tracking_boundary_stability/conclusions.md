# Evidence-specific conclusions

This interpretation uses the frozen 5M artifacts. Supplementary coverage and per-scenario summaries are descriptive additions after training; no protocol, learning source, endpoint or threshold was changed.

## Which factor helped?

R-Decay improves Final C by 4.2865 percentage points (68.8989%→73.1855%), D by −0.03409 and H by +0.2744 hits/window (+0.325%). Stress C gains 3.9013 points (62.4473%→66.3487%), D changes −0.02390 and H +0.746 (+1.010%). Guarded A is 100% in all four cells. Neither split reaches the prespecified +5pp mean practical gate; H also misses the stronger +2% gate. No condition or individual training seed reaches absolute C≥75%. Only R-Decay seed 62 passes the practical per-seed gate in both splits; no seed passes the stronger simultaneous gate.

Final C sample SD falls from 1.5612pp to 0.0422pp; Stress C SD falls from 1.2536pp to 0.1967pp. This is clear stabilization of C in these five seeds. It is not uniform stabilization of every metric: Final H SD rises slightly (0.0887→0.1011), and Stress D SD rises (0.0123→0.0205). Stress D worsens for seed 63 despite its C/H gains.

All five relative-representation seeds improve C, so the benefit is not solely rescue of one bad seed. Seed 62 has the largest gain (+7.0593pp Final, +6.0967pp Stress); other Final seeds gain 3.3987–3.7033pp. This explains part of the reduction in worst-seed risk without attributing the full mean gain to one seed.

| Split | Seed | R-Fixed C | R-Decay C | C difference | D difference | H difference |
| --- | --- | --- | --- | --- | --- | --- |
| final | 60 | 69.653% | 73.229% | +3.575 pp | -0.0187 | +0.312 |
| final | 61 | 69.502% | 73.198% | +3.696 pp | -0.0522 | +0.238 |
| final | 62 | 66.111% | 73.171% | +7.059 pp | -0.0521 | +0.396 |
| final | 63 | 69.721% | 73.120% | +3.399 pp | -0.0147 | +0.336 |
| final | 64 | 69.507% | 73.210% | +3.703 pp | -0.0329 | +0.090 |
| stress | 60 | 63.200% | 66.437% | +3.237 pp | -0.0126 | +0.690 |
| stress | 61 | 62.892% | 66.440% | +3.548 pp | -0.0574 | +0.910 |
| stress | 62 | 60.237% | 66.333% | +6.097 pp | -0.0327 | +1.055 |
| stress | 63 | 63.200% | 66.017% | +2.817 pp | +0.0239 | +0.540 |
| stress | 64 | 62.708% | 66.517% | +3.808 pp | -0.0408 | +0.535 |

The boundary augmentation worsens D, C and H against the matching R condition in all five seeds, with both learning-rate settings and both splits. At fixed alpha its Final C effect is −12.5151pp and H −23.6164; under decay C is −15.3604pp and H −14.9376. The counterexample still proves missing boundary context; the fixed-budget result does not support this particular larger tabular encoding as a practical improvement. Sample dilution is a plausible explanation, supported by coverage below, but was not independently randomized and is not proven as the sole cause.

Within B, decay improves Final C in four of five seeds and H in all five; seed 61 loses C and worsens D. Its Final mean C gain is only 1.4412pp, with H +8.9532. The C interaction is negative (−2.8453pp Final, −2.3927pp Stress); the H interaction is positive (+8.6788 Final, +6.586 Stress). Thus the two factors do not yield an additive C improvement.

| Condition | Rows visited | Pairs visited | Mean seed median updates / visited pair |
| --- | --- | --- | --- |
| R-Fixed | 96.73% | 77.84% | 5.0 |
| R-Decay | 96.70% | 77.75% | 5.0 |
| B-Fixed | 52.26% | 11.77% | 1.0 |
| B-Decay | 52.18% | 11.75% | 1.0 |

## Which scenes improved?

The following slices use each policy's own physical trajectory and preserve the historical slice definitions. They are descriptive; moving into a different state distribution is part of the observed result.

| Split | Slice | R-Fixed C | R-Decay C | C difference | D difference |
| --- | --- | --- | --- | --- | --- |
| final | boundary | 68.70% | 72.92% | +4.22 pp | -0.0308 |
| final | interior | 69.15% | 73.52% | +4.37 pp | -0.0382 |
| final | speed_1 | 81.94% | 90.43% | +8.49 pp | -0.0724 |
| final | speed_2 | 56.27% | 56.48% | +0.22 pp | +0.0030 |
| stress | boundary | 62.47% | 66.27% | +3.79 pp | -0.0193 |
| stress | interior | 62.42% | 66.45% | +4.03 pp | -0.0300 |
| stress | speed_1 | 75.82% | 83.30% | +7.48 pp | -0.0600 |
| stress | speed_2 | 49.41% | 49.82% | +0.41 pp | +0.0113 |

Per-scenario C differences below first average over the five paired trained seeds. Counts describe scene breadth; they do not turn scenes into independent trained models. All scene/seed differences are retained in scenario_differences.csv.

| Split | Condition | C improved scenes | C worsened scenes | Tied |
| --- | --- | --- | --- | --- |
| final | R-Decay | 500 | 0 | 0 |
| final | B-Fixed | 0 | 500 | 0 |
| final | B-Decay | 0 | 500 | 0 |
| stress | R-Decay | 200 | 0 | 0 |
| stress | B-Fixed | 0 | 200 | 0 |
| stress | B-Decay | 0 | 200 | 0 |

## Did firing ranking and Bellman stability improve?

On distinct common unaligned physical states, Fire>Stay falls from 39.611% to 32.778% with R decay; event-weighted rates fall from 39.176% to 29.884%. Greedy invalid Fire on these same states falls from 1.347% to 0.623% (state weight) and 1.690% to 0.542% (event weight). Mean absolute Q change over 4.5M→5M drops 0.8685→0.7169 with state weighting. This supports a finite-update stability contribution.

It does not establish convergence or make fixed alpha the sole cause: common-state Fire Bellman MAE is 65.892→66.013, and Stay MAE is 65.420→65.801, so those residuals do not improve. The common set includes far and poorly visited states; large residuals and the representation's aliasing remain. The unit Fire/Stay one-step target identity is fully verified despite learned misranking.

B has lower common-state error fractions but about 36.7% of Fire pairs are unvisited versus 2.6% in R (state weighting). Zero-tied rows reduce strict Fire>Stay fractions. B also makes more greedy invalid Fire on common states and performs worse physically. Lower strict misranking fractions or smaller absolute Bellman residuals alone are insufficient evidence that its tracking solution is better.

| Split | Condition | Invalid Fire / all ticks, original | Original accuracy |
| --- | --- | --- | --- |
| final | R-Fixed | 1.5249% | 95.171% |
| final | R-Decay | 0.0349% | 99.877% |
| final | B-Fixed | 5.4301% | 79.213% |
| final | B-Decay | 2.6837% | 89.671% |
| stress | R-Fixed | 1.5130% | 94.549% |
| stress | R-Decay | 0.0420% | 99.832% |
| stress | B-Fixed | 5.2960% | 77.045% |
| stress | B-Decay | 2.6600% | 88.206% |

Most raw invalid-fire reduction for R occurs in seeds 61/62; seed 60 has a small Final increase. Empty Fire and Stay have identical physical effects. Their ranking improvement cannot by itself explain the tracking gain; other movement-action values must also change. Guard preserves every physical metric and hit, so the tracking improvement above survives removal of empty-fire penalties.

## Recommended next research

Prioritize a controlled planning baseline with the current action set. One-step lookahead is already represented by the historical Rule policy; the next planning study should test a prespecified multi-step horizon with stochastic target outcomes and the same observation/action constraints. It can test whether temporal action coordination raises the C/H frontier before altering environment semantics. The present evidence does not establish that the existing actions impose a 75% ceiling.

Action design is the next separate factor if planning exposes a persistent C/H conflict: valid Fire happens before target motion and consumes a stationary tick, while speed-two motion often moves a hit outside the close threshold. A move-and-fire action or changed timing would alter the task and must not be mixed into the present factorial comparison. Eligibility traces are lower priority from this evidence: decay stabilizes C and ranking, but the remaining boundary/sample-density and action-timing issues are not specifically evidence of delayed credit assignment. This is a research recommendation, not a tested ranking of the three proposed interventions.

## Remaining validation limitation

All three GIFs contain the initial frame plus every one of the 300 ticks. Decoding verified all 903 frame pixel arrays and exact per-frame timings. Static layout was visually inspected. Actual browser/player playback was not verified: the Windows Computer Use tool returned “Computer Use was not approved to use Google Chrome”; the in-app browser then rejected local file URLs by its protocol policy. No security setting or alternate transport was used to bypass those controls. See demos/player_check.json. This is the only requested validation left incomplete.
