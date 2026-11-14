# Paper addendum: boundary information × update stability

## Experimental question and design

We test whether omitted boundary context and persistent Q-update noise limit continuous tracking. The published relative encoding already observes relative position, target direction and speed. We cross its boundary augmentation with the prespecified per-pair learning rate alpha(n)=0.15(1+n/1000)^(-0.6), using n before the actual update. The augmentation preserves the original four features and adds each object's x/y boundary groups {0},{1},{2},{3},{4…20},{21},{22},{23},{24}. It is not fully Markov. The 56.71-fold row increase changes sample density and is part of this fixed-budget treatment.

All four cells use identical reward, exploration, target mechanism, training-seed pairing, reset distribution and 5M budget. Environment and exploration RNGs are separate. Counts are incremented once per genuine update. Q and increments are float32 with double-precision reward/target/alpha calculation. Continuing bootstrap includes truncation and budget tails. Python and native per-step records agree on all actions, states, reward parts, alpha, targets, updated Q and counts. Formal main endpoints are fixed at 5M.

## Evidence and interpretation

The boundary counterexample establishes a real alias: identical observed R state can have different deterministic next relative states near a wall. The Fire/Stay controlled experiment establishes a one-unit expected one-step-target disadvantage for every unaligned Fire, given the same frozen next-state Q. Persistent learned action misranking therefore is compatible with finite sampling, asynchronous updates, aggregation and noise; it does not imply the environment secretly rewards empty fire.

The factorial contrasts, exact results and all prespecified gates are reported in results_summary.md and accompanying tables. An effect of decay supports an update-stability explanation only to the extent that paired endpoint and checkpoint diagnostics improve. A null or negative result does not disprove boundary aliasing. With the much larger table, lower raw ranking-error rates must be checked against unvisited fractions, Bellman residuals and tracking performance. The rate schedule is intentionally unchanged despite differing visit frequencies.

Five paired training seeds, rather than 500 scenarios or individual ticks, are the training replication units. We report their sample SD, worst seed and all paired deltas without declaring statistical significance. Event slices condition on policy-induced trajectories and are descriptive. The common-state set provides a matched complement. Historical frozen three-seed models are not substituted for the new five-seed R-Fixed control. No historical reference models were newly evaluated in this study; that optional comparison is omitted.

## Subsequent research choice

Action timing creates a structural tension: valid Fire occurs before motion, consumes the tick and makes no tracking move. Speed-two target motion then places every successful shot outside distance≤1 unless reflection reduces net displacement. D/C/H tradeoffs should therefore be considered alongside any gain from state or update changes. A next experiment should compare a prespecified planning controller against the same action set, and then isolate action redesign (for example move-and-fire or changed timing) as a separate environment intervention. Eligibility traces would address temporal credit assignment, but these one-step identities and the shooting/motion conflict alone are not evidence that delayed credit is the leading bottleneck. Final evidence-specific recommendations are in the conclusion supplement.

## Evidence-specific supplement

See [the measured interpretation and next research recommendation](conclusions.md) for factor effects, seed/scene attribution, residual caveats and the actual-player limitation.

## Validation contract repair and final acceptance

The frozen artifact verifier initially passed saved int16 initial-state arrays to the historical target-hash function, whose original evaluation contract uses a six-integer tuple. NumPy-array bytes contain two bytes per int16 field, while tuple bytes contain one byte per coordinate. The explicit `verify_artifacts.py` adapter restores the actual evaluation contract. All original validation checks still execute; no expected digest, model, historical file, protocol or frozen source hash was changed. Both ordinary Python and python -O artifact checks pass with this adapter. The original failed attempts remain in postflight.json and execution/ for review. Use the corrected commands in REPRODUCE.md.

Final test evidence: full suite 89/89, new-study optimized suite 7/7, both historical publication verifiers in ordinary and optimized Python, and exhaustive frozen-artifact checks of 220 checkpoints, 100M final updates, 28,000 evaluation windows and 903 GIF frames. Additional CSV and final common-state Q diagnostics were independently recomputed from saved models. File inventory checks are recorded separately after manifest creation. Actual-player observation remains blocked as described in demos/player_check.json; decoding and static inspection passed.
