# Paper addendum: finite-horizon stochastic feedback planning

This zero-training study separates horizon from terminal value inside a full-state, known-model planner family. A complete physical-state finite-horizon dynamic program evaluates all ten original actions and merges identical stochastic successor states. Shared successor expectations avoid rebuilding a random tree per tick while preserving the feedback Bellman recurrence. Cached policies are indexed solely by current physical observation; model probabilities remain 0.05 under the real 0.10 Stress process.

The terminal conversion follows shaped reward = task reward + gamma Phi(next) - Phi(current). A task value equals shaped value plus Phi; accordingly the learned terminal is max(Q_shaped)+Phi, with no fitted coefficient. An explicit shaping/terminal equivalence test preserves root action rankings up to the root-state constant. The complete precomputation, numerical precision, tie rule and cache identities are frozen before Final access.

The same-information H1/H2/H3 contrasts test horizon. H3 versus Zero tests distance heuristic; five Hybrid instances test five distinct frozen terminal estimates and are compared to both Phi planning and their same-seed R-Decay policies. Planners additionally observe full coordinates and use a known motion model, so their differences from relative-state R cannot be assigned exclusively to search depth. Pure planners have one instance, not five independent training seeds. Scenario differences are paired; ticks are not treated as independent statistical replicates. No confidence intervals or significance claims are made.

All prespecified methods are retained, including any poor zero-terminal or hit/close tradeoff controller. The 75% close target is separated from the joint relative gate and hit retention. Speed-before-decision is separated from actual speed-after-toggle; event and trajectory slices are descriptive, with common-state action diagnostics as a matched complement. Complete event clips follow frozen earliest-event rules, not an outcome-based highlight selection. See conclusions.md for the measured results and implications; a negative horizon or terminal result does not establish an optimality bound for the action set.

## Behavior interpretation supplement

See [the measured speed-two, movement and threshold interpretation](behavior_interpretation.md). This post-evaluation reading changes no frozen method, metric or gate.

## Final acceptance

All 96 complete-suite tests and all seven new optimized-mode tests passed. Historical publication, reward/guard-study and previous-boundary canonical verifiers passed in ordinary and optimized Python, with old receipt writes redirected outside protected files. Both new saved-artifact verifiers passed, including exhaustive reconstruction of all 21 DP layers and comparison of 5.4 million stored policy proposals. Formal environment evaluations were not replayed. All 2,972 protected historical and previously uncommitted files remain unchanged. See final_acceptance.json and execution/. The delivery manifest is checked separately after creation; receipts are excluded from their own hashes. Actual-player inspection was intentionally not executed by this protocol.
