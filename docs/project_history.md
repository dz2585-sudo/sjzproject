# Research Progress and Design Decisions

This history explains how the project changed, why each change was made, what evidence was produced, and what remained unresolved. It combines the intended experience with implementation and experimental-design reasoning in one project narrative. Numerical outcomes and completed budgets are taken from the archived records linked below. Proposed future work is explicitly separated from completed work.

Return to the [project overview and result tables](../README.md). For exact current paths and migration rules, see the [artifact guide](artifact_paths.md).

## 1. Establish a small, inspectable aiming task

**Need.** Build a manageable local reinforcement-learning project inspired by aiming practice, with a short initial implementation horizon and modest resource requirements.

**Decision.** Use a coordinate-based 5 by 5 grid, discrete movement/fire actions, a known reward, and tabular Q-learning. The small table makes memory cost and state/action indexing explicit. The simulator can generate interaction experience automatically; manually supplied aiming demonstrations are not required by this learning setup.

**Implementation.** Define rewards, transition order, terminal/truncated behavior, seeds, metrics, and the planned comparison before implementing training. Centralize parameters in a configuration and provide a setup checker. The initial table contains 7,500 float32 values, or 30,000 element bytes.

**Tradeoff.** Structured coordinates remove perception uncertainty and simplify the task. They do not reproduce a real game's visual input or motor dynamics.

**Evidence.** [Specification](../spec.md), [base configuration](../configs/base.json), and [setup checker](../scripts/check_setup.py).

## 2. Separate environment correctness from learned performance

**Need.** Determine whether movement, firing, target updates, episode endings, and rendering reflect the intended rules.

**Decision.** Implement a transparent rule policy and an independent Random baseline before a learning agent. Fix action order and test edge cases, including moving onto a target without firing and a shot that precedes target motion.

**Implementation.** Add the environment, policies, automated demonstration, and environment tests. A hit ends an episode; 50 unsuccessful steps truncate it. Use deterministic scenario seeds for diagnostics.

**Outcome.** In the initial 100-scenario diagnostic, the rule succeeded in every episode with 5.07 mean steps; Random succeeded in 25% with 43.77 mean steps. These are diagnostic results, not the later formal test.

**Interpretation.** The rule shows that the task is feasible and provides a strong comparison. Random's poor behavior is expected, and retaining it makes the difficulty visible.

**Evidence.** [Rule diagnostic](../results/environment_baselines/rule_policy_diagnostic.csv), [random diagnostic](../results/environment_baselines/random_policy_diagnostic.csv), and [preview](../results/environment_baselines/rule_policy_preview.png).

## 3. Add learning and distinguish a short demo from evaluation

**Need.** Verify that interaction-based learning improves performance and that a saved policy behaves consistently after reloading.

**Decision.** Use ordinary Q-learning first. Keep evaluation independent of the training random streams and use fixed scenario sets. A few successful demonstration episodes are illustrations rather than a reliable success-rate estimate.

**Implementation.** Add epsilon-greedy training, checkpoint metrics, model serialization, exact reload checks, and tests for termination/truncation updates. A true terminal hit removes bootstrap; an external time limit retains the real next-state value.

**Outcome.** The seed-0 moving-target model reached 70% success at 10,000 updates, 99% at 30,000, and 100% at 100,000. A separate 10,000-step static-target diagnostic reached only 13.5%.

**Interpretation.** High small-grid success can result from a limited state space and a generous episode cap. The static result was retained and investigated rather than removed. Repeated-state and boundary-action behavior in five replayed failures suggested coverage/tie issues, but did not identify the complete cause.

**Evidence.** [Moving-target validation](../results/q_learning_development/base_seed0/validation.csv) and [static replay note](../results/reward_shaping_development/static_diagnostic_note.md).

## 4. Test informative intermediate feedback

**Need.** Help learning when a terminal hit is rare and distant actions receive similar feedback.

**Decision.** Add a potential difference based on negative Manhattan distance. Keep base rewards in the environment and apply shaping once in the learning update. Use identical initialization, actions, budget, and exploration between reward modes.

**Implementation.** Add the reward transformation, a development comparison, and tests that distinguish real termination from external truncation. Match gamma in the shaping formula to the learning discount. Preserve base returns separately from shaped training returns.

**Outcome.** At 10,000 interactions, seed-0 Potential had 100% success versus Base's 70%. At 100,000 both reached 100%; mean steps were 4.670 and 5.055 respectively. The first checkpoints in two consecutive >=90% validations were 10,000 for Potential and 20,000 for Base.

**Interpretation.** The useful observation was earlier learning in this development run. The checkpoint spacing prevents an exact training-speed multiplier, and one seed cannot establish robustness.

**Evidence.** [Comparison CSV](../results/reward_shaping_development/comparison/comparison.csv), [figure](../results/reward_shaping_development/comparison/comparison.png), and [frozen next-stage protocol](../results/reward_shaping_development/protocol_frozen.json).

## 5. Freeze a three-seed small-grid benchmark

**Need.** Check whether the development observation repeats and whether models tolerate different target movement probabilities.

**Decision.** Train both reward modes from zero with seeds 0, 1, and 2; use an independent final scenario set only after fixing final models. Report seed-level means and sample SDs, and evaluate each untrained baseline once per condition.

**Implementation.** Six runs of 100,000 interactions; validation every 10,000; final movement probabilities 0.10, 0.25, and 0.50. Preserve per-scenario records, paired differences, configuration/source hashes, and reward identities.

**Outcome.** Both learned methods reached 100% final success in all tested conditions. Potential used fewer mean steps than Base, while the rule was faster than both at movement probability 0.10. Small amounts of missed shooting remained in some Potential models.

**Interpretation and next decision.** Final success had reached a ceiling. A larger environment could expose the limits of the absolute-state table. The completed benchmark remains a separate result, not a discarded prototype.

**Evidence.** [Formal report](../results/grid5_benchmark/summary/results_summary.md), [learned summaries](../results/grid5_benchmark/test/learned_summary.csv), and [paired differences](../results/grid5_benchmark/summary/paired_differences.csv).

## 6. Increase scale before changing the task

**Need.** Increase aiming difficulty and provide an observable demonstration of movement over a larger area.

**Decision.** Execute a 25 by 25 development trial before attempting the proposed 50 by 50 extension. Use five million interactions per reward mode and keep the experiment small enough to inspect locally.

**Implementation.** Enlarge the coordinate space, increase the cap to 100 steps, and set gamma to 0.99. Add trial-specific configuration/loading checks, four-policy summaries, and a three-panel GIF. Preserve earlier outputs.

**Outcome.** Base achieved 6% success and Potential 100% on 200 development scenarios. The rule also achieved 100% and used fewer steps than Potential. On the fixed demo, the two successful panels finished at tick 14 and then held their final states while Base continued to its timeout.

**Interpretation.** The frozen successful panels were correct for a hit-once task. They exposed a mismatch between the desired continuous-tracking experience and the existing objective. Changing grid size alone could not fix that mismatch. Gamma and the cap also changed, preventing a grid-size-only causal comparison.

**Presentation correction.** An early GIF export had layout/legend defects. A corrected export was produced under `demo/`; the first export remains under `demo_export_attempt1/` as historical evidence and is not the featured result.

**Evidence.** [Trial report](../results/grid25_development/summary/results_summary.md), [corrected GIF](../results/grid25_development/demo/demo_comparison.gif), and [trial instructions](trial25.md).

## 7. Redesign the mechanics for continuous tracking

**Need.** Show sustained pursuit across the arena, target turns and speed changes, and variable-speed crosshair control.

**Decision.** Use fixed 300-tick continuing windows and retain target motion after every hit. Add two target speeds, cardinal heading changes, and separate slow/fast movement actions. Pair target randomness across methods so apparent differences are not caused by different target paths.

**Implementation.** Introduce a separate tracking environment and workflow. The target reflects one unit substep at a time. Fire is a separate action, checked before target motion. The full six-field absolute state leads to a 125,000,000-byte Q table. Train Base and Potential for 20 million interactions each with seed 0.

**Outcome.** Potential achieved mean distance 1.702, close fraction 67.662%, and 70.535 hits per window. Base remained weak at mean distance 16.254. The predictive rule had better proximity and accuracy, while Potential shot more and scored more hits. Both learned policies selected both movement speeds.

**Presentation and metric decisions.** Record all 301 frames, show actual pre-motion shot positions, keep panels active after hits, and verify decoded playback duration. Distinguish hit accuracy, distance, acquisition, speed selection, and recovery instead of using one success number. Overlapping recovery windows are descriptive, not isolated reaction-time experiments.

**Next decision.** Treat poor Base pursuit as a substantive failed learning result. Correct mechanics and passing tests alone cannot establish a competent policy. Diagnose table usage and compare representations at a controlled budget.

**Evidence.** [Mechanics](tracking25_v2.md), [measured results](../results/continuous_tracking/results_summary.md), and [full GIF](../results/continuous_tracking/render/seed3601000_comparison.gif).

## 8. Diagnose failure and share values across relative arrangements

**Need.** Improve distant pursuit and recovery without quietly changing the target rules or replacing learned actions with a heuristic.

**Decision.** First replay the saved policies and measure action choices, clipping, ties, far streaks, and wall/speed slices. Then cross state representation with reward mode in a four-method, three-seed study. Hold the environment, action IDs, update rule, exploration, and per-run budget fixed within this study.

**Implementation.** Encode `(dx, dy, heading, speed)` with exact offsets from -24 to 24. This merges translated arrangements and reduces the table to 768,320 float32 element bytes. Add real update counters, explicit model representation metadata, final and stress splits, predeclared practical targets, and fixed displayed training seed 10.

**Important cost.** The representation omits absolute wall information. Two states with the same relative offset can produce different next relative states after crosshair clipping or target reflection. Tests establish this aliasing. The representation is approximate; no boundary rule fallback is inserted.

**Outcome.** Relative Base and Relative Potential reduced mean final distance to 1.502 and 1.440 respectively, compared with 16.469 and 7.531 for matched absolute controls. Relative Potential passed the practical multi-metric comparison, but its 60.54% close fraction missed the 75% absolute target. Relative Base had slightly more close tracking; neither relative reward mode won every metric. Three seeds remain a descriptive comparison.

**Execution and presentation corrections.** A pre-formal test used an unsuitable temporary directory and was changed to a workspace location before experiments. A later final-figure script clarified the 5M versus historical 20M budgets using saved records. The original audit initially rejected a link to its own not-yet-created output; a preparation step created an explicitly incomplete marker, after which the unchanged audit reran its checks. Failed attempts were retained. These corrections did not retrain models or rewrite frozen experimental outcomes.

**Evidence.** [Complete report](../results/tracking_improvement/results_summary.md), [cross-seed statistics](../results/tracking_improvement/evaluation/cross_seed_summary.json), [protocol](../results/tracking_improvement/protocol.json), [execution audit](../results/tracking_improvement/execution_checks.json), and [original execution notes](tracking_improvement.md).

## 9. Publish navigable results while retaining evidence

**Need.** Make the work understandable outside the original local folder and expose the actual animations, not just screenshots or inaccessible local links.

**Decision.** Name archives by experiment purpose, present English documentation, and link code, numerical records, figures, models, and GIFs together. Preserve numerical records and frozen hashes rather than editing the past to match a new layout.

**Implementation.** Rename published result locations to functional directories, retire an unrelated explanatory folder, and record exact path/source compatibility. Package the large historical continuous models losslessly. Publish all formal representation models/counters and four full GIFs; retain disposable pilot binaries locally while documenting their omission and hashes.

**Verification design.** Distinguish the original local-workspace audit from a read-only verifier for the published checkout. Keep an exact training-source bundle, validate package contents and paired records, and preserve prior Git history. An old statement that no publication occurred remains correct for that earlier execution stage.

**Evidence.** [Archive guide](../provenance/README.md), [publication explanation](../provenance/tracking_improvement_publication.md), [publication manifest](../provenance/tracking_improvement_publication.json), and [verification record](../provenance/tracking_improvement_publication_checks.json).

## 10. Make the accumulated reasoning and results readable

**Need.** Present the full project progression item by item, with every major result visible from the README and every asset findable at its current path.

**Decision and implementation.** Reorganize the overview around the latest result, a chronological research progression, embedded evidence, a complete animation index, and a migration table. Add this decision history and a dedicated artifact guide. Keep public explanations focused on the project's requirements, choices, evidence, and tradeoffs.

**Scope.** This is a documentation/navigation change. Current result and GIF locations are documented rather than moved again. Existing configurations, model values, numerical records, frozen protocols, and executable training code are preserved. The exact README change is recorded as documentation-only compatibility; it does not widen executable-source acceptance.

## Open work

The current project is ready for a course-format report and presentation built from its archived evidence. Potential research extensions include boundary-aware state information or a separately specified tracking objective that explicitly examines firing-versus-following behavior. Neither extension, a 50 by 50 run, nor real-game control has been executed. A future study needs a fresh protocol and evaluation split instead of tuning against the current final test.
