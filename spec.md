# AimRL Specification (Grid5 benchmark)

## Research Question and Scope

Under an equal environment-interaction budget, does distance-based potential reward shaping help Q-learning learn to hit a moving target sooner? How does changing the target movement probability affect the trained policies? The project compares a rule policy, base-reward Q-learning, and potential-shaped Q-learning.

The rule policy fires when aligned; otherwise it corrects the horizontal position first and then the vertical position. This is a small reproducible experiment. Potential-based reward shaping is an established method; the contribution is the environment implementation, controlled comparison, and analysis, not a new algorithm. Reinforcement learning is not assumed to outperform the rule policy. Preliminary prototypes established feasibility only; formal three-seed results are recorded separately under `results/grid5_benchmark/`.

## Environment, Coordinates, and State

The field is a 5×5 grid. The x coordinate increases to the right and y increases downward; both range from 0 to 4. Each reset randomly chooses distinct crosshair `(xc, yc)` and target `(xt, yt)` positions. The target row `yt` remains fixed within an episode, and its initial direction is randomly left or right.

The state is `s = (xc, yc, xt, yt, v)`, with `v = -1` for left and `v = +1` for right. Q-table direction index 0 denotes left and index 1 denotes right. The planned and implemented table has shape `(5,5,5,5,2,6)` and dtype float32: 1,250 state combinations, 7,500 Q values, and 30,000 bytes of element data, excluding other Python-process overhead. Observations contain directly available coordinates and do not depend on display resolution.

| ID | Action | Effect |
|---:|---|---|
| 0 | left | Decrease crosshair x by 1 |
| 1 | right | Increase crosshair x by 1 |
| 2 | up | Decrease crosshair y by 1 |
| 3 | down | Increase crosshair y by 1 |
| 4 | stay | Leave the crosshair unchanged |
| 5 | fire | Fire at the current crosshair position without moving it |

Movement is at most one cell and is clipped at grid boundaries. Action IDs match the order in `configs/base.json`.

Each step follows this exact order:

1. Validate the lifecycle and action, execute the crosshair action, and increment the step count.
2. If the action is fire, check whether the current crosshair and target positions coincide exactly.
3. If hit, terminate immediately; do not move the target or sample its direction.
4. Otherwise, reverse the target direction with probability 0.10.
5. Independently attempt a one-cell target move with probability 0.25. If the current direction would cross a boundary, reverse first and then move one cell.
6. Compute the reward and return the actual next state, including the updated target.

Moving onto the target is not a hit; firing is required. A missed shot remains a miss if the target subsequently moves into the crosshair cell. Direction can change without movement. Every non-hit step draws turning and movement randomness in that order.

## Lifecycle and Time Limits

`reset(seed=None)` returns `(observation, info)`. An integer seed initializes the environment's independent NumPy generator; no seed continues its existing stream. Observations are immutable tuples, and each info dictionary is new and includes `step_count`, `hit`, `miss`, and `base_reward`. Python and NumPy integer actions 0–5 are accepted; booleans and other values are rejected without changing the state, count, or random stream. Call reset before the first step and after an episode ends.

A hit sets `terminated=True`. Reaching 50 steps without a hit sets `truncated=True` and `terminated=False`. A hit on step 50 sets `terminated=True` and `truncated=False`. The limit is an external sampling/evaluation bound; remaining time is not part of the state, and timeout is not an additional finite-horizon terminal state.

Learning does not bootstrap after a true hit. External truncation still bootstraps from the actual next state before reset. The environment does not reset automatically. The interface follows the reset/step convention above without requiring Gymnasium.

## Rewards and Learning

Base reward `rA` is −0.05 per valid step, plus a hit bonus of +10 or a miss penalty of −1. Therefore a hit step yields +9.95, a missed shot −1.05, and other steps −0.05. Boundary-clipped movements have the same step cost. Truncation adds no separate reward.

```text
Phi(s) = -(|xc-xt| + |yc-yt|)
rB = rA + shaping_scale * [gamma * Phi(s_next) - Phi(s)]
```

`shaping_scale=1.0`; gamma is shared with `learning.gamma`. A true terminal next state has potential 0. External truncation does not clear the potential. The next state is the actual target-updated state. The environment returns only the base reward; the training loop applies the transformation exactly once. No heuristic Q initialization or rule-policy action substitution is used.

Q-learning updates `Q(s,a) ← Q(s,a) + alpha * [target - Q(s,a)]`. For true termination, `target = r`. Otherwise, including external truncation, `target = r + gamma * max_a Q(s_next,a)`. Both methods use identical states, actions, zero initialization, budgets, and exploration schedules. Training breaks maximum-Q ties randomly; evaluation uses fixed action order. Logged base, learning, and shaping returns are undiscounted sums within an episode. Returns from different reward modes are not compared as performance scores.

## Frozen Formal Experiment Protocol

- Methods: base and potential; training root seeds 0, 1, and 2; 100,000 interactions per run; alpha 0.15; gamma 0.95; initial Q 0, float32.
- Epsilon decreases linearly from 1.00 to 0.05 during the first 80,000 completed global training steps and then remains fixed. It does not restart after reset.
- Training target movement probability is 0.25 and turning probability is 0.10. Independent environment and exploration streams are derived from each root seed. Evaluation does not advance training random streams.
- Validate at step 0 and every 10,000 updates, using 200 scenarios with seeds 100000–100199, epsilon 0, and a frozen Q table.
- Fix the final model at step 100,000. Test each model on 500 scenarios, seeds 200000–200499, for movement probabilities 0.10, 0.25, and 0.50. Turning remains 0.10. Final tests are not used for tuning or checkpoint selection.
- Evaluate the rule and random baselines once per condition using the same scenario seeds. They do not have three training seeds. Random baseline actions use independent per-episode `SeedSequence([314159, scenario_seed])` generators.
- Report success within 50 steps, mean steps across all attempts (timeouts count as 50), mean misses, and mean undiscounted base return. Each completed episode satisfies `base_return = 10 * success - 0.05 * steps - misses` within floating-point tolerance.
- For learning efficiency, find the earliest adjacent validation checkpoints both reaching success ≥90%; record the first and the confirmation checkpoint. If not observed, store `achieved=false` and null step values. Resolution is 10,000 steps.
- Aggregate each training seed first, then compute means and sample standard deviations across the three seeds. They describe between-training-seed variation, not 95% confidence intervals. Do not treat evaluation episodes as independent training runs.

## Implementation Status and Provenance

Grid5 benchmark completed the three-seed experiments and independent final tests. The original frozen protocol remains in `results/reward_shaping_development/protocol_frozen.json`; the actual run manifest is in `results/grid5_benchmark/protocol_used.json`. Source and model hashes in these historical records identify the pre-translation experiments and are preserved unchanged. Later English translation and provenance-validation adaptations are documented in `provenance/README.md` and `provenance/source_compatibility.json`; translated source files are not claimed to have produced the earlier results.

The Q-learning development stationary-target diagnostic used seed 42, 10,000 steps, and an 8,000-step epsilon decay. Validation success increased from 0% to 13.5%. Its limited read-only replay is documented in `results/reward_shaping_development/static_diagnostic_note.md`. The Q-learning development moving-target run used seed 0 and 100,000 steps, reaching 100% from 0%; reward-shaping development base reproduced its Q table and all validation checkpoints.

Earlier environment diagnostics diagnostics passed 10 environment unit tests. On 100 scenarios with seeds 300000–300099, the rule policy achieved 100% success, 5.07 mean steps, and 0.00 mean misses; the random policy achieved 25% success, 43.77 mean steps, and 6.89 mean misses. These diagnostics are separate from the later validation and formal test sets. The saved preview was visually checked. The initial user-directory font-cache warning did not prevent image export and was subsequently resolved by the project-local cache.

The reward-shaping development seed-0 development comparison gave base/potential success of 70%/100% and mean steps of 26.605/6.990 at step 10,000. Their first confirmed ≥90% checkpoints were 20,000/10,000 steps. These development observations are separate from the three-seed results. The formal results and limitations are in `results/grid5_benchmark/summary/results_summary.md`.

The environment is small and fully observable, with only three training seeds and a finite scenario set. Training seed 0 was used during development. Independent scenario seeds do not guarantee unseen initial states. The rule policy is not proven optimal. Descriptive results do not establish statistical significance, real-game generalization, image-based control, or performance in arbitrary environments. Headless figures and model previews were checked; the interactive GUI remains unverified. The project-local Matplotlib cache resolved the earlier user-directory cache warning.

The final course deliverables are an ICML two-column report with at most five body pages (references and member contributions excluded), GitHub code, and reproduction instructions. A completed report is not yet available.

## References

- Ng, Harada & Russell (1999). *Policy invariance under reward transformations: Theory and application to reward shaping*. https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- Gymnasium. *Handling Time Limits*. https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- Gymnasium. *Solving Frozenlake with Tabular Q-Learning*. https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
