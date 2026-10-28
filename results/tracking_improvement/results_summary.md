# Tracking diagnostics and representation study

## Scope and integrity

The study completed 12 fresh formal runs and exactly 60,000,000 training interactions, plus 400,000 disposable pilot interactions. No pilot or historical Q values were transferred to formal runs. The environment physics, rewards, and frozen V2 source bytes were preserved. No Git commit, push, identity change, or training extension was performed.

The original project root has an unborn Git HEAD. The separately published local clone was at reviewed commit 43aa7cc9e02a91ba7a5ee9cd0986a62a269e56ff. Both facts and the original working-tree status are recorded in protocol.json. The NPZ historical packages were loaded from that publication clone, their embedded original NPY hashes were checked, and the original root source hashes matched the archived metadata.

## Archived-model diagnosis

The following values replay the original 200 development scenarios, not the independent final test. Saved zero-valued Q rows are observable model properties, not historical visitation counts. Exact ties use float32 equality; numerical near ties use a separately reported absolute tolerance of 1e-6.

| Policy | Mean distance | P95 distance | Close fraction | Far fraction | Mean longest far streak | Zero-Q fraction | Exact tie fraction | Clipped movement fraction | Local pursuit fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_base | 16.2538 | 31.0000 | 0.0262 | 0.8940 | 120.1600 | 0.0100 | 0.8977 | 0.0592 | 0.4744 |
| historical_potential | 1.7017 | 4.0000 | 0.6766 | 0.0415 | 11.9400 | 0.0020 | 0.0091 | 0.0172 | 0.5517 |
| rule | 1.3819 | 3.0000 | 0.7251 | 0.0171 | 5.0500 | undefined | undefined | 0.0008 | 0.6946 |
| random | 17.1609 | 32.0000 | 0.0063 | 0.9242 | 129.5200 | undefined | undefined | 0.0765 | 0.4603 |

Observed evidence: archived Base spends 89.39% of ticks farther than five cells, with a mean longest far streak of 120.16 ticks per window. Its saved rows are all zero on 1.00% of evaluation ticks, with exact maximal ties on 89.77%. The clipping and pursuit fractions above use movement actions as the denominator. Pursuit tests distance to the target's pre-motion position and is not an optimal-action test.

Plausible hypotheses: sparse physical-state coverage, lowest-ID choices on uninformative rows, and delayed sparse hit rewards can limit the large absolute table at a finite budget. This archive does not establish exact training visitation histories or identify a single causal explanation. Random remains an intentionally untrained negative control. Detailed action counts, unaligned shots, speed bins, wall slices, and all three demo traces are retained in diagnostics/.

## Matched results and all training seeds

All new models use five million interactions, seeds 10/11/12, and epsilon decay over four million interactions. Historical models use 20 million interactions and training seed 0; they are cross-budget references. Values below are seed-level means and sample standard deviations (ddof=1), never standard errors over correlated ticks. Accuracy is pooled within each seed before averaging.

| Split | Method | Mean distance (SD) | Close fraction (SD) | Far fraction (SD) | Hits/window (SD) | Pooled accuracy (SD) | Capped acquisition (SD) |
|---|---|---:|---:|---:|---:|---:|---:|
| final | absolute_base | 16.4690 (0.1049) | 0.0085 (0.0004) | 0.9150 (0.0038) | 0.1767 (0.0115) | 0.0120 (0.0013) | 170.1433 (7.3810) |
| final | absolute_potential | 7.5314 (0.0639) | 0.2581 (0.0035) | 0.5251 (0.0062) | 21.8727 (0.2349) | 0.7491 (0.0045) | 38.3787 (1.5894) |
| final | relative_base | 1.5016 (0.0337) | 0.6165 (0.0402) | 0.0252 (0.0027) | 84.4420 (0.1301) | 0.9940 (0.0043) | 8.8587 (0.1824) |
| final | relative_potential | 1.4403 (0.0220) | 0.6054 (0.0195) | 0.0234 (0.0015) | 84.4893 (0.1161) | 0.9949 (0.0030) | 8.2873 (0.0227) |
| stress | absolute_base | 16.4941 (0.0575) | 0.0072 (0.0003) | 0.9205 (0.0007) | 0.1183 (0.0388) | 0.0077 (0.0025) | 177.2583 (4.6171) |
| stress | absolute_potential | 8.4010 (0.0392) | 0.1845 (0.0014) | 0.5911 (0.0009) | 15.0033 (0.3399) | 0.6625 (0.0075) | 40.5283 (2.1228) |
| stress | relative_base | 1.6734 (0.0284) | 0.5596 (0.0348) | 0.0290 (0.0037) | 73.6483 (0.3313) | 0.9907 (0.0070) | 9.0250 (0.1550) |
| stress | relative_potential | 1.6221 (0.0193) | 0.5500 (0.0157) | 0.0275 (0.0016) | 73.7100 (0.1114) | 0.9910 (0.0064) | 8.6400 (0.1776) |

| Split | Method | Training seed | Mean distance | Close fraction | Far fraction | Hits/window | Accuracy | Base return |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| final | absolute_base | 10 | 16.4797 | 0.0088 | 0.9159 | 0.1900 | 0.0135 | -26.9620 |
| final | absolute_base | 11 | 16.5682 | 0.0081 | 0.9183 | 0.1700 | 0.0112 | -28.2740 |
| final | absolute_base | 12 | 16.3592 | 0.0086 | 0.9108 | 0.1700 | 0.0113 | -28.2340 |
| final | absolute_potential | 10 | 7.4874 | 0.2609 | 0.5192 | 21.6620 | 0.7500 | 194.4000 |
| final | absolute_potential | 11 | 7.5022 | 0.2592 | 0.5246 | 22.1260 | 0.7531 | 199.0060 |
| final | absolute_potential | 12 | 7.6047 | 0.2541 | 0.5315 | 21.8300 | 0.7443 | 195.8000 |
| final | relative_base | 10 | 1.4660 | 0.6630 | 0.0275 | 84.5920 | 0.9913 | 830.1740 |
| final | relative_base | 11 | 1.5060 | 0.5939 | 0.0223 | 84.3600 | 0.9917 | 827.8940 |
| final | relative_base | 12 | 1.5330 | 0.5928 | 0.0258 | 84.3740 | 0.9989 | 828.6500 |
| final | relative_potential | 10 | 1.4578 | 0.5941 | 0.0248 | 84.3880 | 0.9928 | 828.2720 |
| final | relative_potential | 11 | 1.4475 | 0.5941 | 0.0218 | 84.4640 | 0.9984 | 829.5060 |
| final | relative_potential | 12 | 1.4156 | 0.6279 | 0.0235 | 84.6160 | 0.9935 | 830.6080 |
| stress | absolute_base | 10 | 16.4891 | 0.0068 | 0.9202 | 0.0750 | 0.0051 | -28.8500 |
| stress | absolute_base | 11 | 16.4393 | 0.0074 | 0.9200 | 0.1500 | 0.0102 | -28.1250 |
| stress | absolute_base | 12 | 16.5540 | 0.0074 | 0.9213 | 0.1300 | 0.0079 | -30.0400 |
| stress | absolute_potential | 10 | 8.4304 | 0.1833 | 0.5918 | 14.6350 | 0.6555 | 123.6600 |
| stress | absolute_potential | 11 | 8.3564 | 0.1841 | 0.5915 | 15.0700 | 0.6615 | 127.9900 |
| stress | absolute_potential | 12 | 8.4161 | 0.1861 | 0.5901 | 15.3050 | 0.6704 | 130.5250 |
| stress | relative_base | 10 | 1.6457 | 0.5998 | 0.0327 | 74.0300 | 0.9865 | 724.2850 |
| stress | relative_base | 11 | 1.6721 | 0.5402 | 0.0253 | 73.4350 | 0.9868 | 718.3650 |
| stress | relative_base | 12 | 1.7024 | 0.5389 | 0.0290 | 73.4800 | 0.9988 | 719.7100 |
| stress | relative_potential | 10 | 1.6374 | 0.5412 | 0.0290 | 73.6900 | 0.9874 | 720.9600 |
| stress | relative_potential | 11 | 1.6285 | 0.5406 | 0.0258 | 73.6100 | 0.9984 | 720.9800 |
| stress | relative_potential | 12 | 1.6004 | 0.5681 | 0.0275 | 73.8300 | 0.9872 | 722.3450 |

### Paired Relative minus Absolute differences

Negative distance/far differences and positive close/hit differences favor Relative. Scenario paths are paired; training seeds are paired. Three seeds do not establish statistical significance.

| Split | Reward | Metric | Seed 10 | Seed 11 | Seed 12 | Mean difference | Sample SD |
|---|---|---|---:|---:|---:|---:|---:|
| final | base | mean_distance | -15.0137 | -15.0622 | -14.8262 | -14.9674 | 0.1246 |
| final | base | close_fraction | 0.6542 | 0.5858 | 0.5842 | 0.6081 | 0.0400 |
| final | base | far_fraction | -0.8884 | -0.8960 | -0.8850 | -0.8898 | 0.0057 |
| final | base | hits_per_episode | 84.4020 | 84.1900 | 84.2040 | 84.2653 | 0.1186 |
| final | base | pooled_shot_accuracy | 0.9777 | 0.9805 | 0.9877 | 0.9820 | 0.0051 |
| final | potential | mean_distance | -6.0296 | -6.0547 | -6.1891 | -6.0911 | 0.0858 |
| final | potential | close_fraction | 0.3332 | 0.3349 | 0.3738 | 0.3473 | 0.0229 |
| final | potential | far_fraction | -0.4944 | -0.5027 | -0.5080 | -0.5017 | 0.0068 |
| final | potential | hits_per_episode | 62.7260 | 62.3380 | 62.7860 | 62.6167 | 0.2432 |
| final | potential | pooled_shot_accuracy | 0.2428 | 0.2453 | 0.2492 | 0.2458 | 0.0032 |
| stress | base | mean_distance | -14.8434 | -14.7672 | -14.8516 | -14.8207 | 0.0465 |
| stress | base | close_fraction | 0.5930 | 0.5328 | 0.5315 | 0.5524 | 0.0351 |
| stress | base | far_fraction | -0.8875 | -0.8947 | -0.8923 | -0.8915 | 0.0037 |
| stress | base | hits_per_episode | 73.9550 | 73.2850 | 73.3500 | 73.5300 | 0.3695 |
| stress | base | pooled_shot_accuracy | 0.9814 | 0.9766 | 0.9909 | 0.9830 | 0.0073 |
| stress | potential | mean_distance | -6.7930 | -6.7279 | -6.8157 | -6.7789 | 0.0456 |
| stress | potential | close_fraction | 0.3579 | 0.3565 | 0.3821 | 0.3655 | 0.0144 |
| stress | potential | far_fraction | -0.5628 | -0.5656 | -0.5626 | -0.5637 | 0.0017 |
| stress | potential | hits_per_episode | 59.0550 | 58.5400 | 58.5250 | 58.7067 | 0.3018 |
| stress | potential | pooled_shot_accuracy | 0.3319 | 0.3368 | 0.3168 | 0.3285 | 0.0104 |

### Independently evaluated references

Each archived model, the unchanged predictive rule, and Random appears once per split. No artificial training-seed standard deviation is attached to these references.

| Split | Reference | Distance | Close fraction | Far fraction | Hits/window | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| final | historical_base | 16.2773 | 0.0271 | 0.8937 | 2.7220 | 0.0733 |
| final | historical_potential | 1.6515 | 0.6817 | 0.0385 | 70.7700 | 0.9058 |
| final | rule | 1.3707 | 0.7304 | 0.0165 | 43.7300 | 1.0000 |
| final | random | 16.9770 | 0.0065 | 0.9233 | 0.0420 | 0.0014 |
| stress | historical_base | 16.4300 | 0.0200 | 0.9007 | 1.9350 | 0.0506 |
| stress | historical_potential | 1.8131 | 0.6119 | 0.0402 | 60.6800 | 0.8979 |
| stress | rule | 1.4683 | 0.6821 | 0.0172 | 42.2200 | 1.0000 |
| stress | random | 16.7844 | 0.0080 | 0.9197 | 0.0400 | 0.0013 |

## Predeclared targets

Overall practical improvement target: **ACHIEVED**.

- distance_reduced_at_least_10_percent: ACHIEVED.
- close_gain_at_least_5_percentage_points: ACHIEVED.
- far_fraction_no_worse: ACHIEVED.
- hits_at_least_90_percent_of_control: ACHIEVED.
- accuracy_at_least_85_percent: ACHIEVED.

Aspirational targets are assessed independently on Relative Potential three-seed final-test means:

- mean_distance_at_most_1_5: ACHIEVED.
- close_fraction_at_least_75_percent: NOT ACHIEVED.
- far_fraction_at_most_5_percent: ACHIEVED.
- mean_capped_acquisition_at_most_15: ACHIEVED.

Worst Relative Potential training seed for each final-test metric:

- mean_distance: seed 10, 1.4578.
- close_fraction: seed 10, 0.5941.
- far_fraction: seed 10, 0.0248.
- hits_per_episode: seed 10, 84.3880.
- pooled_shot_accuracy: seed 10, 0.9928.
- mean_first_acquisition_tick: seed 12, 8.3060.

## Boundary aliasing, target changes, and shooting

Relative state is (xt-xc+24, yt-yc+24, heading, speed-1). It has 19,208 rows versus 3,125,000 absolute rows, and 768,320 versus 125,000,000 float32 element bytes. This approximately 162.70-fold storage reduction is not a speedup or sample-efficiency guarantee. Absolute configurations sharing one relative observation can differ in crosshair clipping and target reflection; deterministic tests demonstrate both. No rule fallback or boundary information was added.

Slices use the pre-action state, with either object within three cells of a wall defining boundary_near. These are policy-dependent diagnostics, not randomized subgroups. Counts below are pooled across training seeds; performance values are means of seed-level ratios.

| Method | Slice | Tick denominator | Mean distance | Close fraction | Far fraction | Zero-visit row fraction |
|---|---|---:|---:|---:|---:|---:|
| absolute_base | boundary_near | 354365 | 17.7442 | 0.0066 | 0.9337 | 0.1381 |
| absolute_base | interior | 95635 | 11.7440 | 0.0156 | 0.8456 | 0.1789 |
| absolute_base | target_near | 227679 | 18.3161 | 0.0090 | 0.9205 | 0.1524 |
| absolute_base | crosshair_near | 256984 | 18.1542 | 0.0081 | 0.9281 | 0.1160 |
| absolute_base | speed_1 | 226758 | 16.4413 | 0.0087 | 0.9153 | 0.1452 |
| absolute_base | speed_2 | 223242 | 16.4971 | 0.0082 | 0.9148 | 0.1484 |
| absolute_potential | boundary_near | 264968 | 9.0139 | 0.2123 | 0.5980 | 0.1462 |
| absolute_potential | interior | 185032 | 5.4086 | 0.3237 | 0.4208 | 0.0752 |
| absolute_potential | target_near | 227679 | 8.6887 | 0.2329 | 0.5686 | 0.1349 |
| absolute_potential | crosshair_near | 155932 | 7.1470 | 0.3298 | 0.4406 | 0.1357 |
| absolute_potential | speed_1 | 226758 | 6.0816 | 0.3576 | 0.4371 | 0.0956 |
| absolute_potential | speed_2 | 223242 | 9.0042 | 0.1570 | 0.6145 | 0.1388 |
| relative_base | boundary_near | 251868 | 1.6429 | 0.6181 | 0.0334 | 0.0000 |
| relative_base | interior | 198132 | 1.3221 | 0.6146 | 0.0147 | 0.0000 |
| relative_base | target_near | 227679 | 1.5971 | 0.6186 | 0.0294 | 0.0001 |
| relative_base | crosshair_near | 237075 | 1.4631 | 0.6287 | 0.0200 | 0.0001 |
| relative_base | speed_1 | 226758 | 1.3578 | 0.6516 | 0.0193 | 0.0000 |
| relative_base | speed_2 | 223242 | 1.6477 | 0.5809 | 0.0312 | 0.0001 |
| relative_potential | boundary_near | 251511 | 1.5314 | 0.6077 | 0.0297 | 0.0001 |
| relative_potential | interior | 198489 | 1.3249 | 0.6025 | 0.0153 | 0.0000 |
| relative_potential | target_near | 227679 | 1.4986 | 0.6083 | 0.0263 | 0.0001 |
| relative_potential | crosshair_near | 236379 | 1.3628 | 0.6178 | 0.0164 | 0.0001 |
| relative_potential | speed_1 | 226758 | 1.3029 | 0.6295 | 0.0163 | 0.0000 |
| relative_potential | speed_2 | 223242 | 1.5799 | 0.5809 | 0.0306 | 0.0001 |

absolute_potential: observed recovery fraction 0.6428; mean capped recovery lag 10.8284 ticks. Recovery windows around turns, speed toggles, and reflections may overlap. Events with insufficient 20-tick follow-up are censored; unrecovered events have capped lag 20 and are not successful 20-tick recoveries.

relative_potential: observed recovery fraction 0.9999; mean capped recovery lag 1.2568 ticks. Recovery windows around turns, speed toggles, and reflections may overlap. Events with insufficient 20-tick follow-up are censored; unrecovered events have capped lag 20 and are not successful 20-tick recoveries.

Firing consumes a tick while the target moves; it cannot be combined with a crosshair move. Compare hits and accuracy with proximity rather than assuming they measure the same skill. Raw action frequencies, slow/fast proportions by pre-action distance, and unaligned-shot counts are available for every policy and split. Stress testing changes only turn and speed-toggle probabilities from 0.05 to 0.10; it tests this defined shift, not arbitrary real-world generalization.

## Resource use, execution, and verification

Formal run times sum to 1325.75 seconds, including checkpoint validation, coverage scans, logging, model compression, and reload checks. Peak observed process working set was 1091.52 MiB. It is a Windows process-lifetime high-water mark, not a per-run memory delta. Free disk at preflight was 588.97 GiB.

| Method | Seed | Interactions | Measured seconds | Visited rows / denominator | Visited pairs / denominator |
|---|---:|---:|---:|---:|---:|
| absolute_base | 10 | 5000000 | 128.71 | 1669878 / 3125000 | 4744950 / 31250000 |
| absolute_base | 11 | 5000000 | 123.39 | 1672395 / 3125000 | 4747878 / 31250000 |
| absolute_base | 12 | 5000000 | 103.99 | 1674589 / 3125000 | 4748308 / 31250000 |
| absolute_potential | 10 | 5000000 | 118.43 | 1764227 / 3125000 | 3376353 / 31250000 |
| absolute_potential | 11 | 5000000 | 115.78 | 1770388 / 3125000 | 3383804 / 31250000 |
| absolute_potential | 12 | 5000000 | 115.86 | 1772003 / 3125000 | 3384893 / 31250000 |
| relative_base | 10 | 5000000 | 102.72 | 18789 / 19208 | 174567 / 192080 |
| relative_base | 11 | 5000000 | 107.91 | 18795 / 19208 | 174061 / 192080 |
| relative_base | 12 | 5000000 | 99.74 | 18805 / 19208 | 174373 / 192080 |
| relative_potential | 10 | 5000000 | 103.26 | 18571 / 19208 | 149918 / 192080 |
| relative_potential | 11 | 5000000 | 99.73 | 18568 / 19208 | 149312 / 192080 |
| relative_potential | 12 | 5000000 | 106.24 | 18604 / 19208 | 149726 / 192080 |

Audit: 132 development summaries, 26,400 development windows, 7,920,000 development ticks; 8,000 final windows and 3,200 stress windows. Each formal run has 16,666 complete 300-tick windows and one 200-tick budget-stop partial window. All update-count sums equal actual interactions. Original histories, models, source, configuration and protocols retain their original hashes; README is the sole intentionally edited pre-existing file.

Setup and the complete existing/new unittest suite were executed. Exact commands, exit codes, output logs, peak memory, stage times, protocol snapshots, artifact digests, and target-pairing checks are retained in execution_checks.json and stage_execution/. A pre-formal new-test failure caused by Windows sandbox temporary-directory permissions was fixed by using the workspace temporary directory; no formal experiments had started. Superseded preflights are retained if present.

## Figures and full-length animations

- [Learning curves](render/learning_curves.png)
- [Independent final comparisons](render/final_comparison.png)
- [Distance distributions and far tails](render/distance_tails.png)
- [Boundary and speed diagnostics](render/boundary_speed_diagnostics.png)
- [Genuine update coverage](render/coverage.png)
- [Slow/fast choices](render/speed_choices.png)

- [Full main comparison, scenario 3601000](render/seed3601000_main_comparison.gif)
- [Full main comparison, scenario 3601001](render/seed3601001_main_comparison.gif)
- [Full main comparison, scenario 3601002](render/seed3601002_main_comparison.gif)
- [Six-policy comparison with negative control](render/seed3601000_six_policy_comparison.gif)

Each GIF contains all 301 frames and decodes to 20.07 seconds with actual 60/70 ms frame delays. New displayed agents always use training seed 10, not a selected best seed. Historical Potential is clearly labeled 20M, seed 0; new agents are labeled 5M. Trace CSVs and decoded start/middle/end PNGs accompany the animations. Actual-displacement arrows are drawn only for observed crosshair movement; hit stars show pre-motion shot coordinates separately from post-motion target circles. Export and decoding are verified separately from GUI playback, which remains unverified.

## Reproduction

[Commands and controls](../../docs/tracking_improvement.md). Models and genuine update counters are compressed NPZ files in runs/<method>/seed_<seed>/. Completed runs are reused only after configuration, budget, source, model and record integrity checks. Incomplete attempts are retained and rejected for silent reuse. Evaluation is epsilon-zero with separate environment/action streams and before/after Q/counter/RNG snapshots.

## Answers and remaining limitations

**Did representation sharing help Base?** Yes in mean final-test distance: Absolute Base 16.4690, Relative Base 1.5016. The complete close/far, shooting, paired differences, and individual seeds above qualify this answer.

**Did it improve sustained tracking beyond matched Potential?** Mean distance improved: matched Absolute Potential 7.5314, Relative Potential 1.4403. The predeclared multi-metric practical target was achieved. This answer uses matched 5M controls, not the 20M historical reference.

**What happens near walls and after target changes?** The boundary/speed slices and overlapping recovery measurements above quantify the behavior. State aliasing at walls is established mechanically, but these policy-dependent observational slices do not isolate its causal contribution. Any boundary-aware representation must be a separately frozen follow-up, not an expansion tuned against this final set.

**Was there a shooting tradeoff?** Relative Potential obtained 84.4893 hits/window at mean seed-level pooled accuracy 0.9949, versus matched control 21.8727 and 0.7491. Firing necessarily forgoes movement for that tick; compare these measured outcomes with the proximity and action-frequency records rather than claiming an isolated causal tradeoff.

**Which measured limitation should the next study address?** The candidate missed at least one practical or absolute target. Use its reported far tails, worst seed, wall slices, and shooting/acquisition deficits to design a separately preregistered boundary-aware or reward/coverage study; do not extend this budget.

The study establishes neither real-game performance nor tabular optimality of the lossy relative controller. Passing software checks, shaped training reward, or one attractive clip is not evidence that tracking is solved.
