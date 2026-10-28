# Tracking representation study

This bounded study compares absolute and relative Q tables at a matched five-million-interaction budget. Relative observations deliberately discard wall information and are not an exact Markov state. Historical 20-million-interaction models are reference comparisons only. All new learned results include training seeds 10, 11, and 12; displayed models always use seed 10.

Run from the original project root in Windows PowerShell, using the existing environment:

```powershell
Set-Location "C:\Users\zhaod\OneDrive\Documents\ChatGPT\6885 delta force 2"
.\.venv\Scripts\python.exe scripts\check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage check
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage diagnose
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage pilot
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage train
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage evaluate
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage render
.\.venv\Scripts\python.exe scripts\plot_tracking_final.py
.\.venv\Scripts\python.exe scripts\prepare_tracking_audit.py
.\.venv\Scripts\python.exe -m aimrl.tracking_research --config configs/tracking_research.json --stage audit
Start-Process ".\results\tracking_improvement\render\seed3601000_main_comparison.gif"
```

The four disposable pilots use seed 99 and exactly 100,000 interactions each, solely for instrumentation and execution checks. The formal matrix has 12 zero-initialized runs and exactly 60 million interactions. No archived or pilot values initialize a formal model. The formal protocol records source/configuration bytes before formal training; later edits cannot retroactively change completed-run provenance.

The development split contains 200 scenarios per checkpoint. Final testing contains 500 scenarios per policy instance and stress testing 200; final/stress data are used only after all formal runs complete. Stress changes only turn and speed-toggle probabilities to 0.10. Each untrained baseline and historical model is evaluated once per split. Target trajectory hashes include the initial target observation and every tick. Evaluation never updates Q values or genuine visitation counters and uses separate environment/action streams.

Final models and counters are lossless NPZ packages. Models include explicit representation/indexing versions and reject incompatible loads. A compressed model is loaded with `ResearchAgent(config, representation).load_package(path, with_counts=True)`. Counters increment exactly once per training update. Their shape denominator differs across representations; relative coverage merges many physical configurations and is not directly equivalent to absolute physical-state coverage.

Completed stages and runs are reused only after configuration, budget, source and artifact checks. Partial attempts are retained and rejected for silent resume. Use a fresh descriptive output session if necessary; do not overwrite existing evidence. Formal-source edits require a new study, not a widened historical hash acceptance rule.

Read the [measured report](../results/tracking_improvement/results_summary.md), [frozen protocol](../results/tracking_improvement/protocol.json), and [execution audit](../results/tracking_improvement/execution_checks.json). Figure and GIF paths are in the report. Exported GIFs have 301 frames, real 60/70 ms delays, and 20.07 seconds total duration. GUI playback is reported separately from complete image decoding.

The reward transform is `r_base + gamma * Phi(next_state) - Phi(state)`, with `Phi = -L1_distance`. It is applied once before the continuing Q update. Hits, external 300-tick truncations, and partial interaction-budget stops all bootstrap from the actual next observation, never a reset observation. The potential-based form follows [Ng, Harada, and Russell (1999), Policy invariance under reward transformations: Theory and application to reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). Its theoretical assumptions do not supply an optimality or finite-budget performance guarantee for this lossy relative controller. The termination/truncation distinction is also described in [Farama's handling-time-limits documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/); Gymnasium is not a dependency of this project.

The standalone final-figure script was added during post-training visual review to make the 5M versus historical 20M budgets explicit. It reads existing evaluation records only. Its source/data/figure hashes are recorded in render/figure_label_review.json. Frozen training sources, models, numerical results, and the formal protocol were not edited to accommodate this later presentation change.

The first audit reached link checking and rejected its own not-yet-written execution_checks.json link. The small preparation script creates an explicitly incomplete marker so the unchanged frozen audit can validate the link before writing its verified final record. It never marks checks as passed. The failed audit attempt is retained in stage_execution/; the subsequent full audit repeats all checks. This post-training bookkeeping correction changes no experiment or training provenance.
