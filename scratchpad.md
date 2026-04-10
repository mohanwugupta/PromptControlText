# Scratchpad

## Progress Update
*   **Phase 1**: Completed Execution pipeline over XSTest & HarmBench. Includes model caches, download logic, offline parsing, rules-based heuristic generation, tests, and plots.
*   **Phase 2**: Scaffolded IHEval evaluations. `download_data.py` pulls `google/iheval` caching it offline. Added `hierarchy_scorer.py` mapping instruction conflicts into `hierarchy_following`, `stop_compliance`, and `unsafe_continuation`.
*   **Phase 2 Plots**: Scaled `plots.py` with argument `--phase 2` which outputs Hierarchy Conflict grouped bar charts analyzing `Accuracy vs Unsafe Continuation` across vague vs explicit prompt variants.
*   **Version Control**: `.gitignore` populated with standard Python/OS filters.
