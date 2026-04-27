# For the next agent

Read `.cursor/project.md` and `.cursor/instructions.md` first — they cover the goal and Arif's working style. This file fills the gaps.

## Where things live

1. `experiment_spec.md` — the paper's full experimental design and every ambiguity we identified, with the resolution (or "deferred") next to each. Always check here before assuming anything about the paper.
2. `our_task_log.md` — running record of every step taken. Has the full reproduction recipe at the top (commands to run from scratch) and a "Decisions on ambiguities" list. Read it end-to-end on day one.
3. `resource/tong.pdf` and `resource/supplementary.xlsx` — the paper itself and Table S1 (gitignored, must be placed manually).
4. `additionals/` — gitignored, raw downloaded ICRAF/ISRIC data.
5. `data/` — gitignored, all derived CSVs (raw per-domain, splits, preprocessed train/test).
6. `results/` — gitignored, all model outputs (per-cell JSONs, predictions CSVs, comparison tables).

## Status

1. Steps 1-5 done: download → join/filter → 80/20 grouped+stratified split → 5 preprocessings (now 6 — see point 4) → PLSR with repeated 10-fold CV + one-SE rule + test eval.
2. **Step 5b done (27 Apr 2026)** — diagnostic ablation: re-fit PLSR with the paper's reported LV count (LV one-SE picker bypassed), keeping our SG/SGD window/polyorder. Goal was to isolate whether the LV-selection rule or the unknown split seed drives our R² gap from paper Table 1. **Result: forcing paper LVs made R² *worse* in 18/20 cells, not better.** Conclusion: the split seed (unrecoverable from paper) dominates the divergence; the one-SE rule definition is secondary. Our higher LVs were *compensating* for a harder test draw. Artefacts: `train_plsr_fixed_lv.py`, `results/per_cell_fixed_lv/`, `results/table1_replication_fixed_lv.csv`. **All publishable-spec stones are now turned**: data, n, window, algorithm, mean-centring, CV protocol, stratification, grouping, metrics, peat handling, duplicate handling, LV count — all match. Two unobservables remain: split seed (paper says "fixed seed" without value) and SE definition for the one-SE rule. Decision: accept current single-split + seed=42 PLSR results as our reproduction baseline; the qualitative claim (regional preprocessing dependence) survives. **Do not chase exact paper numerics further** unless an author email yields the seed.
3. **PLSR-minmax added (27 Apr 2026)** — Arif asked for the 6th preprocessing `minmax` to be run through PLSR too; `train_plsr.py` / `summarise_results.py` / `print_plsr_tables.py` now include minmax. `results/table1_replication.csv` has 24 rows (paper columns NaN for the 4 minmax rows since paper Table 1 has no minmax). MinMax math is duplicated in `train_plsr.py::apply_minmax_per_feature` so it can be re-fit per inner CV fold (no leakage).
4. **Step 6 / Step 7 — DL replacement (current state)**. The DL replacement IS the BN-front-end matrix, **2 methods only** after deliberate trim (28 Apr 2026):
   - `baseline` — MLP only (`Linear(1763 → 32) → ReLU → Linear(32 → 1)`), 200 sup epochs.
   - `rbn` — same MLP head with a learnable `BatchNorm1d(1763)` in front, 200 sup epochs. **This is the main DL algorithm.**
   We previously also ran `pbn`, `plsr_pbn`, `r2bn`, `p2bn` ablations (full 144-cell matrix). All four were dropped because they didn't materially shift the story (RBN ≥ PBN, others marginal or harmful). The full 6-method code + cell JSONs are preserved at git tag `before_drops_1` if a reviewer asks for any specific ablation. Resurrect by `git checkout before_drops_1 -- <file>`.
   Matrix: 4 datasets × 6 preprocessings × 2 methods = **48 cells**. Models in `model_baseline_ann.py` and `model_rbn_ann.py`. Runner `train_pbn_experiment.py`. Outputs in `results/pbn_experiment/cells/`, `…/predictions/`, `…/cell_results.csv` (folder name historical). Headline win-rate (rbn vs baseline on test RMSE): **18/24 cells**, weak only on Indonesia (1/6 — overfitting on 188-row train). CPU runtime: ~7 min from scratch for all 48 cells.

## Important truths about the paper replication (don't relearn the hard way)

1. **Sample counts** look wrong at first glance but are right. Paper §2.1 prose says China 245 / Kenya 239 / Indonesia 226; Paper Table 1 says 262 / 245 / 236. Both numbers are derivable from the same public data — prose counts are post-deduplication on `Batch and labid`; Table 1 counts keep duplicate rows. We keep duplicates so n matches Table 1 exactly (3997 / 262 / 245 / 236). Full reconciliation in `experiment_spec.md` "Sample-count reconciliation".
2. **Two kinds of duplicate rows** in the public data, both kept: 82 reference-table rows that differ only in coordinates (likely DB edit history) and 155 spectra-table rows that are genuine repeat scans of the same physical sample. The group key `Batch and labid` keeps a physical sample's rows always together in any split.
3. **Exact paper numbers don't replicate, and Step 5b proved why.** Our PLSR gets 0/20 cells passing all four acceptance criteria. Re-running with the paper's exact LV counts (Step 5b) made R² *worse* in 18/20 cells, which means: (a) the LV one-SE rule is *not* the dominant divergence cause; (b) the unrecoverable split seed is. Our higher LVs were *compensating* for a harder test draw — paper LVs underfit our split. The paper's *qualitative* claim (preprocessing varies by region) survives in our run with shuffled specifics. Decision is logged: accept current PLSR results as-is, move on to DL. Do not re-litigate this unless someone obtains the paper's split seed (parked: email Tong Li).
4. **Project-wide constants:** seed = 42, group key = `Batch and labid`, spectral window = 4000-600 cm⁻¹ (1763 wavenumbers), SOC quartile stratification, 5×10 = 50 CV folds.
5. **Preprocessing artefact CSVs** (`data/preprocessed/`) are illustrative — they use SG/SGD window=11 polyorder=2 as exemplar values. The actual modelling code re-applies preprocessing inside the CV loop with proper grid search (SG/SGD: window 7-31 odd × polyorder {2,3}); MSC reference is re-fit per inner fold. Do not consume preprocessed CSVs directly inside the modelling loop for sg/sgd/msc — re-apply from raw.

## Operational gotchas

1. **Auto-commit hook is active.** Some tool periodically auto-commits the working tree as commit message "done". Result CSVs and model outputs got dragged into git history before we gitignored `results/`. Be defensive: gitignore new heavy folders *before* writing into them.
2. **PowerShell on Windows** is the default shell. `cat <<EOF` heredocs do not work — use a temp file (`git commit -F file.txt`) when writing multiline commit messages.
3. **Stdout is buffered** when running long Python scripts via Shell. Watch progress by tailing the per-cell JSON output folder, not the terminal text.
4. **scipy + sklearn + scipy savgol_filter + pandas + openpyxl** are pinned in `requirements.txt`. Add `torch` (or whatever DL framework) when starting Step 6.

## Parked items (flag-for-later, not blocking)

1. Email Tong Li to ask (a) whether the 82 reference-table coordinate-edit duplicates were intentional in their analysis, and (b) the random seed used for the 80/20 split. Item (b) is the only thing that could close the R² gap; without it, exact reproduction is unreachable.
2. Decide whether to drop those 82 duplicates as a methodological cleanup in our DL paper.
3. ~~PLSR LV-selection divergence — re-attempt with SE = std/√10 instead of √50.~~ **Resolved 27 Apr 2026**: Step 5b showed LV is not the dominant cause anyway. Don't bother.
4. Try an alternative one-SE-rule SE definition in a future audit if a reviewer asks. Cheap (~30 min compute) but unlikely to change the story given Step 5b result.
5. **Multi-seed sensitivity band** (5 different outer-split seeds) to characterise the seed-induced uncertainty in our R²/RMSE — would let us state "paper's number sits at the Xth percentile of our distribution". Defensible reproduction add-on. Cost: ~3 hours compute. *Not done because we deemed it not worth the time vs. moving on to DL — flag for paper writeup if a reviewer pushes back.*

## What to do next

PLSR replication is **done as far as the paper permits** (see Status point 2). The DL replacement (RBN vs baseline) is also **done at first pass** (see Status point 4). Drive Arif from here.

1. Possible next directions (await Arif's call):
   - Companion `summarise_rbn_vs_plsr.py`: side-by-side table RBN-ours vs PLSR-ours vs PLSR-paper, with paired Wilcoxon + Holm tests the paper used.
   - Indonesia overfitting probe (reduced epochs / early stopping / multi-seed). Brief in `indonesia.md` was written for `pbn` and is now stale; rewritten for `rbn`.
   - 1D CNN replacement of the MLP head (Conv1d → BN → ReLU → MaxPool ×3, then 2 dense layers, ~50k params; Adam, MSE, cosine-anneal LR).
   - Multi-seed sensitivity band on RBN (5 seeds, both methods × 48 cells = 480 runs ≈ 1 hour CPU).
2. The trimmed DL experiment lives in `train_pbn_experiment.py` + `model_baseline_ann.py` + `model_rbn_ann.py`. The dropped ablation models (`model_pbn_ann.py`, `model_p2bn_ann.py`) were deleted from `master` but still exist at tag `before_drops_1`. Indonesia is the known outlier — both `baseline` and `rbn` overfit its 188-row train set (RBN wins only 1/6 preprocessings there). Don't draw conclusions about RBN's overall efficacy from Indonesia in isolation.

## Inventory of artefacts created during PLSR-replication phase (incl. Step 5b)

PLSR pipeline:
1. `train_plsr.py` — main pipeline. CV + one-SE rule + refit + test eval. Writes `results/per_cell/{dataset}_{method}.json` + predictions CSV. Skip-if-exists idempotent.
2. `summarise_results.py` — builds `results/table1_replication.csv` with paper-vs-ours comparison and per-criterion PASS/FAIL flags.
3. `train_plsr_fixed_lv.py` — Step 5b diagnostic. Bypasses the one-SE rule and forces paper's exact LV count per (dataset, method) cell, keeping our chosen SG/SGD window/polyorder. Writes `results/per_cell_fixed_lv/` and `results/table1_replication_fixed_lv.csv`.

Spectra visualisation (added 27 Apr 2026):
1. `plot_preprocessed_spectra.py` — 6×4 grid (rows = methods, cols = regions). Mean spectrum per cell, y-axis free per row. Helps see what each preprocessing does at the dataset level.
2. `plot_one_sample_spectra.py` — 6×4 grid, single random train sample per region traced through all 6 preprocessing methods.
3. `plot_three_samples_spectra.py` — same grid, but with 3 random samples per region (red/blue/green; colours consistent across the 6 method panels of each region's column).

All three plot scripts use seed 42 for sample selection so re-runs are deterministic. Outputs go to `results/{*}.png`.

## What Arif wants from you (style reminders not in instructions.md)

1. He has said multiple times "less text, my eyes hurt." Default to terse. Numbered lists, no decorative tables, no preamble. Use the AskQuestion tool sparingly; pick the best decision and log it instead.
2. When committing, do not paraphrase the diff in the commit body — describe intent in 1-2 lines + a short bullet list. Use a temp file for the message (PowerShell limitation above).
3. Update `our_task_log.md` after every meaningful step. The recipe at the top must stay copy-pasteable.
4. Decisions taken without asking get logged with a one-liner reason in `our_task_log.md` "Decisions on ambiguities".
