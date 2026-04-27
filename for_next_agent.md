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

1. Steps 1-5 done: download → join/filter → 80/20 grouped+stratified split → 5 preprocessings → PLSR with repeated 10-fold CV + one-SE rule + test eval.
2. Step 6 not started: replace PLSR with a DL model (1D CNN baseline first), same splits, same per-cell JSON + predictions CSV layout, write to `results/per_cell_dl/`.

## Important truths about the paper replication (don't relearn the hard way)

1. **Sample counts** look wrong at first glance but are right. Paper §2.1 prose says China 245 / Kenya 239 / Indonesia 226; Paper Table 1 says 262 / 245 / 236. Both numbers are derivable from the same public data — prose counts are post-deduplication on `Batch and labid`; Table 1 counts keep duplicate rows. We keep duplicates so n matches Table 1 exactly (3997 / 262 / 245 / 236). Full reconciliation in `experiment_spec.md` "Sample-count reconciliation".
2. **Two kinds of duplicate rows** in the public data, both kept: 82 reference-table rows that differ only in coordinates (likely DB edit history) and 155 spectra-table rows that are genuine repeat scans of the same physical sample. The group key `Batch and labid` keeps a physical sample's rows always together in any split.
3. **Exact paper numbers don't replicate.** Our PLSR gets 0/20 cells passing all four acceptance criteria. The paper does not state its random seed or its exact one-SE rule SE-base, and these two unspecified items account for almost all of the divergence. The paper's *qualitative* claim (preprocessing varies by region) survives in our run with shuffled specifics. This is documented in `our_task_log.md` Step 5 and is acceptable for the project goal (DL vs PLSR head-to-head on identical splits).
4. **Project-wide constants:** seed = 42, group key = `Batch and labid`, spectral window = 4000-600 cm⁻¹ (1763 wavenumbers), SOC quartile stratification, 5×10 = 50 CV folds.
5. **Preprocessing artefact CSVs** (`data/preprocessed/`) are illustrative — they use SG/SGD window=11 polyorder=2 as exemplar values. The actual modelling code re-applies preprocessing inside the CV loop with proper grid search (SG/SGD: window 7-31 odd × polyorder {2,3}); MSC reference is re-fit per inner fold. Do not consume preprocessed CSVs directly inside the modelling loop for sg/sgd/msc — re-apply from raw.

## Operational gotchas

1. **Auto-commit hook is active.** Some tool periodically auto-commits the working tree as commit message "done". Result CSVs and model outputs got dragged into git history before we gitignored `results/`. Be defensive: gitignore new heavy folders *before* writing into them.
2. **PowerShell on Windows** is the default shell. `cat <<EOF` heredocs do not work — use a temp file (`git commit -F file.txt`) when writing multiline commit messages.
3. **Stdout is buffered** when running long Python scripts via Shell. Watch progress by tailing the per-cell JSON output folder, not the terminal text.
4. **scipy + sklearn + scipy savgol_filter + pandas + openpyxl** are pinned in `requirements.txt`. Add `torch` (or whatever DL framework) when starting Step 6.

## Parked items (flag-for-later, not blocking)

1. Email Tong Li to ask whether the 82 reference-table coordinate-edit duplicates were intentional in their analysis.
2. Decide whether to drop those 82 duplicates as a methodological cleanup in our DL paper.
3. PLSR LV-selection divergence — could be re-attempted with SE = std/√(K_inner_folds=10) instead of √50 to see if our LV picks converge to paper. Not required for DL comparison.

## Step 6 — what to do next

1. Build `train_dl.py` mirroring `train_plsr.py` structure: same per-cell loop, same JSON + predictions CSV schema, write to `results/per_cell_dl/`.
2. Architecture: small 1D CNN — Conv1d → BN → ReLU → MaxPool ×3, then 2 dense layers, ~50k params. Adam, MSE, cosine-anneal LR, early stopping on a single fixed 20% slice of train (no inner CV — too expensive).
3. Same 5 preprocessings as PLSR (None / SNV / MSC / SG / SGD). For SG/SGD use the paper's grid as a hyperparameter (or fix a sensible default and tune separately later — confirm with Arif before committing).
4. Companion `summarise_dl_vs_plsr.py`: side-by-side table DL-ours vs PLSR-ours vs PLSR-paper. Add the paired Wilcoxon + Holm tests the paper used.
5. There is already a `train_bn_ae_global.py` in the repo from a prior session — not part of the current pipeline. Inspect it briefly before starting; it may have reusable PyTorch boilerplate, or it may be discardable.

## What Arif wants from you (style reminders not in instructions.md)

1. He has said multiple times "less text, my eyes hurt." Default to terse. Numbered lists, no decorative tables, no preamble. Use the AskQuestion tool sparingly; pick the best decision and log it instead.
2. When committing, do not paraphrase the diff in the commit body — describe intent in 1-2 lines + a short bullet list. Use a temp file for the message (PowerShell limitation above).
3. Update `our_task_log.md` after every meaningful step. The recipe at the top must stay copy-pasteable.
4. Decisions taken without asking get logged with a one-liner reason in `our_task_log.md` "Decisions on ambiguities".
