# Indonesia investigation — baseline vs PBN at reduced epochs

Targeted hand-off for a side-experiment. Read this top-to-bottom, then read the files in §1.

## 1. Read these for context (don't duplicate, just read)

1. `.cursor/project.md` — what this project is.
2. `.cursor/instructions.md` — Arif's code style and working rules. Follow strictly.
3. `our_task_log.md` — full pipeline. **Read Step 7 carefully** — that's the PBN experiment whose Indonesia behaviour we're now poking at. The 5-method headline win-rates and per-cell numbers are there.
4. `for_next_agent.md` — operational gotchas (Windows + PowerShell, stdout buffering, venv location, auto-commit hook, gitignore-before-write).
5. `experiment_spec.md` — paper spec, only relevant if you need to remember how the splits were made.

## 2. Scope of this side-experiment

1. Dataset: **indonesia** only.
2. Preprocessing: **none** only.
3. Methods: **`baseline` and `pbn` only** (no plsr_pbn / rbn / r2bn — they're already done at 200 epochs).
4. Supervised training epochs: **50** (down from the project default of 200). Phase A (PBN's autoencoder pretrain) stays at 100 epochs unless Arif says otherwise.
5. First pass: the single seed=42 run, same as the rest of the experiment, so it's directly comparable.

## 3. Why this experiment exists

At 200 epochs on indonesia/none we got `baseline` RMSE 1.2521, `pbn` RMSE 1.5539 — i.e. PBN is **worse** on this cell, the only Indonesia cell where `baseline` wins. Hypothesis: with n_train=188 the MLP head overfits well before 200 epochs, and PBN — which has more trainable params (BN γ/β + running stats baked in) — overfits faster. Cutting to 50 epochs may flip the verdict.

This is exploratory. "First 50" implies more cuts will follow (25? 100? early stopping on a held-out slice?). Don't pre-build a giant grid; deliver the 50-epoch result first, then wait for direction.

## 4. Data facts you need to remember while interpreting results

1. n_train = 188, n_test = 48. Smallest of the four datasets.
2. SOC range is wide (includes peat outliers ~30%). One peat sample landing on the wrong side of the split shifts test RMSE by O(0.1).
3. With n_test = 48 a single mispredicted sample moves test RMSE by roughly `(observed - predicted)² / 48` — i.e. one peat sample with a 5%-SOC residual contributes ~0.5 to MSE = ~0.7 to RMSE before averaging effects. Be skeptical of any single-seed delta smaller than ~0.1 RMSE.
4. Suggest doing a multi-seed sweep (e.g. seeds 42, 0, 1, 2, 3) before drawing any conclusion stronger than "trend looks like X". Confirm with Arif before adding seeds — that's a methodology change.

## 5. Existing artefacts you must NOT clobber

1. `results/pbn_experiment/cells/indonesia_none_{baseline,pbn}.json` — the 200-epoch numbers. They are the reference comparison.
2. `results/pbn_experiment/cell_results.csv` — aggregate of all 120 existing cells.
3. The runner `train_pbn_experiment.py` is idempotent and skips cells whose JSON already exists. If you want to add new cells, either change the output path or change the method-name slug (e.g. `pbn_50ep`) so the JSON sits beside, not on top of, the 200-epoch one.

## 6. Code you should reuse, not rewrite

1. `model_baseline_ann.py` — `BaselineSocAnn`. Use as-is.
2. `model_pbn_ann.py` — `LearnedPreprocessingAutoencoder` + `PbnSocAnn`. Use as-is.
3. `train_pbn_experiment.py` — has all the helpers you need: `load_one_preprocessed_pair`, `extract_spectra_and_target`, `pretrain_autoencoder`, `train_supervised_regressor(..., n_epochs)` (already takes epoch count as a parameter), `compute_metrics_dictionary`, `cell_output_paths`, `save_cell_predictions`, etc.

The cleanest path is a small standalone script (e.g. `indonesia_short_epochs_experiment.py`) that imports those helpers, hardcodes `dataset_name="indonesia"`, `preprocessing_name="none"`, `n_epochs=50`, runs both methods, and writes JSONs + predictions to a fresh subdir like `results/pbn_experiment/short_epochs/`. Don't add the new method slugs to the main runner unless Arif asks — keep the side-experiment self-contained.

Match the project's training contract: same Adam, same MSE, same batch=64, same lr=1e-3, same seed=42 (all imported from `train_pbn_experiment.py`).

## 7. Operational

1. Python: `C:\Users\m.rahman\vens\tillage\Scripts\python.exe`. Has torch (CPU build) and sklearn. Plain `python` on PATH is the system Python and lacks torch — don't use it.
2. Indonesia cells run in ~1–2 seconds each on CPU at 200 epochs. At 50 epochs both methods together should finish in well under a minute. No need to background the process; you can just run it foreground and watch the print.
3. Comparison report: print baseline-50ep, pbn-50ep, and the existing 200-epoch numbers for the same cell side-by-side, with delta-RMSE. A 10-line print is enough — don't build a big formatter for two cells.

## 8. What to deliver back to Arif

1. The 50-epoch RMSE/R²/MBD/RPIQ for both methods on indonesia/none, plus the 200-epoch baseline for context, plus delta-RMSE.
2. A one-paragraph read of whether reducing epochs closes the PBN-vs-baseline gap, widens it, or flips it.
3. A suggested next step (e.g. "try 25 epochs", "add multi-seed", "add early stopping on 20% holdout"). Don't act on it without confirmation.
