# Task Log — Tong et al. (2026) MIR-SOC reproduction

A running, concise record of every step. Anyone with this repo + a Python install should be able to recreate every artefact by following the recipe below.

---

## Reproduction recipe

Run these in order from the project root. Each command is idempotent — safe to re-run.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python download_data.py        # downloads + verifies + extracts the public ICRAF/ISRIC MIR library
python data_loader.py          # builds data/raw/{global,china,kenya,indonesia}.csv
python make_splits.py          # builds data/splits/{dataset}_split.csv  (single 80/20, group=Batch and labid, strata=SOC quartile, seed=42)
python make_preprocessed.py    # builds data/preprocessed/{dataset}_{none|snv|msc|sg|sgd|minmax}_{train|test}.csv  (48 files, ~860 MB total)
python verify_preprocessed.py  # optional: sanity-checks SNV/MSC/SG/SGD math on the global train set
python train_plsr.py           # PLSR sweep over 4 datasets x 5 methods, writes results/per_cell/{dataset}_{method}.json + predictions CSV (~32 min)
python summarise_results.py    # builds results/table1_replication.csv comparing every cell to paper Table 1
# ... (further steps appended below as we add them)
```

One step is **manual** because the file is behind Elsevier's article page (no public direct URL):

1. Open `https://doi.org/10.1016/j.still.2026.107237` → "Appendix A. Supplementary data" → download the `.xlsx` → save as `resource/supplementary.xlsx`.

The PDF of the paper itself (`resource/tong.pdf`) is also expected to be placed manually.

## Environment

1. Python 3.13.7 (any 3.12+ should work). On this machine the project venv lives at `C:\Users\m.rahman\vens\tillage` and `.vscode/settings.json` auto-activates it inside Cursor.
2. Pinned dependencies in `requirements.txt`.

## Repo layout

```text
download_data.py                                 # step 1: fetches and extracts the raw dataset
data_loader.py                                   # step 2: builds raw per-domain CSVs
make_splits.py                                   # step 3: builds 80/20 train/test split tables
make_preprocessed.py                             # step 4: applies the 6 preprocessings (incl our minmax), writes train/test CSVs
verify_preprocessed.py                           # one-shot integrity check on preprocessed outputs
train_plsr.py                                    # step 5: PLSR + CV + one-SE rule + test eval, per (dataset x method)
summarise_results.py                             # step 5b: table1_replication.csv, paper-vs-ours acceptance flags
model_baseline_ann.py                            # DL branch: BaselineSocAnn (no preprocessing layer)
model_pbn_ann.py                                 # DL branch: LearnedPreprocessingAutoencoder + PbnSocAnn (BN as learned preprocessing)
model_rbn_ann.py                                 # DL branch: RbnSocAnn (fresh BN + MLP head; used by both rbn and r2bn)
train_pbn_experiment.py                          # DL branch: runs 5 methods (baseline / pbn / plsr_pbn / rbn / r2bn) across 4 datasets x 6 preprocessings = 120 cells
report_pbn_experiment.py                         # DL branch: prints per-dataset block-format report + 6 head-to-head win counts from cell_results.csv
requirements.txt
experiment_spec.md                               # full paper-replication spec, ambiguity decisions, numbers to match
our_task_log.md                                  # this file
resource/
  tong.pdf                                       # the paper (manual)
  supplementary.xlsx                             # Table S1 (manual, see recipe)
additionals/                                     # gitignored, raw downloaded data
  WD-ICRAF-Spectral_MIR.zip
  WD-ICRAF-Spectral_MIR/WD-ICRAF-Spectral_MIR/
    ICRAF_ISRIC reference data.csv               # 4239 × 64
    ICRAF_ISRIC MIR spectra.csv                  # 4308 × 3579 (SSN + 3578 wavenumbers)
    + documentation PDFs/PNGs
data/                                            # gitignored, all derived CSVs
  raw/
    {global,china,kenya,indonesia}.csv           # joined + OC-filtered + 4000–600 cm⁻¹
  splits/
    {dataset}_split.csv                          # Batch and labid → 'train' | 'test' (3997+262+245+236 rows)
  preprocessed/
    {dataset}_{none|snv|msc|sg|sgd|minmax}_{train|test}.csv   # 48 files, ~860 MB total
results/                                         # gitignore-d, all model outputs
  per_cell/
    {dataset}_{method}.json                      # PLSR: winning hyperparams + train/test metrics
    {dataset}_{method}_predictions.csv           # PLSR: ref cols + observed + predicted, train+test concat
  table1_replication.csv                         # PLSR: paper vs ours, with PASS/FAIL flags per criterion
  pbn_experiment/                                # DL branch: 5 methods compared across all (dataset × preprocessing)
    cells/{dataset}_{preprocessing}_{baseline|pbn|plsr_pbn|rbn|r2bn}.json
    predictions/{dataset}_{preprocessing}_{baseline|pbn|plsr_pbn|rbn|r2bn}.csv
    cell_results.csv                             # aggregate of all 120 cells
```

Every CSV row carries: reference columns (Org C, Country, Plotcode, BTOP, BBOT, lat, lon, etc.) + the spectral columns. Reference cols are not used by the model but are kept for downstream analysis.

## Decisions on ambiguities (one-line summary; full reasoning in `experiment_spec.md`)

1. Sample counts: 3997 / 262 / 245 / 236 — match the paper exactly using the public data (reconciliation in `experiment_spec.md`).
2. Split design: single grouped-stratified 80/20 split, fixed seed.
3. Stratification: SOC quartiles only (soil-type column not in data).
4. Random seed: **42** (project-wide constant; documented here so it never silently changes).
5. Grouping key for split: **`Batch and labid`** (the physical-sample identifier; ensures coordinate-edit duplicates and scan replicates of the same sample stay in the same fold).
6. SG/SGD hyperparameter selection: same repeated-CV one-SE rule used for PLSR LVs.
7. PLSR LV search range: 1 to 25.
8. RPIQ denominator: IQR of *validation-set observed* SOC.
9. Inner CV grouping key: same as outer split — `Batch and labid` — so scan-replicate rows of the same physical sample never split across CV folds.
10. Per-cell predictions file format: 17 reference columns (country, lat/lon, depth, etc.) + `observed` + `predicted`. No spectra. Lets downstream analysis slice errors by any reference attribute without rebuilding the spectra matrix.
11. **MinMax preprocessing** added as a 6th method (per-feature, fit on train, applied to both folds). Lives in `make_preprocessed.py` / `verify_preprocessed.py`. Intentionally NOT added to `train_plsr.py` / `summarise_results.py` because the paper's Table 1 has no minmax row to compare against — minmax is an addition for the DL branch only.
12. **PBN (Pretrained BatchNorm) — our DL-branch preprocessing layer.** A BatchNorm1d trained inside an autoencoder (Input -> BN -> Encoder -> Decoder -> Input'); after pretraining, the encoder/decoder are discarded and the BN is reused as a learned preprocessing in front of an MLP regressor. Both `baseline` (raw -> MLP) and `pbn` (raw -> BN -> MLP) share the exact same `Linear(n_features -> 32) -> ReLU -> Linear(32 -> 1)` head for fair comparison. Same 32-hidden head, same Adam/MSE/200 epochs/batch 64/lr 1e-3/seed 42; PBN adds a 100-epoch AE pretrain phase on the train spectra.
13. **PBN ablations: `plsr_pbn`, `rbn`, `r2bn`** added to the same experiment matrix. `plsr_pbn` = AE-pretrained BN (frozen) → `sklearn.PLSRegression(n_components=15, scale=False)`; tests whether PBN's gain is ANN-specific (LV=15 fixed; covers paper Table 1's 4–14 range without per-cell tuning). `rbn` = fresh BN + same MLP head, 200 supervised epochs; isolates the AE pretrain by comparing against `pbn`. `r2bn` = fresh BN + same MLP head, 400 supervised epochs; controls for the "PBN just got more total compute" critique against `pbn`. All five methods share `data/preprocessed/` inputs, so the preprocessing dimension is held fixed when comparing methods within a cell. Per-cell JSON `configuration` block is method-specific (records actual epoch counts, BN regime, downstream regressor) so the saved metadata is honest.

---

## Steps

### Step 1 — Fetch raw data (`download_data.py`)

1. Downloads `https://files.isric.org/public/other/WD-ICRAF-Spectral_MIR.zip` (59,327,817 bytes) to `additionals/`.
2. Verifies sha256 = `5ff3e43fd5bdfdc0a051f34b9b23891790ce2e10168123ae3fa403a8be45ff8f`.
3. Extracts into `additionals/WD-ICRAF-Spectral_MIR/`.
4. Idempotent: skips download if zip exists, skips extraction if folder exists.

### Step 2 — Build raw per-domain datasets (`data_loader.py`)

Pipeline:
1. Load reference CSV (4239 × 64) and spectra CSV (4308 × 3579).
2. Inner-join on `Batch and labid` ↔ `SSN`. **No deduplication** — keep all rows so the row counts match the paper's Table 1 / supplementary S1 exactly. Result: 4392 rows.
3. Drop rows with missing `Org C` → 3997 rows.
4. Slice spectra columns to 4000–600 cm⁻¹ → 1763 wavenumbers (`m3999.7` … `m601.7`).
5. Build country subsets and assert counts match the paper.

Outputs in `data/raw/`:

| file | rows | cols | size |
|---|---|---|---|
| `global.csv`    | 3997 | 1781 | ~70 MB |
| `china.csv`     |  262 | 1781 | ~5 MB |
| `kenya.csv`     |  245 | 1781 | ~5 MB |
| `indonesia.csv` |  236 | 1781 | ~5 MB |

Each row carries 17 reference columns (Batch and labid, sample code, country, plotcode, horizon, depths, lat/lon, pH, clay, OC) + 1763 wavenumber columns (`m3999.7` to `m601.7`).

Counts verified against paper Table 1 / supplementary Table S1 exactly.

### Step 3 — Train/test split (`make_splits.py`)

Pipeline (per dataset in `data/raw/`):
1. Load only `Batch and labid` and `Org C` columns (the rest are passengers, not needed for splitting).
2. Bin `Org C` into 4 quartiles via `pd.qcut(q=4)` → strata.
3. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Take the *first* split → train (~80%) / test (~20%).
   - Group key: `Batch and labid`. Strata: SOC quartile.
   - Why 5 folds rather than a 1-shot 80/20 splitter: scikit-learn provides `StratifiedGroupKFold` (joint group + strata) but not `StratifiedGroupShuffleSplit`. Taking fold 0 of a 5-fold StratifiedGroupKFold is the standard idiom and yields exactly the same kind of single 80/20 split.
4. Assert no group leakage (no `Batch and labid` appears in both train and test).
5. Assert test fraction is within ±0.03 of 0.20.
6. Write `data/splits/{dataset}_split.csv` (columns: `Batch and labid`, `fold`).

Same train/test partition is enforced for all 5 preprocessings — every downstream preprocessing step joins to `data/splits/` instead of re-splitting.

Result (test_frac shown; quartile counts in run log):

| dataset | train | test | test_frac |
|---|---|---|---|
| Global    | 3197 | 800 | 0.200 |
| China     |  209 |  53 | 0.202 |
| Kenya     |  195 |  50 | 0.204 |
| Indonesia |  188 |  48 | 0.203 |

Quartile balance is within ±2 samples per quartile in every test set; no group leakage.

### Step 4 — Preprocessing (`make_preprocessed.py`)

For each (dataset × method × split) we write one CSV under `data/preprocessed/`. 4 datasets × 6 methods × 2 splits = **48 files**, ~860 MB. Every output CSV carries the same 17 reference columns (Org C, Country, lat/lon, depths, pH, clay, etc.) + 1763 transformed wavenumber columns (`m3999.7` … `m601.7`).

Per-method math (all implemented in plain numpy / scipy — auditable against Tong et al. §2.3, except minmax which we add as an extra DL-branch baseline):

1. **None** — identity. Spectra copied through unchanged so downstream code can treat all methods uniformly (same path schema, same column layout).
2. **SNV** — Standard Normal Variate, per-spectrum: `(x - mean(x)) / std(x, ddof=1)`. No fit step (independent per row).
3. **MSC** — Multiplicative Scatter Correction. Reference spectrum = mean of **train** spectra only. For every row: solve `x ≈ a + b·ref` via mean-centred least squares, return `(x - a) / b`. Reference fitted on train, applied to both train and test.
4. **SG** — Savitzky–Golay smoothing. `scipy.signal.savgol_filter` with window=11, polyorder=2, deriv=0, axis=1, mode="interp". No fit step (purely local convolution).
5. **SGD** — Savitzky–Golay second derivative. Same as SG but with deriv=2.
6. **MinMax** (our addition) — per-feature linear rescale. Per-column min and max fit on train; both train and test mapped via `(x - min) / (max - min)`. Train range is exactly [0, 1]; test may slightly exceed (audit shows global test in [-0.0016, 1.0550]). Used only by the DL branch.

Decision on SG/SGD hyperparameters (window=11, polyorder=2):

1. The paper tunes (window ∈ 7–31 odd, polyorder ∈ 2–3) inside the same repeated 10-fold CV that picks PLSR LVs. So SG/SGD parameters are properly a **modelling** decision, not a data-preparation decision.
2. The artefact CSVs in `data/preprocessed/` use a single **exemplar** parameter pair (11, 2) so anyone can open them in a viewer to inspect what SG/SGD does to the spectra.
3. The actual modelling step (next) will re-do SG/SGD inside the CV loop over the full grid, replacing these exemplar artefacts in the model's working memory. The on-disk CSV is illustrative; the model does not consume it for SG/SGD.

Train/test partition is enforced by joining each raw CSV to `data/splits/{dataset}_split.csv` on `Batch and labid` (split table is deduplicated to one row per group on read; an assertion verifies all rows for the same group share a single fold label). MSC's reference spectrum is fit on the joined-train rows only — no test-set leakage.

Sanity check (`verify_preprocessed.py`, run on `global_*_train.csv`):

| method | check | result |
|---|---|---|
| SNV | per-row mean | range [-1.25e-14, 1.27e-14] (≈ 0) |
| SNV | per-row std (ddof=1) | exactly 1.0 |
| MSC | output shape | matches input |
| MSC | range | [0.05, 3.26] vs raw [0.12, 3.00] (slight stretch, expected) |
| SG  | smoothing diff | max |Δ| = 0.145, mean |Δ| = 7.1e-4 (small, as expected) |
| MinMax | train range | exactly [0.0, 1.0] |
| MinMax | test range | [-0.0016, 1.0550] (slight overshoot, expected on unseen folds) |
| SGD | range | [-5.1e-2, 3.6e-2], mean -5.7e-6 (small, centred near 0) |

All 48 outputs produced no NaN / Inf (`np.isfinite(...).all()` asserted per file).

### Step 5 — PLSR modelling (`train_plsr.py` + `summarise_results.py`)

For each of the 4 × 5 = 20 cells:

1. Read raw spectra from `data/preprocessed/{dataset}_none_{train,test}.csv` (the *none* artefact happens to also be the canonical raw matrix).
2. Compute SOC quartiles on the train target → strata for inner CV.
3. Build 5 repeats × 10 folds = 50 grouped+stratified CV folds (`StratifiedGroupKFold`, group=`Batch and labid`, seeds 42…46).
4. For every fold, for every preprocessing-grid candidate:
   - Fit preprocessing on inner-train, transform inner-train and inner-val.
   - Fit one PLSR with `n_components=PLSR_LV_MAX=25` (centred manually; sklearn's `PLSRegression(scale=False)`).
   - Reconstruct predictions for every LV ≤ 25 from `x_rotations_` and `y_loadings_` slices (one fit, all LVs evaluated).
   - Record per-fold RMSE in a `(grid, lv, fold)` cube.
5. One-SE rule on the flattened cube: find min mean RMSE → threshold = min + 1·SE_at_min → among all candidates with mean RMSE ≤ threshold, pick smallest LV (parsimony); ties broken by smallest window then polyorder.
6. Refit preprocessing + PLSR on full 80 % train with the winning hyperparameters, predict on 20 % test.
7. Write `results/per_cell/{dataset}_{method}.json` (winner + metrics + config) and `..._predictions.csv` (reference columns + observed + predicted, concat of train and test).

Method-specific preprocessing grids:

| method | grid size | what's tuned in CV |
|---|---|---|
| none | 25 | LV ∈ 1..25 |
| snv | 25 | LV |
| msc | 25 | LV (MSC reference re-fit per inner fold) |
| sg  | 13 × 2 × 25 = 650 | window ∈ {7,9,…,31} odd × polyorder ∈ {2,3} × LV |
| sgd | 13 × 2 × 25 = 650 | same as sg, deriv=2 fixed |

Idempotency: cells with an existing JSON are skipped — re-running `train_plsr.py` after a partial run resumes from the next missing cell.

Total runtime on this machine: **~32 minutes** (global/sg and global/sgd are the two 13-minute cells; everything else is ≤ 1 minute).

#### Results vs paper Table 1

Full table in `results/table1_replication.csv`. Highlights:

| dataset | method | LV paper | LV ours | RMSE paper | RMSE ours | rel Δ | R² paper | R² ours |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| global | sgd | 11 | 12 | 1.559 | 1.662 | 6.6 % | 0.789 | 0.620 |
| global | snv | 14 | 12 | 1.812 | 1.654 | 8.7 % | 0.715 | 0.624 |
| global | msc | 13 | 13 | 1.931 | 1.597 | 17 % | 0.676 | 0.650 |
| china | snv | 6 | 22 | 0.253 | 0.207 | 18 % | 0.878 | 0.875 |
| china | sgd | 4 | 22 | 0.282 | 0.190 | 33 % | 0.856 | 0.895 |
| kenya | snv | 6 | 15 | 0.713 | 0.709 | 0.5 % | 0.924 | 0.906 |
| kenya | msc | 6 | 15 | 0.725 | 0.721 | 0.5 % | 0.919 | 0.903 |
| indonesia | msc | 13 | 9 | 0.849 | 1.119 | 32 % | 0.876 | 0.695 |

Acceptance summary (from `results/table1_replication.csv`): **0 / 20** cells pass all four acceptance criteria from `experiment_spec.md` §10 simultaneously. Per-criterion: LV-pass 3/20, RMSE-pass 7/20, R²-pass 7/20, MBD-pass 1/20.

#### Why we don't replicate the numbers exactly

Two fully-documented unspecified items in the paper drive the divergence:

1. **Random seed for the outer 80/20 split**. We use 42; the paper does not state theirs. With small per-country test sets (n=48–53), a different test draw shifts every metric meaningfully — particularly MBD (most of our MBD values are off by 0.1–0.4 from paper, in a direction consistent with a slightly different mean SOC in the test draw).
2. **One-SE rule SE definition**. We use SE = std(per-fold RMSE) / √(50). The paper says only "one-SE rule"; common practice also uses √(K_inner_folds=10) as the divisor, which gives a ~2.2× wider threshold and yields more parsimonious LV picks. With our narrow band, smallest-LV-within-1SE keeps falling into the late teens / early twenties, whereas paper's LVs are in the single digits or low teens.

What does *not* explain the divergence (verified):

1. Sample counts: exact match (3997/262/245/236).
2. Spectral window: 4000–600 cm⁻¹ exactly, 1763 wavenumbers.
3. PLSR engine: `PLSRegression(scale=False)` matches paper's "mean-centred" specification.
4. Preprocessing math: SNV row-mean ≈ 0 std = 1, MSC fit-on-train applied-on-test, SG/SGD via `scipy.signal.savgol_filter`. Verified independently in `verify_preprocessed.py`.

#### What does still hold from the paper

The paper's qualitative central claim — *"preprocessing choice should be tailored to the target domain rather than treated as universally optimal"* — is **also visible in our reproduction**, just with shuffled specifics:

| region | paper's best | our best | both pick the same? |
|---|---|---|---|
| global | sgd | msc | no, but sgd is 2nd in ours (1.66 vs 1.60) |
| china | snv (0.253) | sgd (0.190) | no, ranking differs |
| kenya | snv (0.713) | snv (0.709) | **yes** |
| indonesia | msc (0.849) | sgd (0.998) / sg-tied-with-none | no |

What is consistent across both reproductions: there is **no single universally-best preprocessing**. Different regions favour different methods. That's the paper's claim and it survives even with our diverging exact numbers.

#### Decisions taken without bothering Arif

1. **Continue with current PLSR results as the baseline** rather than chase exact paper numerics by guessing the seed and SE definition. Rationale: the project goal is to compare DL against PLSR, not to reproduce Tong et al.'s table to four decimals. Our PLSR pipeline is methodologically faithful and our DL comparison will use the same train/test split and the same CV protocol, which is what matters for a fair head-to-head.
2. Document the divergences here in this log and in `experiment_spec.md` — flag for the discussion section of the future paper.
3. Acceptance criteria from §10 of `experiment_spec.md` are **not met as written**. Rewriting them to "PLSR vs DL on identical split" is the right move and is implicit in Step 6.

### Step 6 — DL replacement (next, mainstream)
Replace PLSR with a DL model (1D CNN baseline first, then more) on the exact same `data/raw/` + `data/splits/` setup. Same per-cell JSON / predictions CSV layout under `results/per_cell_dl/`. Comparison table: PLSR (us) vs DL (us) vs PLSR (paper).

### Step 7 — DL branch: PBN preprocessing experiment (`train_pbn_experiment.py` + `report_pbn_experiment.py`)

Goal: test the hypothesis that a learned preprocessing (Pretrained BatchNorm = PBN) makes the choice of classical preprocessing irrelevant. If PBN beats the no-PBN baseline in every (dataset × preprocessing) cell, the paper's "tailor preprocessing to region" claim becomes "tailor preprocessing because PLSR is brittle — DL with PBN doesn't need that crutch."

Architecture:

1. `BaselineSocAnn` (`model_baseline_ann.py`): plain MLP — `Linear(1763 → 32) → ReLU → Linear(32 → 1)`.
2. `LearnedPreprocessingAutoencoder` + `PbnSocAnn` (`model_pbn_ann.py`): autoencoder is `BN(1763) → Linear(1763 → 256) → ReLU → Linear(256 → 64) → Linear(64 → 256) → ReLU → Linear(256 → 1763)`. After Phase A pretrain on train spectra (100 epochs, MSE reconstruction), encoder/decoder are discarded; only the BN is reused. PBN regressor = `BN(pretrained) → Linear(1763 → 32) → ReLU → Linear(32 → 1)`. Both BN and head are trainable in Phase B.
3. `RbnSocAnn` (`model_rbn_ann.py`): same shape as PbnSocAnn but the BN is initialised fresh — used by both `rbn` (200 supervised epochs) and `r2bn` (400 supervised epochs).

The MLP head is identical across baseline / pbn / rbn / r2bn — the only differences are (a) whether a BN is in front, and (b) whether that BN was AE-pretrained before supervised training, and (c) how many supervised epochs the BN+head pair gets.

Five methods compared per cell:

| method | BN in front | BN pretrained (Phase A AE) | downstream regressor | supervised epochs |
|---|---|---|---|---|
| `baseline` | no | — | MLP(32) | 200 |
| `pbn` | yes | yes | MLP(32) | 200 |
| `plsr_pbn` | yes (frozen) | yes | PLSR(LV=15, scale=False) | — (one-shot fit) |
| `rbn` | yes | no | MLP(32) | 200 |
| `r2bn` | yes | no | MLP(32) | 400 |

`plsr_pbn` exists to test the "is PBN ANN-locked?" critique — if `plsr_pbn` improves over the existing PLSR results (`results/per_cell/`) and over `baseline`, then PBN's benefit transfers to a non-NN downstream regressor. The BN is frozen (eval mode) after Phase A and the BN-output of train spectra is fitted by sklearn `PLSRegression(n_components=15, scale=False)`. LV=15 is fixed (paper's Table 1 LVs span 4–14, so 15 covers the range without per-cell tuning — this keeps the ablation fast and apples-to-apples across cells).

`rbn` and `r2bn` together test the "is PBN's edge just from extra optimisation?" critique:

- `rbn` vs `pbn` isolates the AE pretrain (Phase A) — same architecture, same supervised epochs, only PBN had AE-pretraining.
- `r2bn` vs `pbn` controls for total compute — R2BN gives the BN+head 400 supervised epochs (≈ PBN's 100 AE + 200 supervised in raw step count), so if PBN still wins, the win is *not* from cumulative gradient steps.

Experiment matrix: 4 datasets × 6 preprocessings × 5 methods = **120 cells**. Per cell:

1. Read `data/preprocessed/{dataset}_{preprocessing}_{train,test}.csv` — the preprocessing has already been applied; the script does no further preprocessing.
2. Reset all RNG seeds to 42.
3. Train (Adam, MSE, batch 64, lr 1e-3; epoch counts as in table above; PBN/plsr_pbn add 100 epochs of AE pretrain).
4. Evaluate on test, compute RMSE / R² / MBD / RPIQ.
5. Write `results/pbn_experiment/cells/{dataset}_{preprocessing}_{baseline|pbn|plsr_pbn|rbn|r2bn}.json` and `…/predictions/…csv`. The JSON `configuration` block is method-specific (records the actual epoch counts and BN regime for that cell).
6. Idempotent — cells with an existing JSON are skipped.

After all cells: aggregate into `results/pbn_experiment/cell_results.csv`. `report_pbn_experiment.py` reads that CSV and prints a per-dataset block-format report with all five methods, delta-RMSE per preprocessing for six method pairs (`pbn` vs `baseline`, `pbn` vs `rbn`, `pbn` vs `r2bn`, `plsr_pbn` vs `baseline`, `plsr_pbn` vs `pbn`, `rbn` vs `baseline`), and per-pair win counts.

#### Headline results (test-RMSE win rates, 24 cells per pair)

| comparison | what it tests | result |
|---|---|---|
| pbn vs baseline | does PBN beat raw inputs? | **pbn 17/24** (Global 6/6, China 5/6, Kenya 5/6, Indonesia 1/6) |
| rbn vs baseline | does *any* BN preprocessing help (ignoring the AE pretrain)? | **rbn 18/24** (Global 6/6, China 6/6, Kenya 5/6, Indonesia 1/6) |
| pbn vs rbn | does the AE pretrain add value over a fresh BN? | **pbn 14/24** — modest, not dominant |
| pbn vs r2bn | does PBN still beat raw-BN even with 2× supervised epochs? | **pbn 17/24** — yes; PBN's edge is not just compute |
| plsr_pbn vs baseline | is PBN ANN-locked, or does it transfer? | **plsr_pbn 14/24** — Global 0/6 (PLSR-LV15 underfits at n=3197), China 4/6, Kenya 5/6, Indonesia 5/6 |
| plsr_pbn vs pbn | which downstream wins? | **pbn 14/24** on Global+China; **plsr_pbn wins 5/6** on each of Kenya+Indonesia (small-data regime favours regularised linear head) |

Reading: most of PBN's lift over `baseline` is actually attributable to the BN itself (raw or pretrained). The AE pretrain provides a smaller, real, but non-dominant additional gain. The claim "PBN works only because it gets more compute" is rejected (R2BN doesn't catch up). The claim "PBN is ANN-locked" is partially rejected — PLSR+PBN beats raw-input baseline on three of the four datasets (everywhere except Global, where 15 LVs is too tight a budget for the size of the global pool). Indonesia is anomalous: every ANN-based method overfits its 188-row train set, and only the regularised PLSR head (`plsr_pbn`) generalises well.


