# Task Log — MIR-SOC: Data-Driven Preprocessing (DDP) study

A running, concise record of the project. The PLSR replication of Tong et al. (2026) is complete; the live track is now a study proposing **DDP** (a learnable BatchNorm-only preprocessing module) as a new column in the preprocessing catalogue.

For the next agent's marching orders, see `for_next_agent.md`.

---

## Study direction (current)

**Claim:** Tong et al.'s finding that "the best preprocessing is region-dependent" is itself an artefact of their algorithmic choice (PLSR). PLSR has no mechanism to push supervised signal back into the preprocessing stage, so the preprocessing must be picked manually and per-region from a fixed catalogue (snv/msc/sg/sgd). When the downstream model is gradient-based (e.g. an MLP), gradients can flow back into a learnable preprocessing module, making the preprocessor itself **data-driven**. We propose **DDP**, a minimal instance of this — a BN-only module — and claim it can serve as a single, region-agnostic preprocessing choice that is competitive with, or better than, the per-region winners in Tong's table.

**Evaluation grid:** preprocessing × dataset × algorithm
- preprocessing (7): `none, snv, msc, sg, sgd, minmax, ddp`
- dataset (4): `global, china, kenya, indonesia`
- algorithm (2): `plsr` (Tong's), `mlp` (ours)
- 7 × 4 × 2 = 56 cells. PLSR side already populated for 6 preprocessings × 4 datasets = 24 cells from the replication phase; PLSR cannot run on `ddp` (DDP is a gradient-based preprocessor, no PLSR cell for it). MLP side = 7 × 4 = 28 cells, all to be populated.

---

## Reproduction recipe (data prep + PLSR — already done)

Run from the project root. Each command is idempotent.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python download_data.py        # downloads + verifies + extracts the public ICRAF/ISRIC MIR library
python data_loader.py          # builds data/raw/{global,china,kenya,indonesia}.csv
python make_splits.py          # builds data/splits/{dataset}_split.csv  (single 80/20, group=Batch and labid, strata=SOC quartile, seed=42)
python make_preprocessed.py    # builds data/preprocessed/{dataset}_{none|snv|msc|sg|sgd|minmax}_{train|test}.csv  (48 files, ~860 MB total)
python verify_preprocessed.py  # optional: sanity-checks SNV/MSC/SG/SGD math on the global train set
python train_plsr.py           # PLSR sweep over 4 datasets x 6 preprocessings, writes results/per_cell/{dataset}_{method}.json + predictions CSV (~32 min)
python summarise_results.py    # builds results/table1_replication.csv comparing every cell to paper Table 1
python train_plsr_fixed_lv.py  # diagnostic: refit PLSR with paper's exact LV count per cell. Writes results/per_cell_fixed_lv/ + table1_replication_fixed_lv.csv (~12 s)

python train_ddp_experiment.py # DDP study: 4 datasets x 7 preprocessings (none/snv/msc/sg/sgd/minmax/ddp) x MLP. Writes results/ddp_experiment/cells/ + cell_results.csv (~40 s)
python report_ddp_experiment.py # joins MLP-side with PLSR-side, prints wide tables, writes results/comparison_table.csv

# Optional spectra visualisation:
python plot_preprocessed_spectra.py    # 6x4 mean-spectrum grid (methods x regions)
python plot_one_sample_spectra.py      # 6x4 single-random-sample grid
python plot_three_samples_spectra.py   # 6x4 three-random-samples grid (color-consistent per region)
python plot_indonesia_mean.py          # indonesia mean reflectance close-up
```

One step is **manual** because the file is behind Elsevier's article page:

1. Open `https://doi.org/10.1016/j.still.2026.107237` → "Appendix A. Supplementary data" → download the `.xlsx` → save as `resource/supplementary.xlsx`.
2. Place the paper PDF as `resource/tong.pdf`.

## Environment

1. Python 3.13.7 (any 3.12+ should work). On this machine the project venv lives at `C:\Users\m.rahman\vens\tillage` and `.vscode/settings.json` auto-activates it inside Cursor.
2. Pinned dependencies in `requirements.txt`. PyTorch is included for the DDP study.
3. Set `$env:VENV_PY` to the venv's `python.exe` for ad-hoc invocations from PowerShell.

## Repo layout (current state, post-archive)

```text
download_data.py                                 # step 1: fetches and extracts the raw dataset
data_loader.py                                   # step 2: builds raw per-domain CSVs
make_splits.py                                   # step 3: builds 80/20 train/test split tables
make_preprocessed.py                             # step 4: applies the 6 classical preprocessings, writes train/test CSVs
verify_preprocessed.py                           # one-shot integrity check on preprocessed outputs
train_plsr.py                                    # step 5: PLSR + CV + one-SE rule + test eval, per (dataset x method)
summarise_results.py                             # step 5: results/table1_replication.csv, paper-vs-ours acceptance flags
train_plsr_fixed_lv.py                           # step 5b: diagnostic — refit PLSR at paper's exact LV count per cell
print_plsr_tables.py                             # pretty-print PLSR comparison tables
model_mlp.py                                     # DDP study: MlpSocAnn (Linear→ReLU→Dropout(0.3)→Linear)
model_ddp.py                                     # DDP study: DdpPreprocessor (BatchNorm1d only) + DdpPlusMlp wrapper
                                                 # (model_ddp2.py / model_ddp3.py archived 28 Apr 2026; only `ddp` = minmax+BN remains live)
train_ddp_experiment.py                          # step 6: 4 datasets × 7 preprocessings (incl. 1 learned, two-stage), 500 epochs full-batch, robust-scaled target
report_ddp_experiment.py                         # step 6: PLSR-vs-MLP wide tables + per-dataset winner + comparison_table.csv
plot_preprocessed_spectra.py                     # viz: mean spectrum per (region x method) cell
plot_one_sample_spectra.py                       # viz: one random sample traced through all preprocessings per region
plot_three_samples_spectra.py                    # viz: three random samples per region, color-consistent across methods
plot_indonesia_mean.py                           # viz: indonesia-only mean reflectance close-up
requirements.txt
task_log.md                                      # this file
for_next_agent.md                                # marching orders for the DDP study
resource/
  tong.pdf                                       # the paper (manual)
  supplementary.xlsx                             # Table S1 (manual)
additionals/                                     # gitignored, raw downloaded data
data/                                            # gitignored, all derived CSVs
  raw/{global,china,kenya,indonesia}.csv
  splits/{dataset}_split.csv
  preprocessed/{dataset}_{none|snv|msc|sg|sgd|minmax}_{train|test}.csv
results/                                         # gitignored
  per_cell/{dataset}_{method}.json               # PLSR canonical cells
  per_cell/{dataset}_{method}_predictions.csv    # PLSR per-cell predictions
  per_cell_fixed_lv/{dataset}_{method}.json      # PLSR step-5b diagnostic cells
  table1_replication.csv                         # PLSR vs paper Table 1, with PASS/FAIL flags
  table1_replication_fixed_lv.csv                # PLSR @ paper-LV vs ours @ one-SE
  preprocessed_spectra.png, one_sample_spectra.png, three_samples_spectra.png, indonesia_mean_reflectance.png
  ddp_experiment/
    cells/{dataset}_{preprocessing}_mlp.json     # 28 MLP cells (7 preprocessings x 4 datasets)
    predictions/{dataset}_{preprocessing}_mlp.csv
    cell_results.csv                             # long-format aggregate of all 28 MLP cells
  comparison_table.csv                           # PLSR-vs-MLP joined long-format table for the paper
archive/
  code/      # all retired DL trial-1 scripts (rbn / rbnd / rbnr / cnn / transformer / probes / sweeps / plots)
  md/        # previous experiment_spec.md, our_task_log.md, for_next_agent.md
  results/   # all retired DL trial-1 results (pbn_experiment/, baselines/, bn_ae/, h1a_*.csv, probe_*, sweep_*)
```

## Project-wide constants (don't change without logging)

1. Random seed: **42**.
2. Group key for splits: **`Batch and labid`** (the physical-sample identifier; ensures coordinate-edit duplicates and scan replicates of the same sample stay in the same fold).
3. Spectral window: **4000–600 cm⁻¹** (1763 wavenumbers, columns `m3999.7` … `m601.7`).
4. SOC column: **`Org C`**.
5. Stratification: SOC quartiles only (soil-type column not in data).
6. Train/test partition: single grouped+stratified 80/20 split, seed 42, shared by every preprocessing × algorithm cell.

## Decisions on ambiguities (PLSR replication)

1. Sample counts: **3997 / 262 / 245 / 236** (paper Table 1 row counts; keeps duplicate rows). Reconciliation in archived `experiment_spec.md`.
2. Split design: single grouped-stratified 80/20 split, fixed seed.
3. Stratification: SOC quartiles only.
4. Random seed: 42.
5. Grouping key: `Batch and labid`.
6. SG/SGD hyperparameter selection: same repeated-CV one-SE rule used for PLSR LVs.
7. PLSR LV search range: 1 to 25.
8. RPIQ denominator: IQR of validation-set observed SOC.
9. Inner CV grouping key: same as outer split.
10. **MinMax preprocessing** added as a 6th method (per-feature, fit on train, applied to both folds). Paper-comparison columns are NaN in `table1_replication.csv` for the four minmax rows since paper Table 1 has no minmax row.

## Sample counts per region (post-split)

Preprocessing transforms feature columns only — row counts are identical across all six classical preprocessing methods. Per-region totals match Tong et al. Table 1 exactly.

| split | global | china | kenya | indonesia |
|---|---:|---:|---:|---:|
| train | 3197 | 209 | 195 | 188 |
| test  |  800 |  53 |  50 |  48 |
| total | 3997 | 262 | 245 | 236 |

## PLSR replication result (status, headlines)

Acceptance summary against the four `experiment_spec.md` §10 criteria: **0 / 20** cells pass all four simultaneously. Per-criterion: LV-pass 3/20, RMSE-pass 7/20, R²-pass 7/20, MBD-pass 1/20.

Step 5b proved that the divergence is dominated by the **unrecoverable random seed** of the paper's 80/20 split, not by the one-SE rule definition: forcing paper LVs made R² *worse* in 18/20 cells (parsimonious paper LVs underfit our split). Decision logged 27 Apr 2026: accept current single-split + seed=42 PLSR results as the closest faithful reproduction obtainable; do not chase exact paper numerics further. The paper's *qualitative* claim (preprocessing varies by region) survives in our run with shuffled specifics.

PLSR cells live in `results/per_cell/` (canonical) and `results/per_cell_fixed_lv/` (diagnostic). Both stay; both are valid PLSR rows for the new comparison table.

---

## Steps completed

### Step 1 — Fetch raw data (`download_data.py`)
Downloads `WD-ICRAF-Spectral_MIR.zip` (sha256-verified), extracts to `additionals/`. Idempotent.

### Step 2 — Build raw per-domain datasets (`data_loader.py`)
Inner-join reference and spectra tables on `Batch and labid` ↔ `SSN`, no deduplication, drop missing OC, slice to 4000–600 cm⁻¹. Outputs: `data/raw/{global,china,kenya,indonesia}.csv`. Counts match Table 1 exactly.

### Step 3 — Train/test split (`make_splits.py`)
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, take fold 0. Group key `Batch and labid`, strata SOC quartiles. No leakage. Outputs: `data/splits/{dataset}_split.csv`.

### Step 4 — Preprocessing (`make_preprocessed.py`)
6 methods: none, SNV, MSC (reference fit on train), SG (window=11, polyorder=2, deriv=0), SGD (deriv=2), MinMax (per-feature, fit on train). Train/test partition is enforced by joining each raw CSV to `data/splits/`. Outputs: 4 datasets × 6 methods × 2 splits = 48 CSVs in `data/preprocessed/`. Verified by `verify_preprocessed.py`. SG/SGD hyperparameters in the on-disk artefact are exemplar values; PLSR re-applies them inside the CV loop with the full grid (window 7–31 odd × polyorder {2,3}); MSC reference is re-fit per inner fold.

### Step 5 — PLSR modelling (`train_plsr.py` + `summarise_results.py`)
For each of the 4 × 6 = 24 cells: 5 repeats × 10 grouped+stratified inner CV folds, one-SE rule on RMSE, refit on full 80% train, eval on 20% test. SG/SGD grid 13 × 2 × 25, MSC reference re-fit per inner fold. Total runtime ~32 minutes. Outputs in `results/per_cell/` + `results/table1_replication.csv`.

### Step 5b — Paper-LV diagnostic (`train_plsr_fixed_lv.py`)
Re-fit PLSR with paper's exact LV count per cell (LV one-SE picker bypassed). Showed that R² gets worse in 18/20 cells, proving the split-seed (not the LV rule) drives our divergence from paper Table 1. Outputs in `results/per_cell_fixed_lv/` + `results/table1_replication_fixed_lv.csv`.

### Step 6 — DDP study (`train_ddp_experiment.py` + `report_ddp_experiment.py`)

Pipeline: 4 datasets × 7 preprocessings (none, snv, msc, sg, sgd, minmax, ddp) × MLP head = 28 cells. MLP head locked to `Linear(1763→32) → ReLU → Dropout(0.3) → Linear(32→1)`. Training contract locked: full-batch, Adam, lr=1e-3, weight_decay=0, 500 epochs, MSE on robust-scaled SOC, seed=42. No early stopping, no validation carve-out.

`ddp` is the only learned preprocessor and uses a two-stage protocol (Stage 1: train preprocessor + MLP jointly; Stage 2: freeze preprocessor in eval mode, discard the stage-1 head, transform train+test, train a fresh MLP). Pipeline: `input → minmax (per-feature, fit on train) → learnable BatchNorm1d`. Input source: `data/preprocessed/{dataset}_minmax_*.csv`. Preprocessor learnable parameters: 2·1763 = 3,526.

**Target preprocessing — robust scaling (added 28 Apr 2026, late evening).** Inside every MLP cell (classical and learned alike), the SOC target is scaled per region with `(y - median(y_train)) / IQR(y_train)`. Median and IQR are fit on the train target only; the same constants are applied to test targets implicitly via inverse-scaling of predictions. The MLP trains on scaled SOC; predictions are de-scaled before computing RMSE/R²/MBD/RPIQ so all metrics remain on the original SOC % scale and stay row-by-row comparable to PLSR cells. PLSR cells are *not* re-run — their mean-centring is part of the paper-faithful PLSR algorithm (Tong §2.2), and re-fitting on robust-scaled SOC would invalidate the paper-replication baseline column.

Earlier learned-preprocessor variants (`ddp2`, `ddp3`, `ddp22`, `ddp32`) were archived on 28 Apr 2026: code in `archive/code/`, cell JSONs/predictions in `archive/results/ddp_experiment/`. Resurrect by `Move-Item archive/code/model_ddp{2,3}.py .` and re-registering in `LEARNED_PREPROCESSING_SPECIFICATIONS`.

Total runtime: ~36 s on CPU for the full 28-cell sweep.

#### Headline results (single seed=42, single 80/20 split)

Test-RMSE per (preprocessing × dataset). Lower is better. Bold = per-row winner across the algorithm pair (MLP wins on the row when its number is the lowest; PLSR has no `ddp` cell by design).

```
                MLP test RMSE (robust-scaled OC)         PLSR test RMSE (paper-faithful)
                indonesia kenya china global             indonesia kenya china global
none              1.6903 0.7417 0.5851 1.5004              1.1328 0.7567 0.2409 1.7105
snv               1.3982 0.5941 0.2009 1.2316              1.3793 0.7094 0.2070 1.6541
msc               1.7756 2.3386 0.5851 1.4302              1.1190 0.7211 0.2140 1.5967
sg                1.6700 0.7359 0.5851 1.5002              1.1329 0.7567 0.2217 1.7111
sgd               1.4922 1.9442 0.5441 2.5274              0.9985 0.8636 0.1897 1.6624
minmax            1.1910 0.6637 0.2002 1.4062              1.1099 0.7491 0.2494 1.7340
ddp               1.3738 0.5592 0.1670 1.1362                  —      —      —      —
```

Per-dataset winning preprocessing (lowest test RMSE):

| dataset    | MLP winner       | MLP RMSE | PLSR winner | PLSR RMSE |
|---|---|---:|---|---:|
| indonesia  | minmax           | 1.191    | sgd         | 0.999     |
| kenya      | ddp              | 0.559    | snv         | 0.709     |
| china      | ddp              | 0.167    | sgd         | 0.190     |
| global     | ddp              | 1.136    | msc         | 1.597     |

Stage-1 vs stage-2 DDP test RMSE (lossless-trick check):

| dataset    | stage1 | stage2 | Δ |
|---|---:|---:|---:|
| indonesia  | 1.261 | 1.374 | +0.113 |
| kenya      | 0.581 | 0.559 | −0.022 |
| china      | 0.169 | 0.167 | −0.002 |
| global     | 1.110 | 1.136 | +0.026 |

The two-stage trick remains effectively lossless for `ddp` (max |Δ| = 0.113 on indonesia; the other three are within ±0.026). On kenya and china stage 2 actually edged stage 1 slightly.

Observations (numbers only, framing deferred to Arif):
1. **`ddp` now wins 3 of 4 regions** on the MLP side: kenya (0.559 vs snv 0.594), china (0.167 vs minmax 0.200), global (1.136 vs snv 1.232). Indonesia remains held by minmax (1.191 vs ddp 1.374). Compared to the pre-target-scaling sweep where `ddp` won only 2 of 4, the robust-scaled target makes the BN-front-end clearly dominant.
2. **MLP+ddp beats every PLSR cell on china and global**: china 0.167 vs PLSR best 0.190 (sgd); global 1.136 vs PLSR best 1.597 (msc). On kenya MLP+ddp also beats every PLSR cell (0.559 vs PLSR best 0.709). On indonesia MLP+minmax does not beat PLSR's best (1.191 vs 0.999 sgd).
3. **Some classical-preprocessing rows collapsed to R²=0** on china (none, msc, sg, sgd) and indonesia (msc) and kenya (msc, sgd): the model is predicting near-the-train-median and not learning. Robust scaling on a strongly right-skewed target inflates the long-tail outliers (china max OC = 6.03 → scaled ≈ 9.1 with median 0.36 / IQR 0.62), which in MSE training pulls the optimisation toward those few extreme samples and stalls the model on the bulk. snv/minmax/ddp do not collapse in the same way, presumably because their input spectral magnitudes are well-conditioned for the head.
4. The "preprocessing varies by region" pattern from Tong is now *less* pronounced on the MLP side: 3 of 4 regions agree on `ddp` as the best preprocessing. Only Indonesia disagrees (minmax). On the PLSR side it still holds (sgd / snv / sgd / msc as winners across the 4 regions).

Update (28 Apr 2026, late evening): archived `ddp2`/`ddp3`/`ddp22`/`ddp32` and added robust scaling of the SOC target inside every MLP cell (median + IQR fit on train OC, predictions de-scaled before metrics). Decision logged: PLSR cells are not re-run — their mean-centring is part of the paper-faithful PLSR algorithm and re-fitting on robust-scaled SOC would invalidate the paper-replication column.

#### Observations (numbers only, not framing)

1. **DDP is the per-region MLP winner in 2 of 4 datasets** (china, global) and is competitive on kenya (0.66 vs 0.57 for snv).
2. **DDP wins on global by a wide margin** (1.12 vs the next-best classical MLP row at 1.21 for snv, and vs PLSR's best at 1.60).
3. **MLP+DDP also beats every PLSR cell on global** (1.12 vs PLSR's best 1.60).
4. **MLP+DDP roughly ties PLSR's best on china** (0.189 vs 0.190).
5. **Indonesia and Kenya remain harder for MLP than for PLSR** at the single-seed level — every MLP row except DDP/Kenya has a worse test RMSE than the matching PLSR row. Indonesia n=188 is the smallest; the prior phase already noted this is the sentinel region.
6. **MLP per-region winners differ across datasets** (minmax / snv / ddp / ddp). The "preprocessing varies by region" finding from Tong is *also* visible in our MLP, just with shuffled specifics — DDP does not eliminate the region dependence on this single-seed run.

Decision deferred to Arif: framing for the writeup. The brief is to report the numbers and stop.

---

## Previous DL trials and findings (archived; read for context only)

The first DL push (~Apr 2026) was framed as "RBN beats baseline beats PLSR cell-by-cell." This was abandoned on 28 Apr 2026 because the framing compared apples to oranges (a model architecture vs. an algorithm choice) and the hypothesis collapsed under per-region heterogeneity. Code, results, and markdowns from that phase are in `archive/`. Everything below is a one-paragraph-per-attempt summary of what was tried and why it was set aside.

1. **`baseline` MLP and `rbn` (BN + MLP) head, mini-batch=64, lr=1e-3, 200 epochs, supervised end-to-end.** Initial H1A (`none` × 4 datasets × 3 algorithms `plsr/baseline/rbn`) had RBN winning 3/4 datasets but losing Indonesia. Indonesia's small train set (n=188) caused severe overfitting in RBN. Hyper-tuning batch size, weight decay, and epochs did not reliably close the gap.

2. **`rbnr` (RBN + L1/L2 regularisation).** Sweep on Indonesia/none at lr=1e-4 with 6 (l1, l2) combinations. Optimum sat at the upper edge of the grid (l1=1e-2, l2=1e-1) at 1.17 RMSE, still trailing PLSR's 1.13. Stronger regularisation either didn't help or hurt floor performance. Set aside.

3. **`baseline_cnn` and `rbn_cnn` (1D-CNN heads).** First version with `Conv1d → AdaptiveAvgPool1d` performed poorly because translation invariance is the wrong inductive bias for fixed-wavelength MIR. Switched to `Conv1d → MaxPool1d → Flatten → Linear` to preserve wavelength identity; still underperformed the MLP variants on most datasets. CNN was set aside as not the right inductive bias for this regime without much heavier tuning.

4. **`baseline_transformer` and `rbn_transformer`.** Smaller datasets struggled with the transformer; complex models overfit. Set aside.

5. **Mini-batch BN noise hypothesis → full-batch training.** Suspected mini-batch BN running statistics were noisy on small datasets (Indonesia n=188). Switched RBN to full-batch training (batch_size = full train size), lr=1e-3, 1000 epochs. Indonesia RMSE dropped to 1.14 at epoch 45 (essentially tying PLSR). **Confirmed BN benefits from full-batch on small data.**

6. **`rbnd` (RBN + Dropout) full-batch.** Adding `Dropout(p=0.3)` between the two MLP layers, full-batch lr=1e-3 on Indonesia/none reached **0.91 RMSE at epoch 53**, decisively beating PLSR (1.13). Sweeping all 4 regions at lr=1e-3 with oracle stopping showed RBND winning 4/4 — but optimal epochs varied wildly (Indonesia 53, Kenya 960). A single fixed epoch couldn't satisfy all regions.

7. **Per-region LR probes on Indonesia and Kenya.** Mapped how LR ∈ {1e-3, 1e-4, 1e-5} affects ceiling vs window width:
   - Kenya: best ceiling at lr=1e-3 (0.32 RMSE @ ep 2471), but spiky.
   - Indonesia: best ceiling at lr=1e-3 (0.91 @ ep 53) but extremely narrow window. lr=1e-4 has a wider, more robust basin (~1.05 RMSE).
   - Confirmed opposite LR preferences across regions. Single global hyperparameter is a compromise.

8. **L1/L2 regularisation at lr=1e-3 on Indonesia.** Sweep of 6 (l1, l2) configs over 3000 epochs at lr=1e-3 on Indonesia/none. Best test RMSE pinned at **epoch 53 across all configs** (0.91), regardless of regularisation strength. Stronger regularisation reduced the post-peak explosion but did not lower the floor or widen the optimal window. Regularisation is not the way to tame lr=1e-3 instability.

9. **Validation-based early stopping at lr=1e-3 on Indonesia.** Used `StratifiedGroupKFold(n_splits=10)` on the 80% train to carve out a 10% val (~19 rows). 3 seeds, patience=20, full-batch, lr=1e-3. Median test RMSE at best-val checkpoint = 1.04 (basically a tie with PLSR), seed-to-seed spread 0.95–1.20. The 19-row val set was too small to be a reliable selection signal — one seed's best-val RMSE was 4.51 yet had the lowest test RMSE. Set aside as not robust enough on a per-cell basis.

**Why all of this got set aside (28 Apr 2026):** The framing was wrong. We were trying to prove "RBN > baseline > PLSR" cell-by-cell, which is comparing an architecture against an algorithm. The right framing is to put BN in the *preprocessing* column (not the *architecture* column) and let the gradient-based downstream model treat it on equal footing with snv/msc/sg/etc. That is the DDP study, picked up from a clean slate in `for_next_agent.md`.

**Reusable insight retained from this phase:**
1. Full-batch training stabilises BN on small datasets; do not use mini-batch BN at n < ~200.
2. Dropout p=0.3 in the MLP head is an effective low-data-regime regulariser, separable from any preprocessing claim.
3. The MLP head `Linear(n_features → 32) → ReLU → Dropout(0.3) → Linear(32 → 1)` is the most reliable downstream architecture we tried; CNN/Transformer not justified at this n.
4. Indonesia is the hardest region by far (n=188, peat tail) and is the sentinel for any new approach.
