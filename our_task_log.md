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
  preprocessed/                                  # (later) {dataset}_{none|snv|msc|sg|sgd}_{train|test}.csv
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

### Step 4 — Preprocessing (next)
The five regimes — None / SNV / MSC / SG / SGD — applied **fit-on-train, transform-on-test**, written as `data/preprocessed/{dataset}_{method}_{train|test}.csv`.
