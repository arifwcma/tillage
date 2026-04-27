# Tong et al. (2026) — Experiment Spec for Replication

Source: `resource/tong.pdf` — *Mid-infrared spectroscopy for soil organic carbon estimation. Part II: Evaluating preprocessing at global and national scales*. Soil & Tillage Research 262 (2026) 107237.

This document captures every experimental detail needed to reproduce the paper's PLSR pipeline before swapping PLSR for DL. Items marked **[UNSPECIFIED]** are not stated in the paper and must be confirmed (either from supplementary material, the ISRIC source, or with Arif's decision).

---

## 1. Data source

1. Library: ISRIC global soil IR library — `https://www.isric.org/explore/library`.
2. Original library size: 4438 samples / 754 soil profiles / 56 countries.
3. After retaining samples with non-missing laboratory SOC: **n = 3997** (the "Global" set).
4. Sample preparation (already done by ISRIC): air-dried, ground, sieved to ≤ 2 mm.
5. SOC reference method: **Walkley–Black** (Heanes, 1984).
6. Spectrometer: **Bruker Tensor-27 FT-IR** diffuse-reflectance.
7. Per sample acquisition: 4 replicate wells × 32 scans; scans averaged. **[UNSPECIFIED]** Whether the published library exposes individual replicates or only the per-sample average — must be checked against the ISRIC download.
8. Spectral window used in modelling: **4000–600 cm⁻¹** (held constant across all experiments).

## 2. Subsets

| Subset | Modelling n (paper) | Descriptive n (paper §2.1) |
|---|---|---|
| Global | 3997 | 3997 |
| China | 262 | 245 |
| Kenya | 245 | 239 |
| Indonesia | 236 | 226 |

1. Modelling subsets restricted to "records with complete spectra and QC criteria consistent with prior work" (Table S1).
2. **[UNSPECIFIED — likely paper inconsistency]** Modelling n is *higher* than descriptive n for each country, despite the paper saying modelling adds *additional* QC filters on top of "non-missing OC". This needs to be resolved by reading Table S1 (supplementary) and/or inspecting the ISRIC raw data.
3. Peat / high-organic flag: SOC ≥ 12 % (descriptive only — *not* used to drop samples).

## 3. Preprocessing options compared (5)

1. **None** — raw absorbance.
2. **SNV** — Standard Normal Variate (per-spectrum mean-centre and scale).
3. **MSC** — Multiplicative Scatter Correction (additive + multiplicative correction relative to a reference; reference = mean spectrum of training set, by convention).
4. **SG** — Savitzky–Golay smoothing only.
5. **SGD** — Savitzky–Golay second derivative (with optional smoothing governed by the SG window).

SG / SGD hyperparameters:
1. Window length grid: **7–31 points** (odd values assumed).
2. Polynomial order grid: **2–3**.
3. Selected by grid search. **[UNSPECIFIED]** Selection criterion — paper does not state explicitly. Most plausible: within the same repeated-10-fold-CV used for PLSR latent-variable tuning, choose (window, order) minimising RMSECV (or via the one-SE rule, same as for LVs).

## 4. Splitting and grouping

1. **Within-domain only**: Global→Global; Country→Country. No cross-domain transfer, no spiking.
2. **Outer split**: 80 % calibration / 20 % external validation.
3. **Stratification**: by SOC quartiles (Q1–Q4 of training-set SOC). Soil type was *also* claimed as a stratum "where available", but §2.1 explicitly states soil taxonomy fields were unavailable in the ISRIC extract used → effectively SOC quartiles only.
4. **Grouping**: replicates from the same sample/site/depth are kept in the same fold to prevent leakage. **[UNSPECIFIED]** Exact grouping key (sample ID? profile_id × depth? lat/lon × depth?). Must be inferred from the ISRIC schema.
5. **Random seed**: "fixed random seed" — **[UNSPECIFIED]** value. We must pick one and document it; final-table reproduction may differ slightly from paper if their seed is different.
6. **[AMBIGUITY]** §2.2 simultaneously describes (a) a single stratified 80/20 split with a fixed seed, and (b) a "grouped, stratified K-fold scheme in which each outer fold held out ~20 %". Reported metrics (one row per dataset×preprocessing) suggest a **single** 80/20 split, not averaged-over-folds. Adopt single split as the working interpretation.

## 5. Modelling

1. Algorithm: **PLSR** with mean-centred predictors.
2. Latent variables (LVs) tuned on the calibration split via **repeated 10-fold cross-validation, 5 repeats**.
3. LV selection rule: **one-SE rule** — fewest LVs whose RMSECV is within 1 standard error of the minimum.
4. Refit on the full 80 % calibration split with the chosen LV count.
5. Evaluate on the 20 % external validation split.
6. Preprocessing fitting:
   1. Inside CV: re-fit preprocessing on each training fold, apply to that fold's assessment portion.
   2. Outer holdout: preprocessing parameters fit on the full 80 % calibration split, then applied to the 20 % validation split.
7. **[UNSPECIFIED]** PLSR LV search range — not stated. Reported optima range from 4 to 14, so a search up to ~25 should cover it safely.

## 6. Statistical comparison

1. Paired Wilcoxon signed-rank tests across matched splits, primarily on RMSE and RPIQ.
2. p-values adjusted with **Holm** correction.

## 7. Metrics (validation set, primary; calibration also reported)

1. **RMSE** (lower is better, units = % SOC).
2. **R²** (higher is better).
3. **MBD** = mean(predicted − observed). Positive = over-prediction.
4. **RPIQ** = IQR(observed) / RMSE. Higher is better. **[UNSPECIFIED]** IQR computed on which set — likely the validation set's observed SOC, but could be calibration or full data; must verify against reported numbers.

## 8. Target numbers to reproduce (Table 1)

Validation (external 20 %) — best preprocessing per dataset shown bold:

| Dataset | Pretreatment | Factors | RMSE % | R² | MBD | RPIQ |
|---|---|---|---|---|---|---|
| Global | None | 10 | 2.077 | 0.626 | 0.006 | 0.426 |
| Global | SNV | 14 | 1.812 | 0.715 | −0.028 | 0.488 |
| Global | MSC | 13 | 1.931 | 0.676 | −0.009 | 0.458 |
| Global | SG | 10 | 2.077 | 0.626 | 0.006 | 0.426 |
| **Global** | **SGD** | **11** | **1.559** | **0.789** | **−0.096** | **0.568** |
| China | None | 7 | 0.273 | 0.856 | −0.006 | 2.898 |
| **China** | **SNV** | **6** | **0.253** | **0.878** | **−0.024** | **3.119** |
| China | MSC | 6 | 0.257 | 0.875 | −0.004 | 3.075 |
| China | SG | 7 | 0.272 | 0.856 | −0.006 | 2.900 |
| China | SGD | 4 | 0.282 | 0.856 | −0.026 | 2.803 |
| Kenya | None | 7 | 0.733 | 0.920 | 0.154 | 2.389 |
| **Kenya** | **SNV** | **6** | **0.713** | **0.924** | **0.077** | **2.455** |
| Kenya | MSC | 6 | 0.725 | 0.919 | 0.131 | 2.413 |
| Kenya | SG | 7 | 0.733 | 0.920 | 0.154 | 2.389 |
| Kenya | SGD | 4 | 1.300 | 0.803 | 0.209 | 1.346 |
| Indonesia | None | 14 | 1.105 | 0.784 | 0.018 | 2.598 |
| Indonesia | SNV | 10 | 0.893 | 0.874 | −0.031 | 3.218 |
| **Indonesia** | **MSC** | **13** | **0.849** | **0.876** | **−0.018** | **3.384** |
| Indonesia | SG | 14 | 1.105 | 0.784 | 0.018 | 2.598 |
| Indonesia | SGD | 14 | 1.148 | 0.767 | −0.102 | 2.501 |

Calibration columns are also in the paper (Table 1) and will be tracked but are secondary.

## 9. Open items needing Arif's decision before any code

1. Random seed value to use (paper does not state).
2. Grouping key for "replicates from same sample/site": confirm against ISRIC schema (likely `profile_id` + `top` + `bottom` depths).
3. Resolve modelling-vs-descriptive n discrepancy for China / Kenya / Indonesia.
4. Single 80/20 split vs repeated outer K-fold — adopt single split (working assumption) unless supplementary clarifies.
5. SG / SGD hyperparameter selection criterion (assume same one-SE PLSR-CV pipeline unless stated otherwise).
6. RPIQ denominator IQR scope — assume validation-set observed SOC.
7. Whether ISRIC publishes per-replicate spectra or per-sample averages.

## 10. Reproduction acceptance criteria

The PLSR replication will be considered faithful when, for each (dataset × preprocessing) cell:

1. Selected LV count matches paper's "Factors" column exactly, OR within ±1 LV with RMSE within 5 % of paper's value.
2. Validation RMSE within ±5 % of paper.
3. Validation R² within ±0.02 of paper.
4. MBD within ±0.05 (% SOC) of paper.

Only after these are met do we replace PLSR with DL.
