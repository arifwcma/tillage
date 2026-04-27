"""Diagnostic re-run: refit PLSR with the paper's reported LV count for each
(dataset, method) cell, instead of the one-SE rule. SG/SGD window+polyorder
are kept at the values our one-SE picker chose during the original run
(stored in `results/per_cell/<dataset>_<method>.json`), since the paper does
not report those.

Outputs go under `results/per_cell_fixed_lv/` and a comparison CSV at
`results/table1_replication_fixed_lv.csv`. The original `results/per_cell/`
is left untouched.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

import train_plsr as base


PROJECT_ROOT = Path(__file__).resolve().parent
PER_CELL_FIXED_LV_DIR = PROJECT_ROOT / "results" / "per_cell_fixed_lv"
PER_CELL_ORIGINAL_DIR = PROJECT_ROOT / "results" / "per_cell"
COMPARISON_CSV_PATH = PROJECT_ROOT / "results" / "table1_replication_fixed_lv.csv"

PAPER_LV_BY_CELL = {
    ("global", "none"): 10,
    ("global", "snv"): 14,
    ("global", "msc"): 13,
    ("global", "sg"): 10,
    ("global", "sgd"): 11,
    ("china", "none"): 7,
    ("china", "snv"): 6,
    ("china", "msc"): 6,
    ("china", "sg"): 7,
    ("china", "sgd"): 4,
    ("kenya", "none"): 7,
    ("kenya", "snv"): 6,
    ("kenya", "msc"): 6,
    ("kenya", "sg"): 7,
    ("kenya", "sgd"): 4,
    ("indonesia", "none"): 14,
    ("indonesia", "snv"): 10,
    ("indonesia", "msc"): 13,
    ("indonesia", "sg"): 14,
    ("indonesia", "sgd"): 14,
}

PAPER_TABLE1 = {
    ("global", "none"): {"rmse": 2.077, "r2": 0.626, "mbd": 0.006, "rpiq": 0.426},
    ("global", "snv"): {"rmse": 1.812, "r2": 0.715, "mbd": -0.028, "rpiq": 0.488},
    ("global", "msc"): {"rmse": 1.931, "r2": 0.676, "mbd": -0.009, "rpiq": 0.458},
    ("global", "sg"): {"rmse": 2.077, "r2": 0.626, "mbd": 0.006, "rpiq": 0.426},
    ("global", "sgd"): {"rmse": 1.559, "r2": 0.789, "mbd": -0.096, "rpiq": 0.568},
    ("china", "none"): {"rmse": 0.273, "r2": 0.856, "mbd": -0.006, "rpiq": 2.898},
    ("china", "snv"): {"rmse": 0.253, "r2": 0.878, "mbd": -0.024, "rpiq": 3.119},
    ("china", "msc"): {"rmse": 0.257, "r2": 0.875, "mbd": -0.004, "rpiq": 3.075},
    ("china", "sg"): {"rmse": 0.272, "r2": 0.856, "mbd": -0.006, "rpiq": 2.900},
    ("china", "sgd"): {"rmse": 0.282, "r2": 0.856, "mbd": -0.026, "rpiq": 2.803},
    ("kenya", "none"): {"rmse": 0.733, "r2": 0.920, "mbd": 0.154, "rpiq": 2.389},
    ("kenya", "snv"): {"rmse": 0.713, "r2": 0.924, "mbd": 0.077, "rpiq": 2.455},
    ("kenya", "msc"): {"rmse": 0.725, "r2": 0.919, "mbd": 0.131, "rpiq": 2.413},
    ("kenya", "sg"): {"rmse": 0.733, "r2": 0.920, "mbd": 0.154, "rpiq": 2.389},
    ("kenya", "sgd"): {"rmse": 1.300, "r2": 0.803, "mbd": 0.209, "rpiq": 1.346},
    ("indonesia", "none"): {"rmse": 1.105, "r2": 0.784, "mbd": 0.018, "rpiq": 2.598},
    ("indonesia", "snv"): {"rmse": 0.893, "r2": 0.874, "mbd": -0.031, "rpiq": 3.218},
    ("indonesia", "msc"): {"rmse": 0.849, "r2": 0.876, "mbd": -0.018, "rpiq": 3.384},
    ("indonesia", "sg"): {"rmse": 1.105, "r2": 0.784, "mbd": 0.018, "rpiq": 2.598},
    ("indonesia", "sgd"): {"rmse": 1.148, "r2": 0.767, "mbd": -0.102, "rpiq": 2.501},
}

DATASET_NAMES = ["china", "kenya", "indonesia", "global"]
METHOD_NAMES = ["none", "snv", "msc", "sg", "sgd"]


def load_preprocessing_specification_for_cell(dataset_name, method_name):
    """For SG/SGD reuse the (window, polyorder) chosen by the original one-SE
    run — paper does not specify these. For others, reconstruct the trivial
    spec used by base.transform_with_preprocessing_specification."""
    if method_name in ("none", "snv", "msc"):
        return {"label": method_name}
    original_path = PER_CELL_ORIGINAL_DIR / f"{dataset_name}_{method_name}.json"
    record = json.loads(original_path.read_text())
    return record["winner"]["preprocessing_specification"]


def run_one_cell_with_paper_lv(dataset_name, method_name):
    print(f"\n>>> {dataset_name} / {method_name}")
    cell_start_seconds = time.time()
    paper_lv_count = PAPER_LV_BY_CELL[(dataset_name, method_name)]
    preprocessing_specification = load_preprocessing_specification_for_cell(
        dataset_name, method_name
    )

    train_dataframe, test_dataframe = base.load_raw_train_and_test(dataset_name)
    _, spectra_columns = base.split_columns_into_reference_and_spectra(train_dataframe)
    train_spectra_raw = train_dataframe[spectra_columns].to_numpy(dtype=np.float64)
    test_spectra_raw = test_dataframe[spectra_columns].to_numpy(dtype=np.float64)
    train_target = train_dataframe[base.SOC_COLUMN].to_numpy(dtype=np.float64)
    test_target = test_dataframe[base.SOC_COLUMN].to_numpy(dtype=np.float64)

    train_transformed, test_transformed = base.transform_with_preprocessing_specification(
        preprocessing_specification, train_spectra_raw, test_spectra_raw
    )
    fitted_model, train_spectra_mean, train_target_mean = base.fit_plsr_with_single_lv(
        train_transformed, train_target, paper_lv_count
    )
    train_predictions = base.predict_with_fitted_plsr(
        fitted_model, train_transformed, train_spectra_mean, train_target_mean
    )
    test_predictions = base.predict_with_fitted_plsr(
        fitted_model, test_transformed, train_spectra_mean, train_target_mean
    )
    train_metrics = base.compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = base.compute_metrics_dictionary(test_target, test_predictions)

    print(
        f"  paper LV={paper_lv_count}  preproc={preprocessing_specification}  "
        f"test R2={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}"
    )

    cell_runtime_seconds = time.time() - cell_start_seconds
    output_record = {
        "dataset": dataset_name,
        "method": method_name,
        "lv_count_used": int(paper_lv_count),
        "preprocessing_specification": preprocessing_specification,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": float(cell_runtime_seconds),
        "note": "LV count fixed to paper's reported value; SG/SGD window+polyorder reused from original one-SE pick.",
    }
    output_path = PER_CELL_FIXED_LV_DIR / f"{dataset_name}_{method_name}.json"
    output_path.write_text(json.dumps(output_record, indent=2))
    return output_record


def load_original_one_se_metrics(dataset_name, method_name):
    original_path = PER_CELL_ORIGINAL_DIR / f"{dataset_name}_{method_name}.json"
    record = json.loads(original_path.read_text())
    return {
        "lv_count_one_se": int(record["winner"]["lv_count"]),
        "rmse_one_se": float(record["test_metrics"]["rmse"]),
        "r2_one_se": float(record["test_metrics"]["r2"]),
        "mbd_one_se": float(record["test_metrics"]["mbd"]),
        "rpiq_one_se": float(record["test_metrics"]["rpiq"]),
    }


def write_comparison_csv(fixed_lv_records):
    rows = []
    for dataset_name in DATASET_NAMES:
        for method_name in METHOD_NAMES:
            paper_metrics = PAPER_TABLE1[(dataset_name, method_name)]
            paper_lv = PAPER_LV_BY_CELL[(dataset_name, method_name)]
            fixed_record = fixed_lv_records[(dataset_name, method_name)]
            original_metrics = load_original_one_se_metrics(dataset_name, method_name)
            row = {
                "dataset": dataset_name,
                "method": method_name,
                "lv_paper": paper_lv,
                "lv_one_se_ours": original_metrics["lv_count_one_se"],
                "r2_paper": paper_metrics["r2"],
                "r2_ours_at_paper_lv": fixed_record["test_metrics"]["r2"],
                "r2_ours_one_se": original_metrics["r2_one_se"],
                "r2_diff_paper_vs_paper_lv": fixed_record["test_metrics"]["r2"] - paper_metrics["r2"],
                "rmse_paper": paper_metrics["rmse"],
                "rmse_ours_at_paper_lv": fixed_record["test_metrics"]["rmse"],
                "rmse_ours_one_se": original_metrics["rmse_one_se"],
                "mbd_paper": paper_metrics["mbd"],
                "mbd_ours_at_paper_lv": fixed_record["test_metrics"]["mbd"],
                "mbd_ours_one_se": original_metrics["mbd_one_se"],
                "rpiq_paper": paper_metrics["rpiq"],
                "rpiq_ours_at_paper_lv": fixed_record["test_metrics"]["rpiq"],
                "rpiq_ours_one_se": original_metrics["rpiq_one_se"],
            }
            rows.append(row)
    comparison_dataframe = pd.DataFrame(rows)
    comparison_dataframe.to_csv(COMPARISON_CSV_PATH, index=False)
    return comparison_dataframe


def main():
    PER_CELL_FIXED_LV_DIR.mkdir(parents=True, exist_ok=True)
    fixed_lv_records = {}
    for dataset_name in DATASET_NAMES:
        for method_name in METHOD_NAMES:
            fixed_lv_records[(dataset_name, method_name)] = run_one_cell_with_paper_lv(
                dataset_name, method_name
            )
    comparison_dataframe = write_comparison_csv(fixed_lv_records)
    print(f"\nWrote comparison CSV: {COMPARISON_CSV_PATH}")
    print("\nR² side-by-side (paper vs ours @ paper's LV vs ours @ one-SE):")
    print(
        comparison_dataframe[
            ["dataset", "method", "lv_paper", "r2_paper", "r2_ours_at_paper_lv", "r2_ours_one_se", "r2_diff_paper_vs_paper_lv"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
