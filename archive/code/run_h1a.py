from pathlib import Path
import json
import time
import pandas as pd

from train_pbn_experiment import (
    BATCH_SIZE,
    DEVICE,
    WEIGHT_DECAY,
    load_one_preprocessed_pair,
    extract_spectra_and_target,
    reset_all_random_seeds,
    train_supervised_regressor,
    predict_for_set,
    compute_metrics_dictionary,
)
from model_baseline_ann import BaselineSocAnn
from model_rbn_ann import RbnSocAnn


PROJECT_ROOT = Path(__file__).resolve().parent
PLSR_PER_CELL_DIR = PROJECT_ROOT / "results" / "per_cell"
H1A_OUTPUT_PATH = PROJECT_ROOT / "results" / "h1a_results.csv"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
PREPROCESSING_NAME_FOR_H1A = "none"
ALGORITHM_NAMES_IN_REPORT_ORDER = ["plsr", "baseline", "rbn"]

H1A_EPOCHS = 400
H1A_LEARNING_RATE = 1e-4


def read_plsr_cell_metrics_from_existing_json(dataset_name):
    plsr_json_path = PLSR_PER_CELL_DIR / f"{dataset_name}_{PREPROCESSING_NAME_FOR_H1A}.json"
    payload = json.loads(plsr_json_path.read_text())
    return payload["train_metrics"], payload["test_metrics"]


def train_dl_method_and_get_metrics(dataset_name, model_class):
    reset_all_random_seeds()
    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, PREPROCESSING_NAME_FOR_H1A)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    regressor_model = model_class(n_features).to(DEVICE)
    train_supervised_regressor(
        regressor_model, train_spectra, train_target, H1A_EPOCHS, BATCH_SIZE, H1A_LEARNING_RATE, WEIGHT_DECAY
    )

    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)
    return train_metrics, test_metrics


def build_h1a_row(dataset_name, algorithm_name, train_metrics, test_metrics):
    return {
        "dataset": dataset_name,
        "preprocessing": PREPROCESSING_NAME_FOR_H1A,
        "algorithm": algorithm_name,
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "train_mbd": train_metrics["mbd"],
        "train_rpiq": train_metrics["rpiq"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mbd": test_metrics["mbd"],
        "test_rpiq": test_metrics["rpiq"],
        "n_train": train_metrics["n"],
        "n_test": test_metrics["n"],
    }


def print_one_cell_line(algorithm_name, test_metrics, runtime_seconds=None):
    runtime_text = "" if runtime_seconds is None else f"  ({runtime_seconds:.1f} s)"
    print(
        f"  {algorithm_name:<8s}  test RMSE={test_metrics['rmse']:.4f}  "
        f"R2={test_metrics['r2']:.4f}{runtime_text}"
    )


def run_h1a_for_one_dataset(dataset_name):
    rows = []
    print(f"\n>>> {dataset_name} / {PREPROCESSING_NAME_FOR_H1A}")

    plsr_train_metrics, plsr_test_metrics = read_plsr_cell_metrics_from_existing_json(dataset_name)
    rows.append(build_h1a_row(dataset_name, "plsr", plsr_train_metrics, plsr_test_metrics))
    print_one_cell_line("plsr", plsr_test_metrics)

    cell_started_seconds = time.time()
    baseline_train_metrics, baseline_test_metrics = train_dl_method_and_get_metrics(
        dataset_name, BaselineSocAnn
    )
    rows.append(build_h1a_row(dataset_name, "baseline", baseline_train_metrics, baseline_test_metrics))
    print_one_cell_line("baseline", baseline_test_metrics, time.time() - cell_started_seconds)

    cell_started_seconds = time.time()
    rbn_train_metrics, rbn_test_metrics = train_dl_method_and_get_metrics(dataset_name, RbnSocAnn)
    rows.append(build_h1a_row(dataset_name, "rbn", rbn_train_metrics, rbn_test_metrics))
    print_one_cell_line("rbn", rbn_test_metrics, time.time() - cell_started_seconds)

    return rows


def print_h1a_pivot_table(h1a_dataframe):
    pivot_table = h1a_dataframe.pivot(index="algorithm", columns="dataset", values="test_rmse")
    pivot_table = pivot_table.reindex(index=ALGORITHM_NAMES_IN_REPORT_ORDER, columns=DATASET_NAMES)
    print("\nTest RMSE (rows = algorithm, cols = dataset, lower is better):")
    print(pivot_table.to_string(float_format=lambda value: f"{value:.4f}"))


def print_per_dataset_winners(h1a_dataframe):
    print("\nPer-dataset winner (lowest test RMSE):")
    for dataset_name in DATASET_NAMES:
        dataset_rows = h1a_dataframe[h1a_dataframe["dataset"] == dataset_name]
        winner_row = dataset_rows.loc[dataset_rows["test_rmse"].idxmin()]
        print(f"  {dataset_name:<10s} -> {winner_row['algorithm']}  (test RMSE={winner_row['test_rmse']:.4f})")


def main():
    H1A_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    all_rows = []
    overall_started_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        all_rows.extend(run_h1a_for_one_dataset(dataset_name))
    overall_elapsed_seconds = time.time() - overall_started_seconds

    h1a_dataframe = pd.DataFrame(all_rows)
    h1a_dataframe.to_csv(H1A_OUTPUT_PATH, index=False)
    print(f"\nWrote {H1A_OUTPUT_PATH} ({len(h1a_dataframe)} rows)")
    print(f"Total wall time: {overall_elapsed_seconds:.1f} s")

    print_h1a_pivot_table(h1a_dataframe)
    print_per_dataset_winners(h1a_dataframe)


if __name__ == "__main__":
    main()
