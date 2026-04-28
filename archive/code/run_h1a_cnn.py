from pathlib import Path
import json
import time
import torch
from torch import nn
import pandas as pd

from train_pbn_experiment import (
    DEVICE,
    load_one_preprocessed_pair,
    extract_spectra_and_target,
    reset_all_random_seeds,
    build_loader_features_and_targets,
    predict_for_set,
    compute_metrics_dictionary,
)
from model_baseline_cnn import BaselineSocCnn
from model_rbn_cnn import RbnSocCnn


PROJECT_ROOT = Path(__file__).resolve().parent
PLSR_PER_CELL_DIR = PROJECT_ROOT / "results" / "per_cell"
H1A_CNN_OUTPUT_PATH = PROJECT_ROOT / "results" / "h1a_cnn_results.csv"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
PREPROCESSING_NAME_FOR_H1A = "none"
ALGORITHM_NAMES_IN_REPORT_ORDER = ["plsr", "baseline_cnn", "rbn_cnn"]

CNN_BATCH_SIZE = 32
CNN_EPOCHS = 200
CNN_LEARNING_RATE = 1e-3
CNN_L1_LAMBDA = 1e-4
CNN_L2_WEIGHT_DECAY = 1e-3


def compute_l1_penalty(regressor_model):
    l1_sum = torch.zeros((), device=DEVICE)
    for parameter in regressor_model.parameters():
        if parameter.requires_grad:
            l1_sum = l1_sum + parameter.abs().sum()
    return l1_sum


def train_cnn_with_l1_l2(regressor_model, train_spectra, train_target):
    regressor_model.train()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_L2_WEIGHT_DECAY
    )
    mse_loss_function = nn.MSELoss()
    train_loader = build_loader_features_and_targets(
        train_spectra, train_target, shuffle=True, batch_size=CNN_BATCH_SIZE
    )
    for _ in range(CNN_EPOCHS):
        for batch_spectra, batch_target in train_loader:
            optimizer.zero_grad()
            predicted_soc = regressor_model(batch_spectra)
            mse_loss = mse_loss_function(predicted_soc, batch_target)
            total_loss = mse_loss + CNN_L1_LAMBDA * compute_l1_penalty(regressor_model)
            total_loss.backward()
            optimizer.step()


def read_plsr_cell_metrics_from_existing_json(dataset_name):
    plsr_json_path = PLSR_PER_CELL_DIR / f"{dataset_name}_{PREPROCESSING_NAME_FOR_H1A}.json"
    payload = json.loads(plsr_json_path.read_text())
    return payload["train_metrics"], payload["test_metrics"]


def train_cnn_method_and_get_metrics(dataset_name, model_class):
    reset_all_random_seeds()
    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, PREPROCESSING_NAME_FOR_H1A)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    regressor_model = model_class(n_features).to(DEVICE)
    train_cnn_with_l1_l2(regressor_model, train_spectra, train_target)

    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)
    return train_metrics, test_metrics, regressor_model.count_learnable_parameters()


def build_result_row(dataset_name, algorithm_name, train_metrics, test_metrics, learnable_parameter_count):
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
        "learnable_parameters": learnable_parameter_count,
    }


def print_one_cell_line(algorithm_name, test_metrics, runtime_seconds):
    runtime_text = "" if runtime_seconds is None else f"  ({runtime_seconds:.1f} s)"
    print(
        f"  {algorithm_name:<13s}  test RMSE={test_metrics['rmse']:.4f}  "
        f"R2={test_metrics['r2']:.4f}{runtime_text}"
    )


def run_for_one_dataset(dataset_name):
    rows = []
    print(f"\n>>> {dataset_name} / {PREPROCESSING_NAME_FOR_H1A}")

    plsr_train_metrics, plsr_test_metrics = read_plsr_cell_metrics_from_existing_json(dataset_name)
    rows.append(build_result_row(dataset_name, "plsr", plsr_train_metrics, plsr_test_metrics, learnable_parameter_count=None))
    print_one_cell_line("plsr", plsr_test_metrics, runtime_seconds=None)

    cell_started_seconds = time.time()
    baseline_train_metrics, baseline_test_metrics, baseline_param_count = train_cnn_method_and_get_metrics(
        dataset_name, BaselineSocCnn
    )
    rows.append(build_result_row(
        dataset_name, "baseline_cnn", baseline_train_metrics, baseline_test_metrics, baseline_param_count
    ))
    print_one_cell_line("baseline_cnn", baseline_test_metrics, time.time() - cell_started_seconds)

    cell_started_seconds = time.time()
    rbn_train_metrics, rbn_test_metrics, rbn_param_count = train_cnn_method_and_get_metrics(
        dataset_name, RbnSocCnn
    )
    rows.append(build_result_row(
        dataset_name, "rbn_cnn", rbn_train_metrics, rbn_test_metrics, rbn_param_count
    ))
    print_one_cell_line("rbn_cnn", rbn_test_metrics, time.time() - cell_started_seconds)

    return rows


def print_pivot_table(result_dataframe):
    pivot_table = result_dataframe.pivot(index="algorithm", columns="dataset", values="test_rmse")
    pivot_table = pivot_table.reindex(index=ALGORITHM_NAMES_IN_REPORT_ORDER, columns=DATASET_NAMES)
    print("\nTest RMSE (rows = algorithm, cols = dataset, lower is better):")
    print(pivot_table.to_string(float_format=lambda value: f"{value:.4f}"))


def print_per_dataset_winners(result_dataframe):
    print("\nPer-dataset winner (lowest test RMSE):")
    for dataset_name in DATASET_NAMES:
        dataset_rows = result_dataframe[result_dataframe["dataset"] == dataset_name]
        winner_row = dataset_rows.loc[dataset_rows["test_rmse"].idxmin()]
        print(f"  {dataset_name:<10s} -> {winner_row['algorithm']}  (test RMSE={winner_row['test_rmse']:.4f})")


def print_parameter_counts(result_dataframe):
    cnn_rows = result_dataframe[result_dataframe["algorithm"].isin(["baseline_cnn", "rbn_cnn"])]
    parameter_counts = cnn_rows[["algorithm", "dataset", "learnable_parameters"]].drop_duplicates(
        subset=["algorithm", "dataset"]
    )
    print("\nLearnable parameters per CNN model:")
    print(parameter_counts.to_string(index=False))


def main():
    H1A_CNN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(
        f">>> 1D-CNN H1A  (bs={CNN_BATCH_SIZE}, epochs={CNN_EPOCHS}, lr={CNN_LEARNING_RATE}, "
        f"l1={CNN_L1_LAMBDA}, l2={CNN_L2_WEIGHT_DECAY}, seed=42)"
    )

    all_rows = []
    overall_started_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        all_rows.extend(run_for_one_dataset(dataset_name))
    overall_elapsed_seconds = time.time() - overall_started_seconds

    result_dataframe = pd.DataFrame(all_rows)
    result_dataframe.to_csv(H1A_CNN_OUTPUT_PATH, index=False)
    print(f"\nWrote {H1A_CNN_OUTPUT_PATH} ({len(result_dataframe)} rows)")
    print(f"Total wall time: {overall_elapsed_seconds:.1f} s")

    print_pivot_table(result_dataframe)
    print_per_dataset_winners(result_dataframe)
    print_parameter_counts(result_dataframe)


if __name__ == "__main__":
    main()
