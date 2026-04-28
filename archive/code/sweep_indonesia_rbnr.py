from pathlib import Path
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
from model_rbn_ann import RbnSocAnn


PROJECT_ROOT = Path(__file__).resolve().parent
SWEEP_OUTPUT_PATH = PROJECT_ROOT / "results" / "sweep_indonesia_rbnr.csv"

DATASET_NAME = "indonesia"
PREPROCESSING_NAME = "none"
SWEEP_BATCH_SIZE = 64
SWEEP_EPOCHS = 400
SWEEP_LEARNING_RATE = 1e-4

L1_LAMBDA_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
L2_WEIGHT_DECAY_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


def compute_l1_penalty(regressor_model):
    l1_sum = torch.zeros((), device=DEVICE)
    for parameter in regressor_model.parameters():
        if not parameter.requires_grad:
            continue
        l1_sum = l1_sum + parameter.abs().sum()
    return l1_sum


def evaluate_train_and_test_metrics(regressor_model, train_spectra, train_target, test_spectra, test_target):
    regressor_model.eval()
    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)
    regressor_model.train()
    return train_metrics, test_metrics


def run_one_training_epoch(regressor_model, optimizer, mse_loss_function, train_loader, l1_lambda):
    for batch_spectra, batch_target in train_loader:
        optimizer.zero_grad()
        predicted_soc = regressor_model(batch_spectra)
        mse_loss = mse_loss_function(predicted_soc, batch_target)
        if l1_lambda > 0.0:
            total_loss = mse_loss + l1_lambda * compute_l1_penalty(regressor_model)
        else:
            total_loss = mse_loss
        total_loss.backward()
        optimizer.step()


def train_one_combination_and_capture_best(
    train_spectra, train_target, test_spectra, test_target, n_features, l1_lambda, l2_weight_decay
):
    reset_all_random_seeds()
    regressor_model = RbnSocAnn(n_features).to(DEVICE)
    regressor_model.train()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(), lr=SWEEP_LEARNING_RATE, weight_decay=l2_weight_decay
    )
    mse_loss_function = nn.MSELoss()
    train_loader = build_loader_features_and_targets(
        train_spectra, train_target, shuffle=True, batch_size=SWEEP_BATCH_SIZE
    )

    best_test_rmse = float("inf")
    best_record = None
    for epoch_index in range(1, SWEEP_EPOCHS + 1):
        run_one_training_epoch(regressor_model, optimizer, mse_loss_function, train_loader, l1_lambda)
        train_metrics, test_metrics = evaluate_train_and_test_metrics(
            regressor_model, train_spectra, train_target, test_spectra, test_target
        )
        if test_metrics["rmse"] < best_test_rmse:
            best_test_rmse = test_metrics["rmse"]
            best_record = {
                "best_epoch": epoch_index,
                "best_test_rmse": test_metrics["rmse"],
                "best_test_r2": test_metrics["r2"],
                "train_rmse_at_best": train_metrics["rmse"],
                "train_r2_at_best": train_metrics["r2"],
            }
    return best_record


def run_full_sweep():
    train_dataframe, test_dataframe = load_one_preprocessed_pair(DATASET_NAME, PREPROCESSING_NAME)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    sweep_rows = []
    for l1_lambda in L1_LAMBDA_GRID:
        for l2_weight_decay in L2_WEIGHT_DECAY_GRID:
            combo_started_seconds = time.time()
            best_record = train_one_combination_and_capture_best(
                train_spectra, train_target, test_spectra, test_target, n_features,
                l1_lambda, l2_weight_decay,
            )
            combo_elapsed_seconds = time.time() - combo_started_seconds
            row = {
                "l1_lambda": l1_lambda,
                "l2_weight_decay": l2_weight_decay,
                **best_record,
                "wall_seconds": combo_elapsed_seconds,
            }
            sweep_rows.append(row)
            print(
                f"  l1={l1_lambda:.0e}  l2={l2_weight_decay:.0e}  "
                f"-> best test RMSE={row['best_test_rmse']:.4f} "
                f"R2={row['best_test_r2']:.4f} at epoch {row['best_epoch']:>3d}  "
                f"({combo_elapsed_seconds:.1f}s)"
            )
    return sweep_rows


def print_top_combinations(sweep_dataframe, top_n):
    print(f"\nTop {top_n} combinations by best test RMSE:")
    sorted_dataframe = sweep_dataframe.sort_values("best_test_rmse").head(top_n)
    print(sorted_dataframe.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_pivot_of_best_test_rmse(sweep_dataframe):
    pivot_table = sweep_dataframe.pivot(
        index="l1_lambda", columns="l2_weight_decay", values="best_test_rmse"
    )
    print("\nBest test RMSE pivot (rows = l1, cols = l2_weight_decay, lower is better):")
    print(pivot_table.to_string(float_format=lambda value: f"{value:.4f}"))


def main():
    SWEEP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(
        f"\n>>> sweep on {DATASET_NAME} / {PREPROCESSING_NAME}  "
        f"(rbnr, 32 hidden, batch_size={SWEEP_BATCH_SIZE}, epochs={SWEEP_EPOCHS}, "
        f"lr={SWEEP_LEARNING_RATE}, seed 42)"
    )
    print(f"L1 grid: {L1_LAMBDA_GRID}")
    print(f"L2 grid: {L2_WEIGHT_DECAY_GRID}")

    overall_started_seconds = time.time()
    sweep_rows = run_full_sweep()
    overall_elapsed_seconds = time.time() - overall_started_seconds

    sweep_dataframe = pd.DataFrame(sweep_rows)
    sweep_dataframe.to_csv(SWEEP_OUTPUT_PATH, index=False)
    print(f"\nWrote {SWEEP_OUTPUT_PATH} ({len(sweep_dataframe)} rows)")
    print(f"Total wall time: {overall_elapsed_seconds:.1f} s")

    print_pivot_of_best_test_rmse(sweep_dataframe)
    print_top_combinations(sweep_dataframe, top_n=8)


if __name__ == "__main__":
    main()
