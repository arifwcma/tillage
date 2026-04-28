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
from model_rbnd_ann import RbndSocAnn


PROJECT_ROOT = Path(__file__).resolve().parent
PROBE_OUTPUT_PATH = PROJECT_ROOT / "results" / "probe_full_batch_rbnd_curves_all_regions.csv"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
PREPROCESSING_NAME = "none"
PROBE_EPOCHS = 1000
PROBE_LEARNING_RATE = 1e-3
PROBE_WEIGHT_DECAY = 0.0


def evaluate_train_and_test_metrics(regressor_model, train_spectra, train_target, test_spectra, test_target):
    regressor_model.eval()
    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)
    regressor_model.train()
    return train_metrics, test_metrics


def run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader):
    for batch_spectra, batch_target in full_batch_loader:
        optimizer.zero_grad()
        predicted_soc = regressor_model(batch_spectra)
        mse_loss = mse_loss_function(predicted_soc, batch_target)
        mse_loss.backward()
        optimizer.step()


def collect_per_epoch_curve_for_one_dataset(dataset_name):
    reset_all_random_seeds()
    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, PREPROCESSING_NAME)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]
    full_batch_size = train_spectra.shape[0]

    regressor_model = RbndSocAnn(n_features).to(DEVICE)
    regressor_model.train()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(), lr=PROBE_LEARNING_RATE, weight_decay=PROBE_WEIGHT_DECAY
    )
    mse_loss_function = nn.MSELoss()
    full_batch_loader = build_loader_features_and_targets(
        train_spectra, train_target, shuffle=False, batch_size=full_batch_size
    )

    rows = []
    for epoch_index in range(1, PROBE_EPOCHS + 1):
        run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader)
        train_metrics, test_metrics = evaluate_train_and_test_metrics(
            regressor_model, train_spectra, train_target, test_spectra, test_target
        )
        rows.append(
            {
                "dataset": dataset_name,
                "epoch": epoch_index,
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
    return rows


def main():
    PROBE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(
        f">>> Full-batch RBND per-epoch curves on 4 regions  "
        f"(epochs={PROBE_EPOCHS}, lr={PROBE_LEARNING_RATE}, dropout=0.3, seed=42)"
    )

    all_rows = []
    overall_started_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        cell_started_seconds = time.time()
        rows = collect_per_epoch_curve_for_one_dataset(dataset_name)
        all_rows.extend(rows)
        elapsed_seconds = time.time() - cell_started_seconds
        best_row = min(rows, key=lambda row: row["test_rmse"])
        print(
            f"  {dataset_name:<10s}  best test RMSE={best_row['test_rmse']:.4f} "
            f"at epoch {best_row['epoch']}  ({elapsed_seconds:.1f} s)"
        )
    overall_elapsed_seconds = time.time() - overall_started_seconds

    pd.DataFrame(all_rows).to_csv(PROBE_OUTPUT_PATH, index=False)
    print(f"\nWrote {PROBE_OUTPUT_PATH} ({len(all_rows)} rows)")
    print(f"Total wall time: {overall_elapsed_seconds:.1f} s")


if __name__ == "__main__":
    main()
