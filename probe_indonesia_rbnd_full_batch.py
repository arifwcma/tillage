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
PROBE_OUTPUT_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_full_batch_lr1e5_6k_curve.csv"

DATASET_NAME = "indonesia"
PREPROCESSING_NAME = "none"
PROBE_EPOCHS = 6000
PROBE_LEARNING_RATE = 1e-5
PROBE_WEIGHT_DECAY = 0.0
PROBE_PRINT_EVERY_N_EPOCHS = 100


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


def train_rbnd_full_batch_with_periodic_evaluation():
    reset_all_random_seeds()
    train_dataframe, test_dataframe = load_one_preprocessed_pair(DATASET_NAME, PREPROCESSING_NAME)
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

    metrics_per_epoch = []
    for epoch_index in range(1, PROBE_EPOCHS + 1):
        run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader)
        train_metrics, test_metrics = evaluate_train_and_test_metrics(
            regressor_model, train_spectra, train_target, test_spectra, test_target
        )
        metrics_per_epoch.append(
            {
                "epoch": epoch_index,
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
    return metrics_per_epoch, full_batch_size


def print_curve_table(metrics_per_epoch):
    print(f"{'epoch':>6}  {'train RMSE':>10}  {'train R2':>9}  {'test RMSE':>10}  {'test R2':>9}")
    for row in metrics_per_epoch:
        if row["epoch"] % PROBE_PRINT_EVERY_N_EPOCHS != 0:
            continue
        print(
            f"{row['epoch']:>6d}  {row['train_rmse']:>10.4f}  {row['train_r2']:>9.4f}  "
            f"{row['test_rmse']:>10.4f}  {row['test_r2']:>9.4f}"
        )


def report_best_test_checkpoint(metrics_per_epoch):
    best_row = min(metrics_per_epoch, key=lambda row: row["test_rmse"])
    print(
        f"\nBest test RMSE: {best_row['test_rmse']:.4f} at epoch {best_row['epoch']}  "
        f"(test R2={best_row['test_r2']:.4f}, train RMSE={best_row['train_rmse']:.4f}, "
        f"train R2={best_row['train_r2']:.4f})"
    )


def main():
    PROBE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    started_seconds = time.time()
    metrics_per_epoch, full_batch_size = train_rbnd_full_batch_with_periodic_evaluation()
    elapsed_seconds = time.time() - started_seconds

    print(
        f"\n>>> {DATASET_NAME} / {PREPROCESSING_NAME}  "
        f"(rbnd, 32 hidden, dropout=0.3, FULL-batch={full_batch_size}, "
        f"epochs={PROBE_EPOCHS}, lr={PROBE_LEARNING_RATE}, weight_decay={PROBE_WEIGHT_DECAY}, seed=42)"
    )

    pd.DataFrame(metrics_per_epoch).to_csv(PROBE_OUTPUT_PATH, index=False)
    print_curve_table(metrics_per_epoch)
    report_best_test_checkpoint(metrics_per_epoch)
    print(f"\nWrote {PROBE_OUTPUT_PATH}")
    print(f"Total wall time: {elapsed_seconds:.1f} s")


if __name__ == "__main__":
    main()
