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
SWEEP_OUTPUT_PATH = PROJECT_ROOT / "results" / "sweep_indonesia_rbnd_lr1e3_l1l2_curves.csv"

DATASET_NAME = "indonesia"
PREPROCESSING_NAME = "none"
SWEEP_EPOCHS = 3000
SWEEP_LEARNING_RATE = 1e-3
SWEEP_PRINT_EVERY_N_EPOCHS = 500

SWEEP_CONFIGURATIONS = [
    {"label": "l1=0, l2=0",         "l1_lambda": 0.0,    "l2_weight_decay": 0.0},
    {"label": "l1=1e-4, l2=1e-3",   "l1_lambda": 1e-4,   "l2_weight_decay": 1e-3},
    {"label": "l1=1e-3, l2=1e-2",   "l1_lambda": 1e-3,   "l2_weight_decay": 1e-2},
    {"label": "l1=1e-2, l2=1e-1",   "l1_lambda": 1e-2,   "l2_weight_decay": 1e-1},
    {"label": "l1=0, l2=1e-1",      "l1_lambda": 0.0,    "l2_weight_decay": 1e-1},
    {"label": "l1=1e-2, l2=0",      "l1_lambda": 1e-2,   "l2_weight_decay": 0.0},
]


def compute_l1_penalty_on_linear_weights(regressor_model):
    l1_penalty = 0.0
    for module in regressor_model.modules():
        if isinstance(module, nn.Linear):
            l1_penalty = l1_penalty + module.weight.abs().sum()
    return l1_penalty


def evaluate_train_and_test_metrics(regressor_model, train_spectra, train_target, test_spectra, test_target):
    regressor_model.eval()
    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)
    regressor_model.train()
    return train_metrics, test_metrics


def run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader, l1_lambda):
    for batch_spectra, batch_target in full_batch_loader:
        optimizer.zero_grad()
        predicted_soc = regressor_model(batch_spectra)
        mse_loss = mse_loss_function(predicted_soc, batch_target)
        if l1_lambda > 0.0:
            l1_penalty = compute_l1_penalty_on_linear_weights(regressor_model)
            total_loss = mse_loss + l1_lambda * l1_penalty
        else:
            total_loss = mse_loss
        total_loss.backward()
        optimizer.step()


def train_one_configuration(configuration, train_spectra, train_target, test_spectra, test_target):
    reset_all_random_seeds()
    n_features = train_spectra.shape[1]
    full_batch_size = train_spectra.shape[0]

    regressor_model = RbndSocAnn(n_features).to(DEVICE)
    regressor_model.train()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(),
        lr=SWEEP_LEARNING_RATE,
        weight_decay=configuration["l2_weight_decay"],
    )
    mse_loss_function = nn.MSELoss()
    full_batch_loader = build_loader_features_and_targets(
        train_spectra, train_target, shuffle=False, batch_size=full_batch_size
    )

    rows = []
    for epoch_index in range(1, SWEEP_EPOCHS + 1):
        run_one_full_batch_epoch(
            regressor_model, optimizer, mse_loss_function, full_batch_loader,
            l1_lambda=configuration["l1_lambda"],
        )
        train_metrics, test_metrics = evaluate_train_and_test_metrics(
            regressor_model, train_spectra, train_target, test_spectra, test_target
        )
        rows.append({
            "label": configuration["label"],
            "l1_lambda": configuration["l1_lambda"],
            "l2_weight_decay": configuration["l2_weight_decay"],
            "epoch": epoch_index,
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        })
    return rows


def report_best_test_per_configuration(all_rows_dataframe):
    print(f"\n{'configuration':<22}  {'best test RMSE':>14}  {'best epoch':>10}  {'final test RMSE':>15}")
    for label, group in all_rows_dataframe.groupby("label", sort=False):
        best_row = group.loc[group["test_rmse"].idxmin()]
        final_row = group.iloc[-1]
        print(
            f"{label:<22}  {best_row['test_rmse']:>14.4f}  {int(best_row['epoch']):>10d}  "
            f"{final_row['test_rmse']:>15.4f}"
        )


def main():
    SWEEP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(f">>> sweep on {DATASET_NAME} / {PREPROCESSING_NAME}  "
          f"(rbnd, dropout=0.3, FULL-batch, lr={SWEEP_LEARNING_RATE}, epochs={SWEEP_EPOCHS}, seed=42)")

    train_dataframe, test_dataframe = load_one_preprocessed_pair(DATASET_NAME, PREPROCESSING_NAME)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    print(f"    train rows={train_spectra.shape[0]}, test rows={test_spectra.shape[0]}, "
          f"n_features={train_spectra.shape[1]}")

    all_rows = []
    overall_started_seconds = time.time()
    for configuration in SWEEP_CONFIGURATIONS:
        configuration_started_seconds = time.time()
        print(f"\n  >>> {configuration['label']}")
        rows = train_one_configuration(
            configuration, train_spectra, train_target, test_spectra, test_target
        )
        all_rows.extend(rows)
        configuration_dataframe = pd.DataFrame(rows)
        best_row = configuration_dataframe.loc[configuration_dataframe["test_rmse"].idxmin()]
        final_row = configuration_dataframe.iloc[-1]
        elapsed_seconds = time.time() - configuration_started_seconds
        print(
            f"      best test RMSE={best_row['test_rmse']:.4f} at epoch {int(best_row['epoch'])}, "
            f"final test RMSE={final_row['test_rmse']:.4f}  ({elapsed_seconds:.1f} s)"
        )

    all_rows_dataframe = pd.DataFrame(all_rows)
    all_rows_dataframe.to_csv(SWEEP_OUTPUT_PATH, index=False)
    report_best_test_per_configuration(all_rows_dataframe)
    print(f"\nWrote {SWEEP_OUTPUT_PATH}  ({len(all_rows_dataframe)} rows)")
    print(f"Total wall time: {time.time() - overall_started_seconds:.1f} s")


if __name__ == "__main__":
    main()
