import copy
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold

from train_pbn_experiment import (
    DEVICE,
    load_one_preprocessed_pair,
    extract_spectra_and_target,
    build_loader_features_and_targets,
    predict_for_set,
    compute_metrics_dictionary,
    SOC_COLUMN,
)
from model_rbnd_ann import RbndSocAnn


PROJECT_ROOT = Path(__file__).resolve().parent
PROBE_CURVES_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_early_stop_lr1e3_curves.csv"
PROBE_SUMMARY_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_early_stop_lr1e3_summary.csv"

DATASET_NAME = "indonesia"
PREPROCESSING_NAME = "none"
GROUP_KEY_COLUMN = "Batch and labid"

PROBE_LEARNING_RATE = 1e-3
PROBE_MAX_EPOCHS = 3000
PROBE_PATIENCE = 20
PROBE_VALIDATION_FOLD_SIZE = 10
PROBE_STRATIFICATION_QUANTILES = 4
PROBE_SEEDS = [42, 1, 2]

PLSR_INDONESIA_TEST_RMSE = 1.1328


def reset_random_seeds_for_probe(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_into_inner_train_and_validation(train_dataframe, seed):
    soc_values = train_dataframe[SOC_COLUMN].to_numpy()
    soc_quartile_strata = pd.qcut(soc_values, q=PROBE_STRATIFICATION_QUANTILES,
                                  labels=False, duplicates="drop")
    group_keys = train_dataframe[GROUP_KEY_COLUMN].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=PROBE_VALIDATION_FOLD_SIZE, shuffle=True, random_state=seed
    )
    sample_indices = np.arange(len(train_dataframe))
    inner_train_idx, validation_idx = next(splitter.split(sample_indices, soc_quartile_strata, group_keys))
    return inner_train_idx, validation_idx


def evaluate_three_sets(regressor_model, inner_train_spectra, inner_train_target,
                        validation_spectra, validation_target, test_spectra, test_target):
    regressor_model.eval()
    inner_train_predictions = predict_for_set(regressor_model, inner_train_spectra)
    validation_predictions = predict_for_set(regressor_model, validation_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    regressor_model.train()
    return (
        compute_metrics_dictionary(inner_train_target, inner_train_predictions),
        compute_metrics_dictionary(validation_target, validation_predictions),
        compute_metrics_dictionary(test_target, test_predictions),
    )


def run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader):
    for batch_spectra, batch_target in full_batch_loader:
        optimizer.zero_grad()
        predicted_soc = regressor_model(batch_spectra)
        mse_loss = mse_loss_function(predicted_soc, batch_target)
        mse_loss.backward()
        optimizer.step()


def train_with_early_stopping_for_one_seed(seed, train_dataframe, test_spectra, test_target):
    inner_train_idx, validation_idx = split_train_into_inner_train_and_validation(train_dataframe, seed)
    inner_train_dataframe = train_dataframe.iloc[inner_train_idx].reset_index(drop=True)
    validation_dataframe = train_dataframe.iloc[validation_idx].reset_index(drop=True)

    inner_train_spectra, inner_train_target = extract_spectra_and_target(inner_train_dataframe)
    validation_spectra, validation_target = extract_spectra_and_target(validation_dataframe)

    n_features = inner_train_spectra.shape[1]
    full_batch_size = inner_train_spectra.shape[0]

    reset_random_seeds_for_probe(seed)
    regressor_model = RbndSocAnn(n_features).to(DEVICE)
    regressor_model.train()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(), lr=PROBE_LEARNING_RATE, weight_decay=0.0
    )
    mse_loss_function = nn.MSELoss()
    full_batch_loader = build_loader_features_and_targets(
        inner_train_spectra, inner_train_target, shuffle=False, batch_size=full_batch_size
    )

    best_validation_rmse = float("inf")
    best_validation_epoch = 0
    best_state_dict = None
    epochs_since_validation_improvement = 0
    stopped_epoch_index = PROBE_MAX_EPOCHS

    rows = []
    for epoch_index in range(1, PROBE_MAX_EPOCHS + 1):
        run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader)
        train_metrics, validation_metrics, test_metrics = evaluate_three_sets(
            regressor_model, inner_train_spectra, inner_train_target,
            validation_spectra, validation_target, test_spectra, test_target,
        )
        rows.append({
            "seed": seed,
            "epoch": epoch_index,
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_rmse": validation_metrics["rmse"],
            "val_r2": validation_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        })

        if validation_metrics["rmse"] < best_validation_rmse:
            best_validation_rmse = validation_metrics["rmse"]
            best_validation_epoch = epoch_index
            best_state_dict = copy.deepcopy(regressor_model.state_dict())
            epochs_since_validation_improvement = 0
        else:
            epochs_since_validation_improvement += 1

        if epochs_since_validation_improvement >= PROBE_PATIENCE:
            stopped_epoch_index = epoch_index
            break

    regressor_model.load_state_dict(best_state_dict)
    final_train_metrics, final_validation_metrics, final_test_metrics = evaluate_three_sets(
        regressor_model, inner_train_spectra, inner_train_target,
        validation_spectra, validation_target, test_spectra, test_target,
    )

    summary_record = {
        "seed": seed,
        "n_inner_train": int(inner_train_spectra.shape[0]),
        "n_validation": int(validation_spectra.shape[0]),
        "n_test": int(test_spectra.shape[0]),
        "stopped_at_epoch": stopped_epoch_index,
        "best_val_epoch": best_validation_epoch,
        "best_val_rmse": final_validation_metrics["rmse"],
        "best_val_r2": final_validation_metrics["r2"],
        "train_rmse_at_best": final_train_metrics["rmse"],
        "test_rmse_at_best": final_test_metrics["rmse"],
        "test_r2_at_best": final_test_metrics["r2"],
    }
    return rows, summary_record


def main():
    PROBE_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(f">>> early-stop probe on {DATASET_NAME} / {PREPROCESSING_NAME}  "
          f"(rbnd, dropout=0.3, FULL-batch, lr={PROBE_LEARNING_RATE}, "
          f"max_epochs={PROBE_MAX_EPOCHS}, patience={PROBE_PATIENCE}, "
          f"val={PROBE_VALIDATION_FOLD_SIZE}-fold StratifiedGroup, seeds={PROBE_SEEDS})")
    print(f"    PLSR baseline test RMSE = {PLSR_INDONESIA_TEST_RMSE:.4f}")

    train_dataframe, test_dataframe = load_one_preprocessed_pair(DATASET_NAME, PREPROCESSING_NAME)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)

    all_curve_rows = []
    all_summary_records = []
    overall_started_seconds = time.time()
    for seed in PROBE_SEEDS:
        seed_started_seconds = time.time()
        print(f"\n  >>> seed={seed}")
        curve_rows, summary_record = train_with_early_stopping_for_one_seed(
            seed, train_dataframe, test_spectra, test_target
        )
        all_curve_rows.extend(curve_rows)
        all_summary_records.append(summary_record)
        elapsed_seconds = time.time() - seed_started_seconds
        print(
            f"      n_inner_train={summary_record['n_inner_train']}, "
            f"n_validation={summary_record['n_validation']}, "
            f"n_test={summary_record['n_test']}\n"
            f"      stopped at epoch {summary_record['stopped_at_epoch']}, "
            f"best val epoch {summary_record['best_val_epoch']}, "
            f"val RMSE {summary_record['best_val_rmse']:.4f}, "
            f"test RMSE {summary_record['test_rmse_at_best']:.4f}  ({elapsed_seconds:.1f} s)"
        )

    pd.DataFrame(all_curve_rows).to_csv(PROBE_CURVES_PATH, index=False)
    summary_dataframe = pd.DataFrame(all_summary_records)
    summary_dataframe.to_csv(PROBE_SUMMARY_PATH, index=False)

    print(f"\n>>> Summary across seeds")
    print(summary_dataframe.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    test_rmse_values = summary_dataframe["test_rmse_at_best"].to_numpy()
    print(
        f"\nTest RMSE at best-val epoch -> "
        f"median={np.median(test_rmse_values):.4f}, "
        f"min={test_rmse_values.min():.4f}, "
        f"max={test_rmse_values.max():.4f}  "
        f"(PLSR={PLSR_INDONESIA_TEST_RMSE:.4f})"
    )
    print(f"\nWrote {PROBE_CURVES_PATH}")
    print(f"Wrote {PROBE_SUMMARY_PATH}")
    print(f"Total wall time: {time.time() - overall_started_seconds:.1f} s")


if __name__ == "__main__":
    main()
