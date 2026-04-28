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
from model_rbn_ann import RbnSocAnn
from model_rbnd_ann import RbndSocAnn


PROJECT_ROOT = Path(__file__).resolve().parent
PLSR_PER_CELL_DIR = PROJECT_ROOT / "results" / "per_cell"
H1A_FULL_BATCH_OUTPUT_PATH = PROJECT_ROOT / "results" / "h1a_full_batch_lr1e4_fixed419_results.csv"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
PREPROCESSING_NAME_FOR_H1A = "none"
ALGORITHM_NAMES_IN_REPORT_ORDER = ["plsr", "rbn_full_batch", "rbnd_full_batch"]

FULL_BATCH_EPOCHS = 419
FULL_BATCH_LEARNING_RATE = 1e-4
FULL_BATCH_WEIGHT_DECAY = 0.0


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


def train_full_batch_and_capture_final_and_best(model_class, train_spectra, train_target, test_spectra, test_target):
    reset_all_random_seeds()
    n_features = train_spectra.shape[1]
    full_batch_size = train_spectra.shape[0]

    regressor_model = model_class(n_features).to(DEVICE)
    regressor_model.train()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(), lr=FULL_BATCH_LEARNING_RATE, weight_decay=FULL_BATCH_WEIGHT_DECAY
    )
    mse_loss_function = nn.MSELoss()
    full_batch_loader = build_loader_features_and_targets(
        train_spectra, train_target, shuffle=False, batch_size=full_batch_size
    )

    best_test_rmse = float("inf")
    best_record = None
    final_record = None
    for epoch_index in range(1, FULL_BATCH_EPOCHS + 1):
        run_one_full_batch_epoch(regressor_model, optimizer, mse_loss_function, full_batch_loader)
        train_metrics, test_metrics = evaluate_train_and_test_metrics(
            regressor_model, train_spectra, train_target, test_spectra, test_target
        )
        if test_metrics["rmse"] < best_test_rmse:
            best_test_rmse = test_metrics["rmse"]
            best_record = {
                "best_epoch": epoch_index,
                "train_metrics_at_best": train_metrics,
                "test_metrics_at_best": test_metrics,
            }
        if epoch_index == FULL_BATCH_EPOCHS:
            final_record = {
                "final_epoch": epoch_index,
                "train_metrics_at_final": train_metrics,
                "test_metrics_at_final": test_metrics,
            }
    return final_record, best_record, regressor_model.count_learnable_parameters()


def read_plsr_cell_metrics_from_existing_json(dataset_name):
    plsr_json_path = PLSR_PER_CELL_DIR / f"{dataset_name}_{PREPROCESSING_NAME_FOR_H1A}.json"
    payload = json.loads(plsr_json_path.read_text())
    return payload["train_metrics"], payload["test_metrics"]


def train_dl_method_for_one_dataset(dataset_name, model_class):
    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, PREPROCESSING_NAME_FOR_H1A)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    final_record, best_record, learnable_parameter_count = train_full_batch_and_capture_final_and_best(
        model_class, train_spectra, train_target, test_spectra, test_target
    )
    return final_record, best_record, learnable_parameter_count


def build_result_row(dataset_name, algorithm_name, train_metrics_final, test_metrics_final,
                     best_epoch, best_test_rmse, learnable_parameter_count):
    return {
        "dataset": dataset_name,
        "preprocessing": PREPROCESSING_NAME_FOR_H1A,
        "algorithm": algorithm_name,
        "best_epoch_seen": best_epoch,
        "best_test_rmse_seen": best_test_rmse,
        "train_rmse": train_metrics_final["rmse"],
        "train_r2": train_metrics_final["r2"],
        "train_mbd": train_metrics_final["mbd"],
        "train_rpiq": train_metrics_final["rpiq"],
        "test_rmse": test_metrics_final["rmse"],
        "test_r2": test_metrics_final["r2"],
        "test_mbd": test_metrics_final["mbd"],
        "test_rpiq": test_metrics_final["rpiq"],
        "n_train": train_metrics_final["n"],
        "n_test": test_metrics_final["n"],
        "learnable_parameters": learnable_parameter_count,
    }


def print_one_cell_line(algorithm_name, test_metrics_final, best_epoch, best_test_rmse, runtime_seconds):
    best_text = (
        ""
        if best_epoch is None
        else f"  [best seen: RMSE={best_test_rmse:.4f} at epoch {best_epoch}]"
    )
    runtime_text = "" if runtime_seconds is None else f"  ({runtime_seconds:.1f} s)"
    print(
        f"  {algorithm_name:<18s}  final test RMSE={test_metrics_final['rmse']:.4f}  "
        f"R2={test_metrics_final['r2']:.4f}{best_text}{runtime_text}"
    )


def run_for_one_dataset(dataset_name):
    rows = []
    print(f"\n>>> {dataset_name} / {PREPROCESSING_NAME_FOR_H1A}")

    plsr_train_metrics, plsr_test_metrics = read_plsr_cell_metrics_from_existing_json(dataset_name)
    rows.append(build_result_row(
        dataset_name, "plsr", plsr_train_metrics, plsr_test_metrics,
        best_epoch=None, best_test_rmse=None, learnable_parameter_count=None,
    ))
    print_one_cell_line("plsr", plsr_test_metrics, best_epoch=None, best_test_rmse=None, runtime_seconds=None)

    cell_started_seconds = time.time()
    rbn_final, rbn_best, rbn_param_count = train_dl_method_for_one_dataset(dataset_name, RbnSocAnn)
    rows.append(build_result_row(
        dataset_name, "rbn_full_batch",
        rbn_final["train_metrics_at_final"], rbn_final["test_metrics_at_final"],
        best_epoch=rbn_best["best_epoch"],
        best_test_rmse=rbn_best["test_metrics_at_best"]["rmse"],
        learnable_parameter_count=rbn_param_count,
    ))
    print_one_cell_line(
        "rbn_full_batch", rbn_final["test_metrics_at_final"],
        best_epoch=rbn_best["best_epoch"],
        best_test_rmse=rbn_best["test_metrics_at_best"]["rmse"],
        runtime_seconds=time.time() - cell_started_seconds,
    )

    cell_started_seconds = time.time()
    rbnd_final, rbnd_best, rbnd_param_count = train_dl_method_for_one_dataset(dataset_name, RbndSocAnn)
    rows.append(build_result_row(
        dataset_name, "rbnd_full_batch",
        rbnd_final["train_metrics_at_final"], rbnd_final["test_metrics_at_final"],
        best_epoch=rbnd_best["best_epoch"],
        best_test_rmse=rbnd_best["test_metrics_at_best"]["rmse"],
        learnable_parameter_count=rbnd_param_count,
    ))
    print_one_cell_line(
        "rbnd_full_batch", rbnd_final["test_metrics_at_final"],
        best_epoch=rbnd_best["best_epoch"],
        best_test_rmse=rbnd_best["test_metrics_at_best"]["rmse"],
        runtime_seconds=time.time() - cell_started_seconds,
    )

    return rows


def print_pivot_table(result_dataframe):
    pivot_table = result_dataframe.pivot(index="algorithm", columns="dataset", values="test_rmse")
    pivot_table = pivot_table.reindex(index=ALGORITHM_NAMES_IN_REPORT_ORDER, columns=DATASET_NAMES)
    print(f"\nTest RMSE at FINAL epoch ({FULL_BATCH_EPOCHS}) (rows = algorithm, cols = dataset, lower is better):")
    print(pivot_table.to_string(float_format=lambda value: f"{value:.4f}"))


def print_per_dataset_winners(result_dataframe):
    print("\nPer-dataset winner (lowest test RMSE):")
    for dataset_name in DATASET_NAMES:
        dataset_rows = result_dataframe[result_dataframe["dataset"] == dataset_name]
        winner_row = dataset_rows.loc[dataset_rows["test_rmse"].idxmin()]
        print(f"  {dataset_name:<10s} -> {winner_row['algorithm']}  (test RMSE={winner_row['test_rmse']:.4f})")


def main():
    H1A_FULL_BATCH_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")
    print(
        f">>> Full-batch H1A  (epochs={FULL_BATCH_EPOCHS}, lr={FULL_BATCH_LEARNING_RATE}, "
        f"wd={FULL_BATCH_WEIGHT_DECAY}, dropout=0.3 for rbnd, seed=42, "
        f"reported = FINAL epoch metrics; best-seen shown for context)"
    )

    all_rows = []
    overall_started_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        all_rows.extend(run_for_one_dataset(dataset_name))
    overall_elapsed_seconds = time.time() - overall_started_seconds

    result_dataframe = pd.DataFrame(all_rows)
    result_dataframe.to_csv(H1A_FULL_BATCH_OUTPUT_PATH, index=False)
    print(f"\nWrote {H1A_FULL_BATCH_OUTPUT_PATH} ({len(result_dataframe)} rows)")
    print(f"Total wall time: {overall_elapsed_seconds:.1f} s")

    print_pivot_table(result_dataframe)
    print_per_dataset_winners(result_dataframe)


if __name__ == "__main__":
    main()
