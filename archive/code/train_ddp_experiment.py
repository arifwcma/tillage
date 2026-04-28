from pathlib import Path
import json
import re
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

from model_mlp import MlpSocAnn
from model_ddp import DdpPreprocessor, DdpPlusMlp


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results" / "ddp_experiment"
CELLS_DIR = RESULTS_DIR / "cells"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DATASET_NAMES = ["indonesia", "kenya", "china", "global"]
PREPROCESSING_NAMES = ["none", "snv", "msc", "sg", "sgd", "minmax", "ddp"]
LEARNED_PREPROCESSING_NAMES = {"ddp"}

LEARNED_PREPROCESSING_SPECIFICATIONS = {
    "ddp": {
        "factory": DdpPreprocessor,
        "input_source": "minmax",
        "pipeline_description": "input -> minmax_per_feature -> learnable_BatchNorm1d",
    },
}

SOC_COLUMN = "Org C"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m\d+(?:\.\d+)?$")

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-3
SUPERVISED_EPOCHS = 500
WEIGHT_DECAY = 0.0


def reset_all_random_seeds():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def split_columns_into_reference_and_spectra(dataframe):
    spectra_columns = [c for c in dataframe.columns if WAVENUMBER_COLUMN_PATTERN.match(c) is not None]
    reference_columns = [c for c in dataframe.columns if c not in spectra_columns]
    return reference_columns, spectra_columns


def extract_spectra_and_target(dataframe):
    _, spectra_columns = split_columns_into_reference_and_spectra(dataframe)
    spectra_matrix = dataframe[spectra_columns].to_numpy(dtype=np.float32)
    target_vector = dataframe[SOC_COLUMN].to_numpy(dtype=np.float32)
    return spectra_matrix, target_vector


def load_train_test_for(dataset_name, preprocessing_name):
    train_dataframe = pd.read_csv(
        PREPROCESSED_DIR / f"{dataset_name}_{preprocessing_name}_train.csv",
        low_memory=False,
    )
    test_dataframe = pd.read_csv(
        PREPROCESSED_DIR / f"{dataset_name}_{preprocessing_name}_test.csv",
        low_memory=False,
    )
    return train_dataframe, test_dataframe


def to_device_tensor(numpy_array):
    return torch.from_numpy(numpy_array).to(DEVICE)


def fit_robust_scaler_on_train_target(train_target):
    median_value = float(np.median(train_target))
    quartile_one = float(np.percentile(train_target, 25))
    quartile_three = float(np.percentile(train_target, 75))
    interquartile_range = quartile_three - quartile_one
    if interquartile_range == 0:
        interquartile_range = 1.0
    return {
        "median": median_value,
        "iqr": float(interquartile_range),
    }


def apply_robust_scaling(target_vector, robust_scaler):
    median_value = robust_scaler["median"]
    interquartile_range = robust_scaler["iqr"]
    scaled = (target_vector - median_value) / interquartile_range
    return scaled.astype(np.float32)


def invert_robust_scaling(scaled_predictions, robust_scaler):
    median_value = robust_scaler["median"]
    interquartile_range = robust_scaler["iqr"]
    return scaled_predictions * interquartile_range + median_value


def train_model_full_batch(model, train_spectra, train_target, n_epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    soc_loss = nn.MSELoss()
    train_spectra_tensor = to_device_tensor(train_spectra)
    train_target_tensor = to_device_tensor(train_target)

    for epoch_index in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        predicted_soc = model(train_spectra_tensor)
        loss = soc_loss(predicted_soc, train_target_tensor)
        loss.backward()
        optimizer.step()


def predict_with_model(model, spectra_matrix):
    model.eval()
    with torch.no_grad():
        predictions = model(to_device_tensor(spectra_matrix)).cpu().numpy()
    return predictions


def compute_rmse(observed, predicted):
    residual = observed.ravel() - predicted.ravel()
    return float(np.sqrt((residual ** 2).mean()))


def compute_r_squared(observed, predicted):
    observed_flat = observed.ravel()
    predicted_flat = predicted.ravel()
    sum_squared_residuals = float(((observed_flat - predicted_flat) ** 2).sum())
    sum_squared_total = float(((observed_flat - observed_flat.mean()) ** 2).sum())
    if sum_squared_total == 0:
        return float("nan")
    return 1.0 - sum_squared_residuals / sum_squared_total


def compute_mean_bias_deviation(observed, predicted):
    return float((predicted.ravel() - observed.ravel()).mean())


def compute_rpiq(observed, predicted):
    rmse_value = compute_rmse(observed, predicted)
    if rmse_value == 0:
        return float("inf")
    interquartile_range = float(
        np.percentile(observed.ravel(), 75) - np.percentile(observed.ravel(), 25)
    )
    return interquartile_range / rmse_value


def compute_metrics_dictionary(observed, predicted):
    return {
        "rmse": compute_rmse(observed, predicted),
        "r2": compute_r_squared(observed, predicted),
        "mbd": compute_mean_bias_deviation(observed, predicted),
        "rpiq": compute_rpiq(observed, predicted),
        "n": int(len(observed)),
    }


def cell_json_path_for(dataset_name, preprocessing_name):
    return CELLS_DIR / f"{dataset_name}_{preprocessing_name}_mlp.json"


def cell_predictions_path_for(dataset_name, preprocessing_name):
    return PREDICTIONS_DIR / f"{dataset_name}_{preprocessing_name}_mlp.csv"


def configuration_block(method_label):
    return {
        "random_seed": RANDOM_SEED,
        "training_mode": "full_batch",
        "learning_rate": LEARNING_RATE,
        "supervised_epochs": SUPERVISED_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "dropout_probability": 0.3,
        "regression_head": "Linear(n_features,32) -> ReLU -> Dropout(0.3) -> Linear(32,1)",
        "target_scaling": "robust_scaler_per_region (fit on train OC: median, IQR)",
        "method": method_label,
    }


def save_classical_cell(dataset_name, preprocessing_name, train_metrics, test_metrics, runtime_seconds):
    payload = {
        "dataset": dataset_name,
        "preprocessing": preprocessing_name,
        "method": "mlp",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": float(runtime_seconds),
        "configuration": configuration_block(method_label="mlp_on_classical_preprocessing"),
    }
    cell_json_path_for(dataset_name, preprocessing_name).write_text(json.dumps(payload, indent=2))


def save_learned_preprocessor_cell(
    dataset_name,
    learned_preprocessor_label,
    pipeline_description,
    preprocessor_parameter_count,
    stage1_train_metrics,
    stage1_test_metrics,
    stage2_train_metrics,
    stage2_test_metrics,
    runtime_seconds,
):
    payload = {
        "dataset": dataset_name,
        "preprocessing": learned_preprocessor_label,
        "method": "mlp",
        "stage1_train_metrics": stage1_train_metrics,
        "stage1_test_metrics": stage1_test_metrics,
        "train_metrics": stage2_train_metrics,
        "test_metrics": stage2_test_metrics,
        "runtime_seconds": float(runtime_seconds),
        "configuration": {
            **configuration_block(method_label=f"mlp_on_{learned_preprocessor_label}_preprocessed"),
            "ddp_input_pipeline": pipeline_description,
            "preprocessor_learnable_parameters": int(preprocessor_parameter_count),
        },
    }
    cell_json_path_for(dataset_name, learned_preprocessor_label).write_text(json.dumps(payload, indent=2))


def save_predictions_csv(dataset_name, preprocessing_name, train_dataframe, train_predictions, test_dataframe, test_predictions):
    reference_columns, _ = split_columns_into_reference_and_spectra(train_dataframe)

    train_output = train_dataframe[reference_columns].copy()
    train_output["observed"] = train_dataframe[SOC_COLUMN].to_numpy()
    train_output["predicted"] = train_predictions.ravel()
    train_output["fold"] = "train"

    test_output = test_dataframe[reference_columns].copy()
    test_output["observed"] = test_dataframe[SOC_COLUMN].to_numpy()
    test_output["predicted"] = test_predictions.ravel()
    test_output["fold"] = "test"

    combined = pd.concat([train_output, test_output], axis=0, ignore_index=True)
    combined.to_csv(cell_predictions_path_for(dataset_name, preprocessing_name), index=False)


def run_classical_preprocessing_cell(dataset_name, preprocessing_name):
    cell_json_path = cell_json_path_for(dataset_name, preprocessing_name)
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name}")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_train_test_for(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    target_robust_scaler = fit_robust_scaler_on_train_target(train_target)
    scaled_train_target = apply_robust_scaling(train_target, target_robust_scaler)

    mlp_model = MlpSocAnn(n_features).to(DEVICE)
    train_model_full_batch(mlp_model, train_spectra, scaled_train_target, SUPERVISED_EPOCHS)

    scaled_train_predictions = predict_with_model(mlp_model, train_spectra)
    scaled_test_predictions = predict_with_model(mlp_model, test_spectra)
    train_predictions = invert_robust_scaling(scaled_train_predictions, target_robust_scaler)
    test_predictions = invert_robust_scaling(scaled_test_predictions, target_robust_scaler)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_classical_cell(dataset_name, preprocessing_name, train_metrics, test_metrics, runtime_seconds)
    save_predictions_csv(
        dataset_name, preprocessing_name, train_dataframe, train_predictions, test_dataframe, test_predictions
    )

    print(
        f"  {preprocessing_name:7s}: train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def run_learned_preprocessor_cell(dataset_name, learned_preprocessor_label):
    cell_json_path = cell_json_path_for(dataset_name, learned_preprocessor_label)
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {learned_preprocessor_label}")
        return

    specification = LEARNED_PREPROCESSING_SPECIFICATIONS[learned_preprocessor_label]
    preprocessor_factory = specification["factory"]
    input_source = specification["input_source"]
    pipeline_description = specification["pipeline_description"]

    cell_start_seconds = time.time()

    train_dataframe, test_dataframe = load_train_test_for(dataset_name, input_source)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    target_robust_scaler = fit_robust_scaler_on_train_target(train_target)
    scaled_train_target = apply_robust_scaling(train_target, target_robust_scaler)

    reset_all_random_seeds()
    stage1_preprocessor = preprocessor_factory(n_features).to(DEVICE)
    stage1_head = MlpSocAnn(n_features).to(DEVICE)
    stage1_joint_model = DdpPlusMlp(stage1_preprocessor, stage1_head).to(DEVICE)
    train_model_full_batch(stage1_joint_model, train_spectra, scaled_train_target, SUPERVISED_EPOCHS)

    scaled_stage1_train_predictions = predict_with_model(stage1_joint_model, train_spectra)
    scaled_stage1_test_predictions = predict_with_model(stage1_joint_model, test_spectra)
    stage1_train_predictions = invert_robust_scaling(scaled_stage1_train_predictions, target_robust_scaler)
    stage1_test_predictions = invert_robust_scaling(scaled_stage1_test_predictions, target_robust_scaler)
    stage1_train_metrics = compute_metrics_dictionary(train_target, stage1_train_predictions)
    stage1_test_metrics = compute_metrics_dictionary(test_target, stage1_test_predictions)

    stage1_preprocessor.eval()
    transformed_train_spectra = stage1_preprocessor.transform_with_frozen_running_statistics(train_spectra)
    transformed_test_spectra = stage1_preprocessor.transform_with_frozen_running_statistics(test_spectra)

    reset_all_random_seeds()
    stage2_head = MlpSocAnn(n_features).to(DEVICE)
    train_model_full_batch(stage2_head, transformed_train_spectra, scaled_train_target, SUPERVISED_EPOCHS)

    scaled_stage2_train_predictions = predict_with_model(stage2_head, transformed_train_spectra)
    scaled_stage2_test_predictions = predict_with_model(stage2_head, transformed_test_spectra)
    stage2_train_predictions = invert_robust_scaling(scaled_stage2_train_predictions, target_robust_scaler)
    stage2_test_predictions = invert_robust_scaling(scaled_stage2_test_predictions, target_robust_scaler)
    stage2_train_metrics = compute_metrics_dictionary(train_target, stage2_train_predictions)
    stage2_test_metrics = compute_metrics_dictionary(test_target, stage2_test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_learned_preprocessor_cell(
        dataset_name,
        learned_preprocessor_label,
        pipeline_description,
        stage1_preprocessor.count_learnable_parameters(),
        stage1_train_metrics,
        stage1_test_metrics,
        stage2_train_metrics,
        stage2_test_metrics,
        runtime_seconds,
    )
    save_predictions_csv(
        dataset_name, learned_preprocessor_label,
        train_dataframe, stage2_train_predictions, test_dataframe, stage2_test_predictions
    )

    print(
        f"  {learned_preprocessor_label:7s}: stage1 test RMSE={stage1_test_metrics['rmse']:.4f}  "
        f"stage2 test RMSE={stage2_test_metrics['rmse']:.4f}  "
        f"stage2 R2={stage2_test_metrics['r2']:.4f}  ({runtime_seconds:.1f} s)"
    )


def aggregate_all_cells_into_summary_csv():
    rows = []
    for dataset_name in DATASET_NAMES:
        for preprocessing_name in PREPROCESSING_NAMES:
            cell_json_path = cell_json_path_for(dataset_name, preprocessing_name)
            if not cell_json_path.exists():
                continue
            payload = json.loads(cell_json_path.read_text())
            row = {
                "dataset": payload["dataset"],
                "preprocessing": payload["preprocessing"],
                "method": payload["method"],
                "train_rmse": payload["train_metrics"]["rmse"],
                "train_r2": payload["train_metrics"]["r2"],
                "train_mbd": payload["train_metrics"]["mbd"],
                "train_rpiq": payload["train_metrics"]["rpiq"],
                "test_rmse": payload["test_metrics"]["rmse"],
                "test_r2": payload["test_metrics"]["r2"],
                "test_mbd": payload["test_metrics"]["mbd"],
                "test_rpiq": payload["test_metrics"]["rpiq"],
                "n_train": payload["train_metrics"]["n"],
                "n_test": payload["test_metrics"]["n"],
                "runtime_seconds": payload["runtime_seconds"],
            }
            if preprocessing_name in LEARNED_PREPROCESSING_NAMES:
                row["stage1_test_rmse"] = payload["stage1_test_metrics"]["rmse"]
                row["stage1_test_r2"] = payload["stage1_test_metrics"]["r2"]
                row["stage1_test_mbd"] = payload["stage1_test_metrics"]["mbd"]
                row["stage1_test_rpiq"] = payload["stage1_test_metrics"]["rpiq"]
                row["preprocessor_learnable_parameters"] = payload["configuration"].get(
                    "preprocessor_learnable_parameters"
                )
            rows.append(row)
    summary_dataframe = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "cell_results.csv"
    summary_dataframe.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path} ({len(rows)} cells)")
    return summary_path


def main():
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}  epochs: {SUPERVISED_EPOCHS}  lr: {LEARNING_RATE}  seed: {RANDOM_SEED}")

    overall_start_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        print(f"\n>>> {dataset_name}")
        for preprocessing_name in PREPROCESSING_NAMES:
            if preprocessing_name in LEARNED_PREPROCESSING_NAMES:
                run_learned_preprocessor_cell(dataset_name, preprocessing_name)
            else:
                run_classical_preprocessing_cell(dataset_name, preprocessing_name)

    aggregate_all_cells_into_summary_csv()
    overall_elapsed_seconds = time.time() - overall_start_seconds
    print(f"\nAll cells done. Total wall time: {overall_elapsed_seconds:.1f} s")


if __name__ == "__main__":
    main()
