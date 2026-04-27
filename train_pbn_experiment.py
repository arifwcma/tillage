from pathlib import Path
import json
import re
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import PLSRegression

from model_baseline_ann import BaselineSocAnn
from model_pbn_ann import LearnedPreprocessingAutoencoder, PbnSocAnn
from model_rbn_ann import RbnSocAnn
from model_p2bn_ann import P2bnSocAnn


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results" / "pbn_experiment"
CELLS_DIR = RESULTS_DIR / "cells"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
PREPROCESSING_NAMES = ["none", "snv", "msc", "sg", "sgd", "minmax"]
METHOD_NAMES = ["baseline", "pbn", "plsr_pbn", "rbn", "r2bn", "p2bn"]

SOC_COLUMN = "Org C"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m\d+(?:\.\d+)?$")

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SUPERVISED_EPOCHS = 200
AUTOENCODER_PRETRAIN_EPOCHS = 100
DOUBLE_SUPERVISED_EPOCHS = 400
PLSR_PBN_LV_COUNT = 15
P2BN_SOC_LOSS_WEIGHT = 0.8
P2BN_RECONSTRUCTION_LOSS_WEIGHT = 0.2


def load_one_preprocessed_pair(dataset_name, preprocessing_name):
    train_dataframe = pd.read_csv(
        PREPROCESSED_DIR / f"{dataset_name}_{preprocessing_name}_train.csv", low_memory=False
    )
    test_dataframe = pd.read_csv(
        PREPROCESSED_DIR / f"{dataset_name}_{preprocessing_name}_test.csv", low_memory=False
    )
    return train_dataframe, test_dataframe


def split_columns_into_reference_and_spectra(dataframe):
    spectra_columns = [c for c in dataframe.columns if WAVENUMBER_COLUMN_PATTERN.match(c) is not None]
    reference_columns = [c for c in dataframe.columns if c not in spectra_columns]
    return reference_columns, spectra_columns


def extract_spectra_and_target(dataframe):
    _, spectra_columns = split_columns_into_reference_and_spectra(dataframe)
    spectra_matrix = dataframe[spectra_columns].to_numpy(dtype=np.float32)
    target_vector = dataframe[SOC_COLUMN].to_numpy(dtype=np.float32)
    return spectra_matrix, target_vector


def to_device_tensor(numpy_array):
    return torch.from_numpy(numpy_array).to(DEVICE)


def reset_all_random_seeds():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def build_loader_features_only(spectra_matrix, shuffle):
    dataset = TensorDataset(to_device_tensor(spectra_matrix))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def build_loader_features_and_targets(spectra_matrix, target_vector, shuffle):
    dataset = TensorDataset(to_device_tensor(spectra_matrix), to_device_tensor(target_vector))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def pretrain_autoencoder(autoencoder_model, train_spectra):
    autoencoder_model.train()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE)
    reconstruction_loss = nn.MSELoss()
    train_loader = build_loader_features_only(train_spectra, shuffle=True)

    for epoch_index in range(AUTOENCODER_PRETRAIN_EPOCHS):
        for (batch_spectra,) in train_loader:
            optimizer.zero_grad()
            reconstructed_spectra = autoencoder_model(batch_spectra)
            loss = reconstruction_loss(reconstructed_spectra, batch_spectra)
            loss.backward()
            optimizer.step()


def train_supervised_regressor(regressor_model, train_spectra, train_target, n_epochs):
    regressor_model.train()
    optimizer = torch.optim.Adam(regressor_model.parameters(), lr=LEARNING_RATE)
    soc_loss = nn.MSELoss()
    train_loader = build_loader_features_and_targets(train_spectra, train_target, shuffle=True)

    for epoch_index in range(n_epochs):
        for batch_spectra, batch_target in train_loader:
            optimizer.zero_grad()
            predicted_soc = regressor_model(batch_spectra)
            loss = soc_loss(predicted_soc, batch_target)
            loss.backward()
            optimizer.step()


def train_p2bn_jointly(p2bn_model, train_spectra, train_target):
    p2bn_model.train()
    optimizer = torch.optim.Adam(p2bn_model.parameters(), lr=LEARNING_RATE)
    soc_loss_fn = nn.MSELoss()
    reconstruction_loss_fn = nn.MSELoss()
    train_loader = build_loader_features_and_targets(train_spectra, train_target, shuffle=True)

    for epoch_index in range(SUPERVISED_EPOCHS):
        for batch_spectra, batch_target in train_loader:
            optimizer.zero_grad()
            soc_prediction, reconstructed_spectra = p2bn_model(batch_spectra)
            soc_loss_value = soc_loss_fn(soc_prediction, batch_target)
            reconstruction_loss_value = reconstruction_loss_fn(reconstructed_spectra, batch_spectra)
            combined_loss = (
                P2BN_SOC_LOSS_WEIGHT * soc_loss_value
                + P2BN_RECONSTRUCTION_LOSS_WEIGHT * reconstruction_loss_value
            )
            combined_loss.backward()
            optimizer.step()


def predict_soc_for_p2bn(p2bn_model, spectra_matrix):
    p2bn_model.eval()
    with torch.no_grad():
        soc_prediction, _ = p2bn_model(to_device_tensor(spectra_matrix))
    return soc_prediction.cpu().numpy()


def transform_spectra_through_frozen_batchnorm(autoencoder_model, spectra_matrix):
    autoencoder_model.eval()
    with torch.no_grad():
        normalised = autoencoder_model.preprocessing_batchnorm(to_device_tensor(spectra_matrix))
    return normalised.cpu().numpy()


def predict_for_set(regressor_model, spectra_matrix):
    regressor_model.eval()
    with torch.no_grad():
        predictions = regressor_model(to_device_tensor(spectra_matrix)).cpu().numpy()
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


def cell_output_paths(dataset_name, preprocessing_name, method_name):
    cell_json_path = CELLS_DIR / f"{dataset_name}_{preprocessing_name}_{method_name}.json"
    predictions_path = PREDICTIONS_DIR / f"{dataset_name}_{preprocessing_name}_{method_name}.csv"
    return cell_json_path, predictions_path


def build_configuration_block_for_method(method_name):
    if method_name == "baseline":
        return {
            "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "supervised_epochs": SUPERVISED_EPOCHS, "uses_batchnorm": False, "batchnorm_pretrained": False,
        }
    if method_name == "pbn":
        return {
            "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "supervised_epochs": SUPERVISED_EPOCHS, "autoencoder_pretrain_epochs": AUTOENCODER_PRETRAIN_EPOCHS,
            "uses_batchnorm": True, "batchnorm_pretrained": True, "batchnorm_jointly_finetuned_with_head": True,
        }
    if method_name == "plsr_pbn":
        return {
            "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "autoencoder_pretrain_epochs": AUTOENCODER_PRETRAIN_EPOCHS,
            "plsr_lv_count": PLSR_PBN_LV_COUNT,
            "uses_batchnorm": True, "batchnorm_pretrained": True, "batchnorm_jointly_finetuned_with_head": False,
            "downstream_regressor": "PLSRegression",
        }
    if method_name == "rbn":
        return {
            "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "supervised_epochs": SUPERVISED_EPOCHS,
            "uses_batchnorm": True, "batchnorm_pretrained": False, "batchnorm_jointly_finetuned_with_head": True,
        }
    if method_name == "r2bn":
        return {
            "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "supervised_epochs": DOUBLE_SUPERVISED_EPOCHS,
            "uses_batchnorm": True, "batchnorm_pretrained": False, "batchnorm_jointly_finetuned_with_head": True,
        }
    if method_name == "p2bn":
        return {
            "random_seed": RANDOM_SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "supervised_epochs": SUPERVISED_EPOCHS,
            "uses_batchnorm": True, "batchnorm_pretrained": False, "batchnorm_jointly_finetuned_with_head": True,
            "joint_autoencoder_branch": True,
            "soc_loss_weight": P2BN_SOC_LOSS_WEIGHT,
            "reconstruction_loss_weight": P2BN_RECONSTRUCTION_LOSS_WEIGHT,
        }
    raise ValueError(f"unknown method: {method_name}")


def save_cell_result(dataset_name, preprocessing_name, method_name, train_metrics, test_metrics, runtime_seconds):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, method_name)
    payload = {
        "dataset": dataset_name,
        "preprocessing": preprocessing_name,
        "method": method_name,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": float(runtime_seconds),
        "configuration": build_configuration_block_for_method(method_name),
    }
    cell_json_path.write_text(json.dumps(payload, indent=2))


def save_cell_predictions(dataset_name, preprocessing_name, method_name, train_dataframe, train_predictions, test_dataframe, test_predictions):
    _, predictions_path = cell_output_paths(dataset_name, preprocessing_name, method_name)
    reference_columns, _ = split_columns_into_reference_and_spectra(train_dataframe)

    train_output = train_dataframe[reference_columns].copy()
    train_output["observed"] = train_dataframe[SOC_COLUMN].to_numpy()
    train_output["predicted"] = train_predictions.ravel()
    train_output["fold"] = "train"

    test_output = test_dataframe[reference_columns].copy()
    test_output["observed"] = test_dataframe[SOC_COLUMN].to_numpy()
    test_output["predicted"] = test_predictions.ravel()
    test_output["fold"] = "test"

    combined_output = pd.concat([train_output, test_output], axis=0, ignore_index=True)
    combined_output.to_csv(predictions_path, index=False)


def run_one_baseline_cell(dataset_name, preprocessing_name):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, "baseline")
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name} / baseline")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    regressor_model = BaselineSocAnn(n_features).to(DEVICE)
    train_supervised_regressor(regressor_model, train_spectra, train_target, SUPERVISED_EPOCHS)

    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_cell_result(dataset_name, preprocessing_name, "baseline", train_metrics, test_metrics, runtime_seconds)
    save_cell_predictions(
        dataset_name, preprocessing_name, "baseline",
        train_dataframe, train_predictions, test_dataframe, test_predictions,
    )
    print(
        f"  baseline: train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def run_one_pbn_cell(dataset_name, preprocessing_name):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, "pbn")
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name} / pbn")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    autoencoder_model = LearnedPreprocessingAutoencoder(n_features).to(DEVICE)
    pretrain_autoencoder(autoencoder_model, train_spectra)

    regressor_model = PbnSocAnn(autoencoder_model, n_features).to(DEVICE)
    train_supervised_regressor(regressor_model, train_spectra, train_target, SUPERVISED_EPOCHS)

    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_cell_result(dataset_name, preprocessing_name, "pbn", train_metrics, test_metrics, runtime_seconds)
    save_cell_predictions(
        dataset_name, preprocessing_name, "pbn",
        train_dataframe, train_predictions, test_dataframe, test_predictions,
    )
    print(
        f"  pbn     : train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def run_one_plsr_pbn_cell(dataset_name, preprocessing_name):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, "plsr_pbn")
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name} / plsr_pbn")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    autoencoder_model = LearnedPreprocessingAutoencoder(n_features).to(DEVICE)
    pretrain_autoencoder(autoencoder_model, train_spectra)

    train_normalised = transform_spectra_through_frozen_batchnorm(autoencoder_model, train_spectra)
    test_normalised = transform_spectra_through_frozen_batchnorm(autoencoder_model, test_spectra)

    plsr_model = PLSRegression(n_components=PLSR_PBN_LV_COUNT, scale=False)
    plsr_model.fit(train_normalised, train_target)
    train_predictions = plsr_model.predict(train_normalised).ravel()
    test_predictions = plsr_model.predict(test_normalised).ravel()
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_cell_result(dataset_name, preprocessing_name, "plsr_pbn", train_metrics, test_metrics, runtime_seconds)
    save_cell_predictions(
        dataset_name, preprocessing_name, "plsr_pbn",
        train_dataframe, train_predictions, test_dataframe, test_predictions,
    )
    print(
        f"  plsr_pbn: train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def run_one_rbn_cell(dataset_name, preprocessing_name):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, "rbn")
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name} / rbn")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    regressor_model = RbnSocAnn(n_features).to(DEVICE)
    train_supervised_regressor(regressor_model, train_spectra, train_target, SUPERVISED_EPOCHS)

    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_cell_result(dataset_name, preprocessing_name, "rbn", train_metrics, test_metrics, runtime_seconds)
    save_cell_predictions(
        dataset_name, preprocessing_name, "rbn",
        train_dataframe, train_predictions, test_dataframe, test_predictions,
    )
    print(
        f"  rbn     : train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def run_one_p2bn_cell(dataset_name, preprocessing_name):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, "p2bn")
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name} / p2bn")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    p2bn_model = P2bnSocAnn(n_features).to(DEVICE)
    train_p2bn_jointly(p2bn_model, train_spectra, train_target)

    train_predictions = predict_soc_for_p2bn(p2bn_model, train_spectra)
    test_predictions = predict_soc_for_p2bn(p2bn_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_cell_result(dataset_name, preprocessing_name, "p2bn", train_metrics, test_metrics, runtime_seconds)
    save_cell_predictions(
        dataset_name, preprocessing_name, "p2bn",
        train_dataframe, train_predictions, test_dataframe, test_predictions,
    )
    print(
        f"  p2bn    : train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def run_one_r2bn_cell(dataset_name, preprocessing_name):
    cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, "r2bn")
    if cell_json_path.exists():
        print(f"  [skip] {dataset_name} / {preprocessing_name} / r2bn")
        return

    cell_start_seconds = time.time()
    reset_all_random_seeds()

    train_dataframe, test_dataframe = load_one_preprocessed_pair(dataset_name, preprocessing_name)
    train_spectra, train_target = extract_spectra_and_target(train_dataframe)
    test_spectra, test_target = extract_spectra_and_target(test_dataframe)
    n_features = train_spectra.shape[1]

    regressor_model = RbnSocAnn(n_features).to(DEVICE)
    train_supervised_regressor(regressor_model, train_spectra, train_target, DOUBLE_SUPERVISED_EPOCHS)

    train_predictions = predict_for_set(regressor_model, train_spectra)
    test_predictions = predict_for_set(regressor_model, test_spectra)
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)

    runtime_seconds = time.time() - cell_start_seconds
    save_cell_result(dataset_name, preprocessing_name, "r2bn", train_metrics, test_metrics, runtime_seconds)
    save_cell_predictions(
        dataset_name, preprocessing_name, "r2bn",
        train_dataframe, train_predictions, test_dataframe, test_predictions,
    )
    print(
        f"  r2bn    : train RMSE={train_metrics['rmse']:.4f}  "
        f"test RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"({runtime_seconds:.1f} s)"
    )


def aggregate_all_cells_into_summary_csv():
    rows = []
    for dataset_name in DATASET_NAMES:
        for preprocessing_name in PREPROCESSING_NAMES:
            for method_name in METHOD_NAMES:
                cell_json_path, _ = cell_output_paths(dataset_name, preprocessing_name, method_name)
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
                rows.append(row)
    summary_dataframe = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "cell_results.csv"
    summary_dataframe.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path} ({len(rows)} cells)")
    return summary_path


def main():
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    overall_start_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        for preprocessing_name in PREPROCESSING_NAMES:
            print(f"\n>>> {dataset_name} / {preprocessing_name}")
            run_one_baseline_cell(dataset_name, preprocessing_name)
            run_one_pbn_cell(dataset_name, preprocessing_name)
            run_one_plsr_pbn_cell(dataset_name, preprocessing_name)
            run_one_rbn_cell(dataset_name, preprocessing_name)
            run_one_r2bn_cell(dataset_name, preprocessing_name)
            run_one_p2bn_cell(dataset_name, preprocessing_name)

    aggregate_all_cells_into_summary_csv()
    overall_elapsed_seconds = time.time() - overall_start_seconds
    print(f"\nAll cells done. Total wall time: {overall_elapsed_seconds:.1f} s")


if __name__ == "__main__":
    main()
