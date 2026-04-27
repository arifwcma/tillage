from pathlib import Path
import json
import re
import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"
PER_CELL_DIR = RESULTS_DIR / "per_cell"

DATASET_NAMES = ["china", "kenya", "indonesia", "global"]
METHOD_NAMES = ["none", "snv", "msc", "sg", "sgd"]

GROUP_KEY_COLUMN = "Batch and labid"
SOC_COLUMN = "Org C"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m\d+(?:\.\d+)?$")

CV_N_REPEATS = 5
CV_N_FOLDS = 10
CV_BASE_RANDOM_SEED = 42

PLSR_LV_MIN = 1
PLSR_LV_MAX = 25

SG_WINDOW_GRID = list(range(7, 32, 2))
SG_POLYORDER_GRID = [2, 3]
SGD_DERIVATIVE_ORDER = 2

N_SOC_QUARTILES = 4


def is_wavenumber_column(column_name):
    return WAVENUMBER_COLUMN_PATTERN.match(column_name) is not None


def split_columns_into_reference_and_spectra(dataset):
    spectra_columns = [c for c in dataset.columns if is_wavenumber_column(c)]
    reference_columns = [c for c in dataset.columns if c not in spectra_columns]
    return reference_columns, spectra_columns


def load_raw_train_and_test(dataset_name):
    train_dataframe = pd.read_csv(
        PREPROCESSED_DIR / f"{dataset_name}_none_train.csv", low_memory=False
    )
    test_dataframe = pd.read_csv(
        PREPROCESSED_DIR / f"{dataset_name}_none_test.csv", low_memory=False
    )
    return train_dataframe, test_dataframe


def apply_no_preprocessing(train_spectra, validation_spectra):
    return train_spectra.copy(), validation_spectra.copy()


def apply_snv(train_spectra, validation_spectra):
    return snv_per_spectrum(train_spectra), snv_per_spectrum(validation_spectra)


def snv_per_spectrum(spectra_matrix):
    row_means = spectra_matrix.mean(axis=1, keepdims=True)
    row_stds = spectra_matrix.std(axis=1, ddof=1, keepdims=True)
    return (spectra_matrix - row_means) / row_stds


def apply_msc(train_spectra, validation_spectra):
    reference_spectrum = train_spectra.mean(axis=0)
    return (
        msc_correct(train_spectra, reference_spectrum),
        msc_correct(validation_spectra, reference_spectrum),
    )


def msc_correct(spectra_matrix, reference_spectrum):
    reference_centered = reference_spectrum - reference_spectrum.mean()
    reference_norm_squared = float(np.dot(reference_centered, reference_centered))
    n_rows = spectra_matrix.shape[0]
    corrected = np.empty_like(spectra_matrix)
    for row_index in range(n_rows):
        row_spectrum = spectra_matrix[row_index]
        row_centered = row_spectrum - row_spectrum.mean()
        slope = float(np.dot(reference_centered, row_centered)) / reference_norm_squared
        intercept = row_spectrum.mean() - slope * reference_spectrum.mean()
        corrected[row_index] = (row_spectrum - intercept) / slope
    return corrected


def apply_savgol_with_parameters(train_spectra, validation_spectra, window_length, polynomial_order, derivative_order):
    train_transformed = savgol_filter(
        train_spectra,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=derivative_order,
        axis=1,
        mode="interp",
    )
    validation_transformed = savgol_filter(
        validation_spectra,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=derivative_order,
        axis=1,
        mode="interp",
    )
    return train_transformed, validation_transformed


def fit_plsr_and_predict_for_all_lv_counts(train_spectra, train_target, validation_spectra, max_lv):
    train_spectra_mean = train_spectra.mean(axis=0)
    train_target_mean = train_target.mean()
    train_spectra_centered = train_spectra - train_spectra_mean
    train_target_centered = train_target - train_target_mean
    validation_spectra_centered = validation_spectra - train_spectra_mean

    model = PLSRegression(n_components=max_lv, scale=False)
    model.fit(train_spectra_centered, train_target_centered)
    rotations = model.x_rotations_
    target_loadings = model.y_loadings_

    predictions_per_lv = np.zeros((validation_spectra.shape[0], max_lv))
    for lv_count in range(1, max_lv + 1):
        coefficient_vector = rotations[:, :lv_count] @ target_loadings[:, :lv_count].T
        validation_predictions_centered = validation_spectra_centered @ coefficient_vector
        predictions_per_lv[:, lv_count - 1] = validation_predictions_centered.ravel() + train_target_mean
    return predictions_per_lv


def fit_plsr_with_single_lv(train_spectra, train_target, lv_count):
    train_spectra_mean = train_spectra.mean(axis=0)
    train_target_mean = train_target.mean()
    model = PLSRegression(n_components=lv_count, scale=False)
    model.fit(train_spectra - train_spectra_mean, train_target - train_target_mean)
    return model, train_spectra_mean, train_target_mean


def predict_with_fitted_plsr(model, validation_spectra, train_spectra_mean, train_target_mean):
    centered = validation_spectra - train_spectra_mean
    centered_predictions = model.predict(centered).ravel()
    return centered_predictions + train_target_mean


def compute_rmse(observed, predicted):
    residual = observed.ravel() - predicted.ravel()
    return float(np.sqrt((residual ** 2).mean()))


def compute_r_squared(observed, predicted):
    observed = observed.ravel()
    predicted = predicted.ravel()
    sum_squared_residuals = float(((observed - predicted) ** 2).sum())
    sum_squared_total = float(((observed - observed.mean()) ** 2).sum())
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


def make_repeated_grouped_stratified_folds(soc_quartiles, group_values):
    fold_specifications = []
    for repeat_index in range(CV_N_REPEATS):
        random_seed = CV_BASE_RANDOM_SEED + repeat_index
        splitter = StratifiedGroupKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=random_seed)
        for fold_index, (inner_train_indices, inner_validation_indices) in enumerate(
            splitter.split(np.zeros(len(soc_quartiles)), soc_quartiles, group_values)
        ):
            fold_specifications.append({
                "repeat": repeat_index,
                "fold": fold_index,
                "train_indices": inner_train_indices,
                "validation_indices": inner_validation_indices,
            })
    return fold_specifications


def assign_soc_quartile_labels(soc_values):
    return pd.qcut(soc_values, q=N_SOC_QUARTILES, labels=False, duplicates="drop").astype(int)


def select_preprocessing_for_cv(method_name):
    if method_name == "none":
        return [{"label": "none"}]
    if method_name == "snv":
        return [{"label": "snv"}]
    if method_name == "msc":
        return [{"label": "msc"}]
    if method_name == "sg":
        return [
            {"label": "sg", "window": w, "polyorder": p, "deriv": 0}
            for w in SG_WINDOW_GRID
            for p in SG_POLYORDER_GRID
        ]
    if method_name == "sgd":
        return [
            {"label": "sgd", "window": w, "polyorder": p, "deriv": SGD_DERIVATIVE_ORDER}
            for w in SG_WINDOW_GRID
            for p in SG_POLYORDER_GRID
        ]
    raise ValueError(f"unknown method: {method_name}")


def transform_with_preprocessing_specification(method_specification, train_spectra, validation_spectra):
    label = method_specification["label"]
    if label == "none":
        return apply_no_preprocessing(train_spectra, validation_spectra)
    if label == "snv":
        return apply_snv(train_spectra, validation_spectra)
    if label == "msc":
        return apply_msc(train_spectra, validation_spectra)
    return apply_savgol_with_parameters(
        train_spectra,
        validation_spectra,
        method_specification["window"],
        method_specification["polyorder"],
        method_specification["deriv"],
    )


def run_cross_validation_for_one_cell(method_name, train_spectra_raw, train_target, train_groups, train_strata):
    fold_specifications = make_repeated_grouped_stratified_folds(train_strata, train_groups)
    preprocessing_grid = select_preprocessing_for_cv(method_name)

    n_lv = PLSR_LV_MAX - PLSR_LV_MIN + 1
    n_preprocessing = len(preprocessing_grid)
    n_folds_total = len(fold_specifications)

    rmse_table = np.full((n_preprocessing, n_lv, n_folds_total), np.nan)

    for fold_index, fold_specification in enumerate(fold_specifications):
        inner_train_indices = fold_specification["train_indices"]
        inner_validation_indices = fold_specification["validation_indices"]
        inner_train_spectra = train_spectra_raw[inner_train_indices]
        inner_validation_spectra = train_spectra_raw[inner_validation_indices]
        inner_train_target = train_target[inner_train_indices]
        inner_validation_target = train_target[inner_validation_indices]

        for preprocessing_index, preprocessing_specification in enumerate(preprocessing_grid):
            inner_train_transformed, inner_validation_transformed = transform_with_preprocessing_specification(
                preprocessing_specification, inner_train_spectra, inner_validation_spectra
            )
            predictions_per_lv = fit_plsr_and_predict_for_all_lv_counts(
                inner_train_transformed,
                inner_train_target,
                inner_validation_transformed,
                PLSR_LV_MAX,
            )
            for lv_index in range(n_lv):
                rmse_table[preprocessing_index, lv_index, fold_index] = compute_rmse(
                    inner_validation_target, predictions_per_lv[:, lv_index]
                )

    mean_rmse_grid = rmse_table.mean(axis=2)
    se_rmse_grid = rmse_table.std(axis=2, ddof=1) / np.sqrt(n_folds_total)
    return preprocessing_grid, mean_rmse_grid, se_rmse_grid


def select_winning_candidate_with_one_se_rule(preprocessing_grid, mean_rmse_grid, se_rmse_grid):
    minimum_index_flat = int(np.argmin(mean_rmse_grid))
    n_preprocessing, n_lv = mean_rmse_grid.shape
    minimum_preprocessing_index = minimum_index_flat // n_lv
    minimum_lv_index = minimum_index_flat % n_lv
    minimum_mean_rmse = float(mean_rmse_grid[minimum_preprocessing_index, minimum_lv_index])
    se_at_minimum = float(se_rmse_grid[minimum_preprocessing_index, minimum_lv_index])
    one_se_threshold = minimum_mean_rmse + se_at_minimum

    candidate_records = []
    for preprocessing_index in range(n_preprocessing):
        for lv_index in range(n_lv):
            if mean_rmse_grid[preprocessing_index, lv_index] <= one_se_threshold:
                candidate_records.append({
                    "preprocessing_index": preprocessing_index,
                    "lv_index": lv_index,
                    "lv_count": lv_index + PLSR_LV_MIN,
                    "mean_rmse": float(mean_rmse_grid[preprocessing_index, lv_index]),
                    "se_rmse": float(se_rmse_grid[preprocessing_index, lv_index]),
                    "preprocessing_specification": preprocessing_grid[preprocessing_index],
                })

    def sort_key(record):
        specification = record["preprocessing_specification"]
        window_value = specification.get("window", 0)
        polyorder_value = specification.get("polyorder", 0)
        return (record["lv_count"], window_value, polyorder_value)

    candidate_records.sort(key=sort_key)
    chosen_record = candidate_records[0]
    chosen_record["minimum_mean_rmse"] = minimum_mean_rmse
    chosen_record["se_at_minimum"] = se_at_minimum
    chosen_record["one_se_threshold"] = one_se_threshold
    chosen_record["n_candidates_within_one_se"] = len(candidate_records)
    return chosen_record


def refit_with_winner_and_evaluate_on_test(winning_record, train_spectra_raw, train_target, test_spectra_raw, test_target):
    winning_specification = winning_record["preprocessing_specification"]
    winning_lv_count = winning_record["lv_count"]
    train_transformed, test_transformed = transform_with_preprocessing_specification(
        winning_specification, train_spectra_raw, test_spectra_raw
    )
    fitted_model, train_spectra_mean, train_target_mean = fit_plsr_with_single_lv(
        train_transformed, train_target, winning_lv_count
    )
    train_predictions = predict_with_fitted_plsr(
        fitted_model, train_transformed, train_spectra_mean, train_target_mean
    )
    test_predictions = predict_with_fitted_plsr(
        fitted_model, test_transformed, train_spectra_mean, train_target_mean
    )
    train_metrics = compute_metrics_dictionary(train_target, train_predictions)
    test_metrics = compute_metrics_dictionary(test_target, test_predictions)
    return train_predictions, test_predictions, train_metrics, test_metrics


def write_per_cell_results(dataset_name, method_name, winning_record, train_metrics, test_metrics, runtime_seconds):
    output = {
        "dataset": dataset_name,
        "method": method_name,
        "winner": {
            "lv_count": int(winning_record["lv_count"]),
            "preprocessing_specification": winning_record["preprocessing_specification"],
            "cv_mean_rmse": winning_record["mean_rmse"],
            "cv_se_rmse": winning_record["se_rmse"],
            "cv_minimum_mean_rmse": winning_record["minimum_mean_rmse"],
            "cv_se_at_minimum": winning_record["se_at_minimum"],
            "cv_one_se_threshold": winning_record["one_se_threshold"],
            "n_candidates_within_one_se": int(winning_record["n_candidates_within_one_se"]),
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": float(runtime_seconds),
        "configuration": {
            "cv_n_repeats": CV_N_REPEATS,
            "cv_n_folds": CV_N_FOLDS,
            "cv_base_random_seed": CV_BASE_RANDOM_SEED,
            "plsr_lv_min": PLSR_LV_MIN,
            "plsr_lv_max": PLSR_LV_MAX,
            "sg_window_grid": SG_WINDOW_GRID,
            "sg_polyorder_grid": SG_POLYORDER_GRID,
            "sgd_derivative_order": SGD_DERIVATIVE_ORDER,
        },
    }
    output_path = PER_CELL_DIR / f"{dataset_name}_{method_name}.json"
    output_path.write_text(json.dumps(output, indent=2))
    return output_path


def write_predictions_csv(dataset_name, method_name, train_dataframe, train_predictions, test_dataframe, test_predictions):
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
    output_path = PER_CELL_DIR / f"{dataset_name}_{method_name}_predictions.csv"
    combined_output.to_csv(output_path, index=False)
    return output_path


def run_one_cell(dataset_name, method_name):
    output_json_path = PER_CELL_DIR / f"{dataset_name}_{method_name}.json"
    if output_json_path.exists():
        print(f"  [skip] {dataset_name} / {method_name}: {output_json_path.name} already exists")
        return

    print(f"\n>>> {dataset_name} / {method_name}")
    cell_start_seconds = time.time()

    train_dataframe, test_dataframe = load_raw_train_and_test(dataset_name)
    _, spectra_columns = split_columns_into_reference_and_spectra(train_dataframe)

    train_spectra_raw = train_dataframe[spectra_columns].to_numpy(dtype=np.float64)
    test_spectra_raw = test_dataframe[spectra_columns].to_numpy(dtype=np.float64)
    train_target = train_dataframe[SOC_COLUMN].to_numpy(dtype=np.float64)
    test_target = test_dataframe[SOC_COLUMN].to_numpy(dtype=np.float64)
    train_groups = train_dataframe[GROUP_KEY_COLUMN].to_numpy()
    train_strata = assign_soc_quartile_labels(train_dataframe[SOC_COLUMN]).to_numpy()

    print(f"  train rows: {len(train_dataframe)}  test rows: {len(test_dataframe)}  features: {len(spectra_columns)}")

    cv_start_seconds = time.time()
    preprocessing_grid, mean_rmse_grid, se_rmse_grid = run_cross_validation_for_one_cell(
        method_name, train_spectra_raw, train_target, train_groups, train_strata
    )
    cv_elapsed_seconds = time.time() - cv_start_seconds
    print(f"  CV elapsed: {cv_elapsed_seconds:.1f} s  (grid size: {mean_rmse_grid.size})")

    winning_record = select_winning_candidate_with_one_se_rule(
        preprocessing_grid, mean_rmse_grid, se_rmse_grid
    )
    print(
        f"  winner: lv={winning_record['lv_count']}  spec={winning_record['preprocessing_specification']}  "
        f"cv_mean_rmse={winning_record['mean_rmse']:.4f}  (min={winning_record['minimum_mean_rmse']:.4f}, "
        f"+1SE={winning_record['one_se_threshold']:.4f}, candidates_within_1SE={winning_record['n_candidates_within_one_se']})"
    )

    train_predictions, test_predictions, train_metrics, test_metrics = refit_with_winner_and_evaluate_on_test(
        winning_record, train_spectra_raw, train_target, test_spectra_raw, test_target
    )
    print(
        f"  test : RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}  "
        f"MBD={test_metrics['mbd']:+.4f}  RPIQ={test_metrics['rpiq']:.3f}  n={test_metrics['n']}"
    )

    cell_runtime_seconds = time.time() - cell_start_seconds
    write_per_cell_results(dataset_name, method_name, winning_record, train_metrics, test_metrics, cell_runtime_seconds)
    write_predictions_csv(dataset_name, method_name, train_dataframe, train_predictions, test_dataframe, test_predictions)
    print(f"  wrote results, total cell time: {cell_runtime_seconds:.1f} s")


def main():
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    overall_start_seconds = time.time()
    for dataset_name in DATASET_NAMES:
        for method_name in METHOD_NAMES:
            run_one_cell(dataset_name, method_name)
    overall_elapsed_seconds = time.time() - overall_start_seconds
    print(f"\nAll cells done. Total wall time: {overall_elapsed_seconds:.1f} s")


if __name__ == "__main__":
    main()
