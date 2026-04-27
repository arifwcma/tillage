from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
METHOD_NAMES = ["none", "snv", "msc", "sg", "sgd", "minmax"]

GROUP_KEY_COLUMN = "Batch and labid"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m\d+(?:\.\d+)?$")

SG_WINDOW_LENGTH = 11
SG_POLYNOMIAL_ORDER = 2
SGD_DERIVATIVE_ORDER = 2


def is_wavenumber_column(column_name):
    return WAVENUMBER_COLUMN_PATTERN.match(column_name) is not None


def split_columns_into_reference_and_spectra(dataset):
    spectra_columns = [c for c in dataset.columns if is_wavenumber_column(c)]
    reference_columns = [c for c in dataset.columns if c not in spectra_columns]
    return reference_columns, spectra_columns


def attach_fold_assignment(dataset, split_table):
    deduplicated_split_table = split_table.drop_duplicates(subset=[GROUP_KEY_COLUMN], keep="first")
    folds_per_group = split_table.groupby(GROUP_KEY_COLUMN)["fold"].nunique()
    inconsistent_groups = folds_per_group[folds_per_group > 1]
    if len(inconsistent_groups) > 0:
        raise RuntimeError(
            f"split table has groups with inconsistent fold labels: {len(inconsistent_groups)}"
        )
    enriched = dataset.merge(
        deduplicated_split_table, on=GROUP_KEY_COLUMN, how="left", validate="many_to_one"
    )
    if enriched["fold"].isna().any():
        raise RuntimeError("some rows could not be matched to a fold")
    return enriched


def apply_no_preprocessing(train_spectra, test_spectra):
    return train_spectra.copy(), test_spectra.copy()


def apply_snv_per_spectrum(spectra_matrix):
    row_means = spectra_matrix.mean(axis=1, keepdims=True)
    row_stds = spectra_matrix.std(axis=1, ddof=1, keepdims=True)
    if (row_stds == 0).any():
        raise RuntimeError("zero-variance spectrum encountered in SNV")
    return (spectra_matrix - row_means) / row_stds


def apply_snv(train_spectra, test_spectra):
    return apply_snv_per_spectrum(train_spectra), apply_snv_per_spectrum(test_spectra)


def fit_msc_reference_spectrum(train_spectra):
    return train_spectra.mean(axis=0)


def apply_msc_to_one_matrix(spectra_matrix, reference_spectrum):
    n_rows = spectra_matrix.shape[0]
    corrected = np.empty_like(spectra_matrix)
    reference_spectrum_centered = reference_spectrum - reference_spectrum.mean()
    reference_norm_squared = float(np.dot(reference_spectrum_centered, reference_spectrum_centered))
    for row_index in range(n_rows):
        row_spectrum = spectra_matrix[row_index]
        row_centered = row_spectrum - row_spectrum.mean()
        slope = float(np.dot(reference_spectrum_centered, row_centered)) / reference_norm_squared
        intercept = row_spectrum.mean() - slope * reference_spectrum.mean()
        corrected[row_index] = (row_spectrum - intercept) / slope
    return corrected


def apply_msc(train_spectra, test_spectra):
    reference_spectrum = fit_msc_reference_spectrum(train_spectra)
    train_corrected = apply_msc_to_one_matrix(train_spectra, reference_spectrum)
    test_corrected = apply_msc_to_one_matrix(test_spectra, reference_spectrum)
    return train_corrected, test_corrected


def apply_savgol_to_one_matrix(spectra_matrix, window_length, polynomial_order, derivative_order):
    return savgol_filter(
        spectra_matrix,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=derivative_order,
        axis=1,
        mode="interp",
    )


def apply_sg(train_spectra, test_spectra):
    train_smoothed = apply_savgol_to_one_matrix(train_spectra, SG_WINDOW_LENGTH, SG_POLYNOMIAL_ORDER, 0)
    test_smoothed = apply_savgol_to_one_matrix(test_spectra, SG_WINDOW_LENGTH, SG_POLYNOMIAL_ORDER, 0)
    return train_smoothed, test_smoothed


def apply_sgd(train_spectra, test_spectra):
    train_derivative = apply_savgol_to_one_matrix(
        train_spectra, SG_WINDOW_LENGTH, SG_POLYNOMIAL_ORDER, SGD_DERIVATIVE_ORDER
    )
    test_derivative = apply_savgol_to_one_matrix(
        test_spectra, SG_WINDOW_LENGTH, SG_POLYNOMIAL_ORDER, SGD_DERIVATIVE_ORDER
    )
    return train_derivative, test_derivative


def fit_minmax_per_feature(train_spectra):
    column_min = train_spectra.min(axis=0, keepdims=True)
    column_max = train_spectra.max(axis=0, keepdims=True)
    column_range = column_max - column_min
    column_range[column_range == 0] = 1.0
    return column_min, column_range


def apply_minmax(train_spectra, test_spectra):
    column_min, column_range = fit_minmax_per_feature(train_spectra)
    train_scaled = (train_spectra - column_min) / column_range
    test_scaled = (test_spectra - column_min) / column_range
    return train_scaled, test_scaled


def transform_with_method(method_name, train_spectra, test_spectra):
    if method_name == "none":
        return apply_no_preprocessing(train_spectra, test_spectra)
    if method_name == "snv":
        return apply_snv(train_spectra, test_spectra)
    if method_name == "msc":
        return apply_msc(train_spectra, test_spectra)
    if method_name == "sg":
        return apply_sg(train_spectra, test_spectra)
    if method_name == "sgd":
        return apply_sgd(train_spectra, test_spectra)
    if method_name == "minmax":
        return apply_minmax(train_spectra, test_spectra)
    raise ValueError(f"unknown method: {method_name}")


def assemble_output_dataframe(reference_dataframe_subset, spectra_columns, transformed_spectra_matrix):
    spectra_dataframe = pd.DataFrame(
        transformed_spectra_matrix,
        index=reference_dataframe_subset.index,
        columns=spectra_columns,
    )
    return pd.concat([reference_dataframe_subset, spectra_dataframe], axis=1)


def write_output_csv(output_dataframe, dataset_name, method_name, fold_label):
    file_name = f"{dataset_name}_{method_name}_{fold_label}.csv"
    output_path = PREPROCESSED_DIR / file_name
    output_dataframe.to_csv(output_path, index=False)
    return output_path


def process_one_dataset_one_method(dataset_name, method_name, raw_dataset, split_table):
    enriched = attach_fold_assignment(raw_dataset, split_table)
    reference_columns, spectra_columns = split_columns_into_reference_and_spectra(raw_dataset)

    train_mask = enriched["fold"] == "train"
    test_mask = enriched["fold"] == "test"

    train_subset = enriched.loc[train_mask].reset_index(drop=True)
    test_subset = enriched.loc[test_mask].reset_index(drop=True)

    train_spectra = train_subset[spectra_columns].to_numpy(dtype=np.float64)
    test_spectra = test_subset[spectra_columns].to_numpy(dtype=np.float64)

    train_transformed, test_transformed = transform_with_method(method_name, train_spectra, test_spectra)

    train_reference = train_subset[reference_columns]
    test_reference = test_subset[reference_columns]

    train_output = assemble_output_dataframe(train_reference, spectra_columns, train_transformed)
    test_output = assemble_output_dataframe(test_reference, spectra_columns, test_transformed)

    train_path = write_output_csv(train_output, dataset_name, method_name, "train")
    test_path = write_output_csv(test_output, dataset_name, method_name, "test")

    train_finite = np.isfinite(train_transformed).all()
    test_finite = np.isfinite(test_transformed).all()
    if not (train_finite and test_finite):
        raise RuntimeError(f"non-finite values produced for {dataset_name} / {method_name}")

    print(
        f"  {dataset_name:>10} / {method_name:<4}: train={len(train_output)} test={len(test_output)} "
        f"-> {train_path.name}, {test_path.name}"
    )


def process_one_dataset(dataset_name):
    print(f"\n--- {dataset_name} ---")
    raw_dataset = pd.read_csv(RAW_DIR / f"{dataset_name}.csv", low_memory=False)
    split_table = pd.read_csv(SPLITS_DIR / f"{dataset_name}_split.csv")
    for method_name in METHOD_NAMES:
        process_one_dataset_one_method(dataset_name, method_name, raw_dataset, split_table)


def main():
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASET_NAMES:
        process_one_dataset(dataset_name)
    print("\nAll preprocessed datasets written.")


if __name__ == "__main__":
    main()
