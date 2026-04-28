from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLSR_CELLS_DIR = RESULTS_DIR / "per_cell"
DDP_CELLS_DIR = RESULTS_DIR / "ddp_experiment" / "cells"
COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.csv"

DATASET_NAMES = ["indonesia", "kenya", "china", "global"]
PREPROCESSING_NAMES = ["none", "snv", "msc", "sg", "sgd", "minmax", "ddp"]
LEARNED_PREPROCESSING_NAMES = {"ddp"}


def read_plsr_cell(dataset_name, preprocessing_name):
    cell_path = PLSR_CELLS_DIR / f"{dataset_name}_{preprocessing_name}.json"
    if not cell_path.exists():
        return None
    return json.loads(cell_path.read_text())


def read_mlp_cell(dataset_name, preprocessing_name):
    cell_path = DDP_CELLS_DIR / f"{dataset_name}_{preprocessing_name}_mlp.json"
    if not cell_path.exists():
        return None
    return json.loads(cell_path.read_text())


def gather_test_metric(payload, metric_name):
    if payload is None:
        return float("nan")
    return payload["test_metrics"][metric_name]


def gather_train_metric(payload, metric_name):
    if payload is None:
        return float("nan")
    return payload["train_metrics"][metric_name]


def build_long_format_rows():
    rows = []
    for dataset_name in DATASET_NAMES:
        for preprocessing_name in PREPROCESSING_NAMES:
            plsr_payload = read_plsr_cell(dataset_name, preprocessing_name)
            mlp_payload = read_mlp_cell(dataset_name, preprocessing_name)

            row = {
                "dataset": dataset_name,
                "preprocessing": preprocessing_name,
                "plsr_test_rmse": gather_test_metric(plsr_payload, "rmse"),
                "plsr_test_r2": gather_test_metric(plsr_payload, "r2"),
                "plsr_test_mbd": gather_test_metric(plsr_payload, "mbd"),
                "plsr_test_rpiq": gather_test_metric(plsr_payload, "rpiq"),
                "mlp_test_rmse": gather_test_metric(mlp_payload, "rmse"),
                "mlp_test_r2": gather_test_metric(mlp_payload, "r2"),
                "mlp_test_mbd": gather_test_metric(mlp_payload, "mbd"),
                "mlp_test_rpiq": gather_test_metric(mlp_payload, "rpiq"),
                "mlp_train_rmse": gather_train_metric(mlp_payload, "rmse"),
                "mlp_train_r2": gather_train_metric(mlp_payload, "r2"),
            }
            if preprocessing_name in LEARNED_PREPROCESSING_NAMES and mlp_payload is not None:
                row["mlp_stage1_test_rmse"] = mlp_payload["stage1_test_metrics"]["rmse"]
                row["mlp_stage1_test_r2"] = mlp_payload["stage1_test_metrics"]["r2"]
            rows.append(row)
    return pd.DataFrame(rows)


def pivot_wide_table(long_format_dataframe, value_column):
    return long_format_dataframe.pivot(
        index="preprocessing",
        columns="dataset",
        values=value_column,
    ).reindex(PREPROCESSING_NAMES)[DATASET_NAMES]


def format_wide_table_for_print(wide_dataframe):
    return wide_dataframe.round(4).to_string(na_rep="—")


def per_dataset_winner(long_format_dataframe, algorithm_prefix):
    column = f"{algorithm_prefix}_test_rmse"
    rows = []
    for dataset_name in DATASET_NAMES:
        slice_for_dataset = long_format_dataframe[long_format_dataframe["dataset"] == dataset_name]
        slice_with_value = slice_for_dataset[~slice_for_dataset[column].isna()]
        if slice_with_value.empty:
            rows.append({"dataset": dataset_name, "winner_preprocessing": "—", "test_rmse": float("nan")})
            continue
        winning_index = slice_with_value[column].idxmin()
        rows.append({
            "dataset": dataset_name,
            "winner_preprocessing": slice_with_value.loc[winning_index, "preprocessing"],
            "test_rmse": float(slice_with_value.loc[winning_index, column]),
        })
    return pd.DataFrame(rows)


def main():
    long_format_dataframe = build_long_format_rows()
    long_format_dataframe.to_csv(COMPARISON_TABLE_PATH, index=False)
    print(f"Wrote long-format comparison: {COMPARISON_TABLE_PATH} ({len(long_format_dataframe)} rows)\n")

    print("=== MLP test RMSE (preprocessing × dataset) ===")
    mlp_wide = pivot_wide_table(long_format_dataframe, "mlp_test_rmse")
    print(format_wide_table_for_print(mlp_wide))

    print("\n=== PLSR test RMSE (preprocessing × dataset) ===")
    plsr_wide = pivot_wide_table(long_format_dataframe, "plsr_test_rmse")
    print(format_wide_table_for_print(plsr_wide))

    print("\n=== MLP test R² (preprocessing × dataset) ===")
    mlp_r2_wide = pivot_wide_table(long_format_dataframe, "mlp_test_r2")
    print(format_wide_table_for_print(mlp_r2_wide))

    print("\n=== PLSR test R² (preprocessing × dataset) ===")
    plsr_r2_wide = pivot_wide_table(long_format_dataframe, "plsr_test_r2")
    print(format_wide_table_for_print(plsr_r2_wide))

    print("\n=== Per-dataset winner (lowest test RMSE) ===")
    mlp_winners = per_dataset_winner(long_format_dataframe, algorithm_prefix="mlp")
    plsr_winners = per_dataset_winner(long_format_dataframe, algorithm_prefix="plsr")
    winners_combined = mlp_winners.merge(
        plsr_winners,
        on="dataset",
        suffixes=("_mlp", "_plsr"),
    )
    print(winners_combined.round(4).to_string(index=False))

    print("\n=== Stage 1 vs stage 2 for learned preprocessors (lossless two-stage check) ===")
    learned_rows = long_format_dataframe[
        long_format_dataframe["preprocessing"].isin(LEARNED_PREPROCESSING_NAMES)
    ][["preprocessing", "dataset", "mlp_stage1_test_rmse", "mlp_test_rmse"]].rename(
        columns={"mlp_test_rmse": "mlp_stage2_test_rmse"}
    )
    print(learned_rows.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
