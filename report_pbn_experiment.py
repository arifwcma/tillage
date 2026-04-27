from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "pbn_experiment"
SUMMARY_CSV_PATH = RESULTS_DIR / "cell_results.csv"

DATASET_DISPLAY_NAMES = {
    "global": "Global",
    "china": "China",
    "kenya": "Kenya",
    "indonesia": "Indonesia",
}

PREPROCESSING_DISPLAY_NAMES = {
    "none": "None",
    "snv": "SNV",
    "msc": "MSC",
    "sg": "SG",
    "sgd": "SGD",
    "minmax": "MinMax",
}

METHOD_DISPLAY_NAMES = {
    "baseline": "baseline",
    "pbn": "pbn",
    "plsr_pbn": "plsr_pbn",
    "rbn": "rbn",
    "r2bn": "r2bn",
}

DATASET_ORDER = ["global", "china", "kenya", "indonesia"]
PREPROCESSING_ORDER = ["none", "snv", "msc", "sg", "sgd", "minmax"]
METHOD_ORDER = ["baseline", "pbn", "plsr_pbn", "rbn", "r2bn"]

PBN_VS_BASELINE_PAIRS = [
    ("pbn", "baseline"),
    ("pbn", "rbn"),
    ("pbn", "r2bn"),
    ("plsr_pbn", "baseline"),
    ("plsr_pbn", "pbn"),
    ("rbn", "baseline"),
]


def load_summary_dataframe():
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(f"summary not found: {SUMMARY_CSV_PATH}")
    return pd.read_csv(SUMMARY_CSV_PATH)


def lookup_cell_metrics(summary_dataframe, dataset_name, preprocessing_name, method_name):
    matched = summary_dataframe[
        (summary_dataframe["dataset"] == dataset_name)
        & (summary_dataframe["preprocessing"] == preprocessing_name)
        & (summary_dataframe["method"] == method_name)
    ]
    if len(matched) == 0:
        return None
    return matched.iloc[0]


def format_one_method_line(method_label, metrics_row):
    return (
        f"  {method_label:<10}: RMSE={metrics_row['test_rmse']:.4f}  "
        f"R2={metrics_row['test_r2']:.4f}  "
        f"MBD={metrics_row['test_mbd']:+.4f}  "
        f"RPIQ={metrics_row['test_rpiq']:.3f}"
    )


def format_delta_line(left_method, right_method, left_metrics, right_metrics):
    rmse_difference = float(left_metrics["test_rmse"]) - float(right_metrics["test_rmse"])
    if rmse_difference < 0:
        verdict = f"{left_method} better"
    elif rmse_difference > 0:
        verdict = f"{right_method} better"
    else:
        verdict = "tie"
    return f"  d({left_method:>8} - {right_method:<8}) RMSE = {rmse_difference:+.4f}   ({verdict})"


def format_one_preprocessing_block(summary_dataframe, dataset_name, preprocessing_name):
    lines = [f"\nPreprocessing: {PREPROCESSING_DISPLAY_NAMES[preprocessing_name]}"]
    method_to_metrics = {}
    for method_name in METHOD_ORDER:
        metrics_row = lookup_cell_metrics(summary_dataframe, dataset_name, preprocessing_name, method_name)
        method_to_metrics[method_name] = metrics_row
        if metrics_row is None:
            lines.append(f"  {METHOD_DISPLAY_NAMES[method_name]:<10}: (missing)")
        else:
            lines.append(format_one_method_line(METHOD_DISPLAY_NAMES[method_name], metrics_row))

    for left_method, right_method in PBN_VS_BASELINE_PAIRS:
        left_metrics = method_to_metrics.get(left_method)
        right_metrics = method_to_metrics.get(right_method)
        if left_metrics is None or right_metrics is None:
            continue
        lines.append(format_delta_line(left_method, right_method, left_metrics, right_metrics))
    return "\n".join(lines)


def format_one_dataset_block(summary_dataframe, dataset_name):
    sample_row = None
    for method_name in METHOD_ORDER:
        sample_row = lookup_cell_metrics(summary_dataframe, dataset_name, "none", method_name)
        if sample_row is not None:
            break
    if sample_row is not None:
        n_train_value = int(sample_row["n_train"])
        n_test_value = int(sample_row["n_test"])
        header_size_text = f"  (n_train={n_train_value}, n_test={n_test_value})"
    else:
        header_size_text = ""

    header_line = f"\n=========================================================\nDataset: {DATASET_DISPLAY_NAMES[dataset_name]}{header_size_text}\n========================================================="
    blocks = [header_line]
    for preprocessing_name in PREPROCESSING_ORDER:
        blocks.append(format_one_preprocessing_block(summary_dataframe, dataset_name, preprocessing_name))
    return "\n".join(blocks)


def count_wins_for_method_pair_per_dataset(summary_dataframe, dataset_name, left_method, right_method):
    n_left_wins = 0
    n_compared = 0
    for preprocessing_name in PREPROCESSING_ORDER:
        left_metrics = lookup_cell_metrics(summary_dataframe, dataset_name, preprocessing_name, left_method)
        right_metrics = lookup_cell_metrics(summary_dataframe, dataset_name, preprocessing_name, right_method)
        if left_metrics is None or right_metrics is None:
            continue
        n_compared += 1
        if float(left_metrics["test_rmse"]) < float(right_metrics["test_rmse"]):
            n_left_wins += 1
    return n_left_wins, n_compared


def format_overall_summary_block(summary_dataframe):
    lines = []
    lines.append("\n=========================================================")
    lines.append("Summary: head-to-head wins on test RMSE (lower is better)")
    lines.append("=========================================================")
    for left_method, right_method in PBN_VS_BASELINE_PAIRS:
        lines.append(f"\n  {left_method} vs {right_method}:")
        total_left_wins = 0
        total_compared = 0
        for dataset_name in DATASET_ORDER:
            n_left_wins, n_compared = count_wins_for_method_pair_per_dataset(
                summary_dataframe, dataset_name, left_method, right_method
            )
            total_left_wins += n_left_wins
            total_compared += n_compared
            lines.append(
                f"    {DATASET_DISPLAY_NAMES[dataset_name]:<10}: "
                f"{left_method} wins {n_left_wins}/{n_compared} preprocessings"
            )
        lines.append(
            f"    {'OVERALL':<10}: {left_method} wins {total_left_wins}/{total_compared} cells"
        )
    return "\n".join(lines)


def main():
    summary_dataframe = load_summary_dataframe()
    for dataset_name in DATASET_ORDER:
        print(format_one_dataset_block(summary_dataframe, dataset_name))
    print(format_overall_summary_block(summary_dataframe))


if __name__ == "__main__":
    main()
