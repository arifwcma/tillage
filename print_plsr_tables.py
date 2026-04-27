from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
TABLE_CSV_PATH = PROJECT_ROOT / "results" / "table1_replication.csv"

REGION_ORDER = ["global", "china", "kenya", "indonesia"]
METHOD_ORDER = ["none", "snv", "msc", "sg", "sgd", "minmax"]

METRIC_COLUMNS = [
    ("RMSE (% SOC, lower is better)", "rmse_ours", "{:.3f}"),
    ("R^2 (higher is better)",        "r2_ours",   "{:.3f}"),
    ("MBD (% SOC, closer to 0)",      "mbd_ours",  "{:+.3f}"),
    ("RPIQ (higher is better)",       "rpiq_ours", "{:.3f}"),
]


def strip_method_hyperparameters(method_label):
    if "(" in method_label:
        return method_label.split("(", 1)[0]
    return method_label


def load_table_with_clean_method_names():
    raw_dataframe = pd.read_csv(TABLE_CSV_PATH)
    raw_dataframe["method"] = raw_dataframe["method"].map(strip_method_hyperparameters)
    return raw_dataframe


def pivot_one_metric(dataframe, metric_column):
    pivoted = dataframe.pivot(index="method", columns="dataset", values=metric_column)
    pivoted = pivoted.reindex(index=METHOD_ORDER, columns=REGION_ORDER)
    return pivoted


def print_one_metric_table(dataframe, metric_title, metric_column, value_format):
    pivoted = pivot_one_metric(dataframe, metric_column)
    formatted = pivoted.map(value_format.format)
    print(f"\n{metric_title}")
    print("-" * len(metric_title))
    print(formatted.to_string())


def main():
    dataframe = load_table_with_clean_method_names()
    for metric_title, metric_column, value_format in METRIC_COLUMNS:
        print_one_metric_table(dataframe, metric_title, metric_column, value_format)
    print()


if __name__ == "__main__":
    main()
