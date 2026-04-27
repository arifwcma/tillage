from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PER_CELL_DIR = PROJECT_ROOT / "results" / "per_cell"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
METHOD_NAMES = ["none", "snv", "msc", "sg", "sgd", "minmax"]

PAPER_TABLE_1 = {
    ("global", "none"):    {"factors": 10, "rmse": 2.077, "r2": 0.626, "mbd":  0.006, "rpiq": 0.426},
    ("global", "snv"):     {"factors": 14, "rmse": 1.812, "r2": 0.715, "mbd": -0.028, "rpiq": 0.488},
    ("global", "msc"):     {"factors": 13, "rmse": 1.931, "r2": 0.676, "mbd": -0.009, "rpiq": 0.458},
    ("global", "sg"):      {"factors": 10, "rmse": 2.077, "r2": 0.626, "mbd":  0.006, "rpiq": 0.426},
    ("global", "sgd"):     {"factors": 11, "rmse": 1.559, "r2": 0.789, "mbd": -0.096, "rpiq": 0.568},
    ("china", "none"):     {"factors":  7, "rmse": 0.273, "r2": 0.856, "mbd": -0.006, "rpiq": 2.898},
    ("china", "snv"):      {"factors":  6, "rmse": 0.253, "r2": 0.878, "mbd": -0.024, "rpiq": 3.119},
    ("china", "msc"):      {"factors":  6, "rmse": 0.257, "r2": 0.875, "mbd": -0.004, "rpiq": 3.075},
    ("china", "sg"):       {"factors":  7, "rmse": 0.272, "r2": 0.856, "mbd": -0.006, "rpiq": 2.900},
    ("china", "sgd"):      {"factors":  4, "rmse": 0.282, "r2": 0.856, "mbd": -0.026, "rpiq": 2.803},
    ("kenya", "none"):     {"factors":  7, "rmse": 0.733, "r2": 0.920, "mbd":  0.154, "rpiq": 2.389},
    ("kenya", "snv"):      {"factors":  6, "rmse": 0.713, "r2": 0.924, "mbd":  0.077, "rpiq": 2.455},
    ("kenya", "msc"):      {"factors":  6, "rmse": 0.725, "r2": 0.919, "mbd":  0.131, "rpiq": 2.413},
    ("kenya", "sg"):       {"factors":  7, "rmse": 0.733, "r2": 0.920, "mbd":  0.154, "rpiq": 2.389},
    ("kenya", "sgd"):      {"factors":  4, "rmse": 1.300, "r2": 0.803, "mbd":  0.209, "rpiq": 1.346},
    ("indonesia", "none"): {"factors": 14, "rmse": 1.105, "r2": 0.784, "mbd":  0.018, "rpiq": 2.598},
    ("indonesia", "snv"):  {"factors": 10, "rmse": 0.893, "r2": 0.874, "mbd": -0.031, "rpiq": 3.218},
    ("indonesia", "msc"):  {"factors": 13, "rmse": 0.849, "r2": 0.876, "mbd": -0.018, "rpiq": 3.384},
    ("indonesia", "sg"):   {"factors": 14, "rmse": 1.105, "r2": 0.784, "mbd":  0.018, "rpiq": 2.598},
    ("indonesia", "sgd"):  {"factors": 14, "rmse": 1.148, "r2": 0.767, "mbd": -0.102, "rpiq": 2.501},
}

ACCEPTANCE_RMSE_RELATIVE_TOLERANCE = 0.05
ACCEPTANCE_R2_ABSOLUTE_TOLERANCE = 0.02
ACCEPTANCE_MBD_ABSOLUTE_TOLERANCE = 0.05
ACCEPTANCE_LV_ABSOLUTE_TOLERANCE = 1


def load_one_cell_record(dataset_name, method_name):
    path = PER_CELL_DIR / f"{dataset_name}_{method_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def format_method_label(method_name, winner_specification):
    if method_name in ("sg", "sgd"):
        window_value = winner_specification.get("window", "?")
        polyorder_value = winner_specification.get("polyorder", "?")
        return f"{method_name}(w={window_value},p={polyorder_value})"
    return method_name


def assess_acceptance_for_cell(paper_row, our_test_metrics, our_lv):
    rmse_relative_difference = abs(our_test_metrics["rmse"] - paper_row["rmse"]) / paper_row["rmse"]
    r2_absolute_difference = abs(our_test_metrics["r2"] - paper_row["r2"])
    mbd_absolute_difference = abs(our_test_metrics["mbd"] - paper_row["mbd"])
    lv_absolute_difference = abs(our_lv - paper_row["factors"])

    rmse_pass = rmse_relative_difference <= ACCEPTANCE_RMSE_RELATIVE_TOLERANCE
    r2_pass = r2_absolute_difference <= ACCEPTANCE_R2_ABSOLUTE_TOLERANCE
    mbd_pass = mbd_absolute_difference <= ACCEPTANCE_MBD_ABSOLUTE_TOLERANCE
    lv_pass = lv_absolute_difference <= ACCEPTANCE_LV_ABSOLUTE_TOLERANCE

    return {
        "rmse_relative_diff": rmse_relative_difference,
        "r2_absolute_diff": r2_absolute_difference,
        "mbd_absolute_diff": mbd_absolute_difference,
        "lv_absolute_diff": lv_absolute_difference,
        "rmse_pass": rmse_pass,
        "r2_pass": r2_pass,
        "mbd_pass": mbd_pass,
        "lv_pass": lv_pass,
        "all_pass": rmse_pass and r2_pass and mbd_pass and lv_pass,
    }


def build_comparison_row(dataset_name, method_name, cell_record, paper_row):
    winner = cell_record["winner"]
    test_metrics = cell_record["test_metrics"]
    our_lv = winner["lv_count"]
    method_label = format_method_label(method_name, winner["preprocessing_specification"])
    acceptance = assess_acceptance_for_cell(paper_row, test_metrics, our_lv)
    return {
        "dataset": dataset_name,
        "method": method_label,
        "lv_paper": paper_row["factors"],
        "lv_ours": our_lv,
        "rmse_paper": paper_row["rmse"],
        "rmse_ours": test_metrics["rmse"],
        "rmse_rel_diff": acceptance["rmse_relative_diff"],
        "r2_paper": paper_row["r2"],
        "r2_ours": test_metrics["r2"],
        "r2_abs_diff": acceptance["r2_absolute_diff"],
        "mbd_paper": paper_row["mbd"],
        "mbd_ours": test_metrics["mbd"],
        "mbd_abs_diff": acceptance["mbd_absolute_diff"],
        "rpiq_paper": paper_row["rpiq"],
        "rpiq_ours": test_metrics["rpiq"],
        "lv_pass": acceptance["lv_pass"],
        "rmse_pass": acceptance["rmse_pass"],
        "r2_pass": acceptance["r2_pass"],
        "mbd_pass": acceptance["mbd_pass"],
        "all_pass": acceptance["all_pass"],
    }


def build_comparison_row_without_paper_reference(dataset_name, method_name, cell_record):
    winner = cell_record["winner"]
    test_metrics = cell_record["test_metrics"]
    method_label = format_method_label(method_name, winner["preprocessing_specification"])
    return {
        "dataset": dataset_name,
        "method": method_label,
        "lv_paper": float("nan"),
        "lv_ours": winner["lv_count"],
        "rmse_paper": float("nan"),
        "rmse_ours": test_metrics["rmse"],
        "rmse_rel_diff": float("nan"),
        "r2_paper": float("nan"),
        "r2_ours": test_metrics["r2"],
        "r2_abs_diff": float("nan"),
        "mbd_paper": float("nan"),
        "mbd_ours": test_metrics["mbd"],
        "mbd_abs_diff": float("nan"),
        "rpiq_paper": float("nan"),
        "rpiq_ours": test_metrics["rpiq"],
        "lv_pass": False,
        "rmse_pass": False,
        "r2_pass": False,
        "mbd_pass": False,
        "all_pass": False,
    }


def collect_all_comparison_rows():
    rows = []
    for dataset_name in DATASET_NAMES:
        for method_name in METHOD_NAMES:
            cell_record = load_one_cell_record(dataset_name, method_name)
            if cell_record is None:
                continue
            paper_row = PAPER_TABLE_1.get((dataset_name, method_name))
            if paper_row is None:
                rows.append(build_comparison_row_without_paper_reference(dataset_name, method_name, cell_record))
            else:
                rows.append(build_comparison_row(dataset_name, method_name, cell_record, paper_row))
    return rows


def print_comparison_table(comparison_dataframe):
    formatters = {
        "rmse_paper": "{:.3f}".format,
        "rmse_ours": "{:.3f}".format,
        "rmse_rel_diff": "{:.1%}".format,
        "r2_paper": "{:.3f}".format,
        "r2_ours": "{:.3f}".format,
        "r2_abs_diff": "{:.3f}".format,
        "mbd_paper": "{:+.3f}".format,
        "mbd_ours": "{:+.3f}".format,
        "mbd_abs_diff": "{:.3f}".format,
        "rpiq_paper": "{:.2f}".format,
        "rpiq_ours": "{:.2f}".format,
    }
    print(comparison_dataframe.to_string(index=False, formatters=formatters))


def main():
    rows = collect_all_comparison_rows()
    if not rows:
        print("No per-cell results found.")
        return
    comparison_dataframe = pd.DataFrame(rows)
    print(f"\nLoaded {len(rows)} cell results.\n")
    print_comparison_table(comparison_dataframe)
    output_path = RESULTS_DIR / "table1_replication.csv"
    comparison_dataframe.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")
    n_full_pass = int(comparison_dataframe["all_pass"].sum())
    print(f"\nAcceptance summary: {n_full_pass} / {len(comparison_dataframe)} cells pass all four criteria.")


if __name__ == "__main__":
    main()
