from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "preprocessed_spectra.png"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]

LEFT_AXIS_METHODS = ["none", "snv", "msc", "sg"]
RIGHT_AXIS_METHOD = "sgd"

METHOD_COLORS = {
    "none": "#222222",
    "snv":  "#1f77b4",
    "msc":  "#2ca02c",
    "sg":   "#ff7f0e",
    "sgd":  "#d62728",
}

GROUP_KEY_COLUMN = "Batch and labid"
SOC_COLUMN = "Org C"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m(\d+(?:\.\d+)?)$")


def parse_wavenumber_columns(column_names):
    parsed_pairs = []
    for column_name in column_names:
        match_object = WAVENUMBER_COLUMN_PATTERN.match(column_name)
        if match_object is None:
            continue
        parsed_pairs.append((column_name, float(match_object.group(1))))
    return parsed_pairs


def find_median_soc_sample_id(dataset_name):
    none_train_path = PREPROCESSED_DIR / f"{dataset_name}_none_train.csv"
    table = pd.read_csv(none_train_path, low_memory=False, usecols=[GROUP_KEY_COLUMN, SOC_COLUMN])
    sorted_table = table.sort_values(SOC_COLUMN).reset_index(drop=True)
    median_position = len(sorted_table) // 2
    return sorted_table.loc[median_position, GROUP_KEY_COLUMN]


def load_one_sample_spectrum(dataset_name, method_name, sample_id):
    file_path = PREPROCESSED_DIR / f"{dataset_name}_{method_name}_train.csv"
    table = pd.read_csv(file_path, low_memory=False)
    matched_rows = table.loc[table[GROUP_KEY_COLUMN] == sample_id]
    first_row = matched_rows.iloc[0]
    wavenumber_pairs = parse_wavenumber_columns(table.columns)
    spectra_columns = [name for name, _ in wavenumber_pairs]
    wavenumber_values = np.array([wn for _, wn in wavenumber_pairs])
    spectrum_values = first_row[spectra_columns].to_numpy(dtype=np.float64)
    return wavenumber_values, spectrum_values


def draw_one_region_panel(panel_axes, dataset_name):
    sample_id = find_median_soc_sample_id(dataset_name)
    panel_axes.set_title(f"{dataset_name}  (sample {sample_id})", fontsize=10, fontweight="bold")

    plotted_lines = []
    plotted_labels = []

    for method_name in LEFT_AXIS_METHODS:
        wavenumber_values, spectrum_values = load_one_sample_spectrum(dataset_name, method_name, sample_id)
        line, = panel_axes.plot(
            wavenumber_values,
            spectrum_values,
            color=METHOD_COLORS[method_name],
            label=method_name,
            linewidth=1.0,
        )
        plotted_lines.append(line)
        plotted_labels.append(method_name)

    panel_axes.set_xlim(4000, 600)
    panel_axes.set_xlabel("wavenumber (cm$^{-1}$)")
    panel_axes.set_ylabel("absorbance / SNV / MSC / SG (left axis)")
    panel_axes.grid(True, alpha=0.25)

    twin_axes = panel_axes.twinx()
    sgd_wavenumber_values, sgd_spectrum_values = load_one_sample_spectrum(
        dataset_name, RIGHT_AXIS_METHOD, sample_id
    )
    sgd_line, = twin_axes.plot(
        sgd_wavenumber_values,
        sgd_spectrum_values,
        color=METHOD_COLORS[RIGHT_AXIS_METHOD],
        label=RIGHT_AXIS_METHOD,
        linewidth=0.8,
        alpha=0.9,
    )
    twin_axes.set_ylabel("SGD (right axis)", color=METHOD_COLORS[RIGHT_AXIS_METHOD])
    twin_axes.tick_params(axis="y", labelcolor=METHOD_COLORS[RIGHT_AXIS_METHOD])
    plotted_lines.append(sgd_line)
    plotted_labels.append(RIGHT_AXIS_METHOD)

    panel_axes.legend(plotted_lines, plotted_labels, loc="upper right", fontsize=8, frameon=False)


def main():
    OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes_grid = plt.subplots(2, 2, figsize=(13, 8.5))
    flattened_axes = axes_grid.flatten()
    for panel_index, dataset_name in enumerate(DATASET_NAMES):
        draw_one_region_panel(flattened_axes[panel_index], dataset_name)
    figure.suptitle(
        "Median-SOC sample under each preprocessing — actual values; SGD on right axis",
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(OUTPUT_FIGURE_PATH, dpi=140)
    print(f"wrote {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
