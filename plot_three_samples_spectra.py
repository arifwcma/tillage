from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "three_samples_spectra.png"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
METHOD_NAMES = ["none", "snv", "msc", "sg", "sgd", "minmax"]

N_SAMPLES_PER_REGION = 3
RANDOM_SEED_FOR_SAMPLE_PICK = 42

SAMPLE_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]

GROUP_KEY_COLUMN = "Batch and labid"
WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m(\d+(?:\.\d+)?)$")


def parse_wavenumber_columns(column_names):
    parsed_pairs = []
    for column_name in column_names:
        match_object = WAVENUMBER_COLUMN_PATTERN.match(column_name)
        if match_object is None:
            continue
        parsed_pairs.append((column_name, float(match_object.group(1))))
    return parsed_pairs


def pick_three_sample_ids_for_region(dataset_name):
    none_train_path = PREPROCESSED_DIR / f"{dataset_name}_none_train.csv"
    table = pd.read_csv(none_train_path, low_memory=False, usecols=[GROUP_KEY_COLUMN])
    unique_sample_ids = table[GROUP_KEY_COLUMN].drop_duplicates().to_numpy()
    random_state = np.random.default_rng(RANDOM_SEED_FOR_SAMPLE_PICK)
    chosen_ids = random_state.choice(unique_sample_ids, size=N_SAMPLES_PER_REGION, replace=False)
    return list(chosen_ids)


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


def draw_one_cell(panel_axes, dataset_name, method_name, sample_ids_for_region, is_top_row, is_left_column):
    for sample_index, sample_id in enumerate(sample_ids_for_region):
        wavenumber_values, spectrum_values = load_one_sample_spectrum(
            dataset_name, method_name, sample_id
        )
        panel_axes.plot(
            wavenumber_values,
            spectrum_values,
            color=SAMPLE_COLORS[sample_index],
            label=str(sample_id),
            linewidth=1.0,
        )

    panel_axes.set_xlim(4000, 600)
    panel_axes.grid(True, alpha=0.2)

    if is_top_row:
        panel_axes.set_title(dataset_name, fontsize=14, fontweight="bold")
    if is_left_column:
        panel_axes.set_ylabel(method_name, fontsize=13, fontweight="bold")
    panel_axes.tick_params(axis="both", labelsize=8)
    panel_axes.legend(loc="upper right", fontsize=7, frameon=False)


def main():
    OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_ids_per_region = {}
    for dataset_name in DATASET_NAMES:
        sample_ids_per_region[dataset_name] = pick_three_sample_ids_for_region(dataset_name)

    n_rows = len(METHOD_NAMES)
    n_cols = len(DATASET_NAMES)
    figure, axes_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(22, 22),
        sharex=True,
        sharey="row",
    )

    for row_index, method_name in enumerate(METHOD_NAMES):
        for col_index, dataset_name in enumerate(DATASET_NAMES):
            is_top_row = row_index == 0
            is_left_column = col_index == 0
            panel_axes = axes_grid[row_index, col_index]
            draw_one_cell(
                panel_axes,
                dataset_name,
                method_name,
                sample_ids_per_region[dataset_name],
                is_top_row,
                is_left_column,
            )

    for col_index in range(n_cols):
        axes_grid[n_rows - 1, col_index].set_xlabel("wavenumber (cm$^{-1}$)", fontsize=11)

    figure.suptitle(
        "3 random train samples per region under each preprocessing — colour identifies the sample (consistent across rows)",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.985])
    figure.savefig(OUTPUT_FIGURE_PATH, dpi=120)
    print(f"wrote {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
