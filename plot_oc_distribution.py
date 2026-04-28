from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "oc_distribution.png"

DATASET_NAMES = ["global", "china", "kenya", "indonesia"]
SOC_COLUMN = "Org C"
N_HISTOGRAM_BINS = 40

DATASET_COLORS = {
    "global":    "#222222",
    "china":     "#d62728",
    "kenya":     "#2ca02c",
    "indonesia": "#1f77b4",
}


def load_oc_values(dataset_name):
    csv_path = RAW_DIR / f"{dataset_name}.csv"
    return pd.read_csv(csv_path, usecols=[SOC_COLUMN], low_memory=False)[SOC_COLUMN].to_numpy(dtype=float)


def draw_one_panel(panel_axes, dataset_name):
    oc_values = load_oc_values(dataset_name)
    sample_count = len(oc_values)
    median_value = float(np.median(oc_values))
    quartile_one = float(np.percentile(oc_values, 25))
    quartile_three = float(np.percentile(oc_values, 75))
    maximum_value = float(oc_values.max())
    minimum_value = float(oc_values.min())

    fill_color = DATASET_COLORS[dataset_name]

    panel_axes.hist(
        oc_values,
        bins=N_HISTOGRAM_BINS,
        color=fill_color,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )
    panel_axes.axvline(median_value, color="black", linewidth=1.2, linestyle="-", label=f"median = {median_value:.2f}")
    panel_axes.axvline(quartile_one, color="black", linewidth=0.8, linestyle="--", label=f"Q1 = {quartile_one:.2f}")
    panel_axes.axvline(quartile_three, color="black", linewidth=0.8, linestyle="--", label=f"Q3 = {quartile_three:.2f}")

    panel_axes.set_title(
        f"{dataset_name}  (n = {sample_count}, range = {minimum_value:.2f}–{maximum_value:.2f}%)",
        fontsize=12,
        fontweight="bold",
        color=fill_color,
    )
    panel_axes.set_xlabel("Org C (%)", fontsize=10)
    panel_axes.set_ylabel("count", fontsize=10)
    panel_axes.grid(True, alpha=0.2)
    panel_axes.legend(fontsize=8, loc="upper right")
    panel_axes.tick_params(axis="both", labelsize=9)


def main():
    OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes_grid = plt.subplots(2, 2, figsize=(13, 9))
    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for dataset_name, (row_index, column_index) in zip(DATASET_NAMES, panel_positions):
        draw_one_panel(axes_grid[row_index, column_index], dataset_name)

    figure.suptitle(
        "Organic carbon (Org C %) distribution per region",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(OUTPUT_FIGURE_PATH, dpi=120)
    print(f"wrote {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
