from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
SWEEP_CSV_PATH = PROJECT_ROOT / "results" / "sweep_indonesia_rbnd_lr1e3_l1l2_curves.csv"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "sweep_indonesia_rbnd_lr1e3_l1l2_test_rmse.png"
OUTPUT_FIGURE_ZOOM_PATH = PROJECT_ROOT / "results" / "sweep_indonesia_rbnd_lr1e3_l1l2_test_rmse_zoom.png"

PLSR_INDONESIA_TEST_RMSE = 1.1328

CONFIGURATION_COLORS = {
    "l1=0, l2=0":       "#1f77b4",
    "l1=1e-4, l2=1e-3": "#ff7f0e",
    "l1=1e-3, l2=1e-2": "#2ca02c",
    "l1=1e-2, l2=1e-1": "#d62728",
    "l1=0, l2=1e-1":    "#9467bd",
    "l1=1e-2, l2=0":    "#8c564b",
}

PLSR_LINE_COLOR = "#888888"


def load_sweep_dataframe():
    return pd.read_csv(SWEEP_CSV_PATH)


def plot_all_configurations(ax, sweep_dataframe, x_max=None):
    for label, color in CONFIGURATION_COLORS.items():
        configuration_subset = sweep_dataframe[sweep_dataframe["label"] == label]
        if x_max is not None:
            configuration_subset = configuration_subset[configuration_subset["epoch"] <= x_max]
        ax.plot(configuration_subset["epoch"], configuration_subset["test_rmse"],
                color=color, linewidth=1.0, alpha=0.85, label=label)
    ax.axhline(PLSR_INDONESIA_TEST_RMSE, color=PLSR_LINE_COLOR, linestyle="--",
               linewidth=1.0, label=f"PLSR baseline ({PLSR_INDONESIA_TEST_RMSE:.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test RMSE (% SOC)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, ncol=2)


def plot_full_range(sweep_dataframe):
    figure, ax = plt.subplots(figsize=(11, 5))
    plot_all_configurations(ax, sweep_dataframe)
    figure.suptitle(
        "RBND test RMSE — Indonesia / none, lr=1e-3, dropout=0.3, FULL-batch=188, seed=42 (full range)"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(OUTPUT_FIGURE_PATH, dpi=150)
    plt.close(figure)
    print(f"Wrote {OUTPUT_FIGURE_PATH}")


def plot_zoom_first_300(sweep_dataframe):
    figure, ax = plt.subplots(figsize=(11, 5))
    plot_all_configurations(ax, sweep_dataframe, x_max=300)
    ax.set_ylim(0.85, 1.7)
    figure.suptitle(
        "RBND test RMSE — Indonesia / none, lr=1e-3 (zoom: epochs 1-300)"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(OUTPUT_FIGURE_ZOOM_PATH, dpi=150)
    plt.close(figure)
    print(f"Wrote {OUTPUT_FIGURE_ZOOM_PATH}")


def main():
    sweep_dataframe = load_sweep_dataframe()
    plot_full_range(sweep_dataframe)
    plot_zoom_first_300(sweep_dataframe)


if __name__ == "__main__":
    main()
