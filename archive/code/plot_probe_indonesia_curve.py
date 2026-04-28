from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CURVE_CSV_PATH = PROJECT_ROOT / "results" / "probe_indonesia_curve.csv"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "probe_indonesia_curve.png"

RMSE_COLOR = "#1f77b4"
R2_COLOR = "#d62728"
BEST_EPOCH_LINE_COLOR = "#2ca02c"


def load_curve_dataframe():
    return pd.read_csv(CURVE_CSV_PATH)


def find_best_test_rmse_epoch(curve_dataframe):
    best_row = curve_dataframe.loc[curve_dataframe["test_rmse"].idxmin()]
    return int(best_row["epoch"]), float(best_row["test_rmse"]), float(best_row["test_r2"])


def plot_rmse_axis(ax, curve_dataframe):
    ax.plot(curve_dataframe["epoch"], curve_dataframe["train_rmse"],
            color=RMSE_COLOR, linestyle="--", linewidth=1.0, label="train RMSE")
    ax.plot(curve_dataframe["epoch"], curve_dataframe["test_rmse"],
            color=RMSE_COLOR, linestyle="-", linewidth=1.4, label="test RMSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE (% SOC)", color=RMSE_COLOR)
    ax.tick_params(axis="y", labelcolor=RMSE_COLOR)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def plot_r2_axis(ax, curve_dataframe):
    ax.plot(curve_dataframe["epoch"], curve_dataframe["train_r2"],
            color=R2_COLOR, linestyle="--", linewidth=1.0, label="train R²")
    ax.plot(curve_dataframe["epoch"], curve_dataframe["test_r2"],
            color=R2_COLOR, linestyle="-", linewidth=1.4, label="test R²")
    ax.set_ylabel("R²", color=R2_COLOR)
    ax.tick_params(axis="y", labelcolor=R2_COLOR)
    ax.set_ylim(-0.2, 1.05)


def annotate_best_test_epoch(ax_left, best_epoch, best_test_rmse, best_test_r2):
    ax_left.axvline(best_epoch, color=BEST_EPOCH_LINE_COLOR, linewidth=1.0, alpha=0.7)
    ax_left.annotate(
        f"best test\nepoch {best_epoch}\nRMSE={best_test_rmse:.4f}\nR²={best_test_r2:.4f}",
        xy=(best_epoch, best_test_rmse),
        xytext=(best_epoch + 30, best_test_rmse + 0.4),
        color=BEST_EPOCH_LINE_COLOR,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": BEST_EPOCH_LINE_COLOR, "alpha": 0.7},
    )


def combine_legends(ax_left, ax_right):
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=9)


def main():
    curve_dataframe = load_curve_dataframe()
    best_epoch, best_test_rmse, best_test_r2 = find_best_test_rmse_epoch(curve_dataframe)

    figure, ax_left = plt.subplots(figsize=(11, 5))
    ax_right = ax_left.twinx()

    plot_rmse_axis(ax_left, curve_dataframe)
    plot_r2_axis(ax_right, curve_dataframe)
    annotate_best_test_epoch(ax_left, best_epoch, best_test_rmse, best_test_r2)
    combine_legends(ax_left, ax_right)

    figure.suptitle("RBN learning curve — Indonesia / none (bs=64, lr=1e-5, wd=0, seed=42)")
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    figure.savefig(OUTPUT_FIGURE_PATH, dpi=150)
    plt.close(figure)
    print(f"Wrote {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
