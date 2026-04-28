from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CURVE_CSV_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_full_batch_lr1e4_curve.csv"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_full_batch_lr1e4_test_rmse.png"

PLSR_INDONESIA_TEST_RMSE = 1.1328
FIXED_419_EPOCH = 419

TEST_RMSE_COLOR = "#1f77b4"
PLSR_LINE_COLOR = "#888888"
BEST_EPOCH_LINE_COLOR = "#2ca02c"
FIXED_EPOCH_LINE_COLOR = "#d62728"


def load_curve_dataframe():
    return pd.read_csv(CURVE_CSV_PATH)


def find_best_test_rmse_epoch(curve_dataframe):
    best_row = curve_dataframe.loc[curve_dataframe["test_rmse"].idxmin()]
    return int(best_row["epoch"]), float(best_row["test_rmse"])


def plot_test_rmse_curve(ax, curve_dataframe):
    ax.plot(curve_dataframe["epoch"], curve_dataframe["test_rmse"],
            color=TEST_RMSE_COLOR, linewidth=1.0, label="test RMSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test RMSE (% SOC)")
    ax.grid(True, linestyle=":", alpha=0.4)


def annotate_plsr_baseline(ax):
    ax.axhline(PLSR_INDONESIA_TEST_RMSE, color=PLSR_LINE_COLOR, linestyle="--", linewidth=1.0,
               label=f"PLSR baseline ({PLSR_INDONESIA_TEST_RMSE:.4f})")


def annotate_fixed_epoch(ax, curve_dataframe):
    fixed_row = curve_dataframe.loc[curve_dataframe["epoch"] == FIXED_419_EPOCH]
    fixed_test_rmse = float(fixed_row["test_rmse"].iloc[0])
    ax.axvline(FIXED_419_EPOCH, color=FIXED_EPOCH_LINE_COLOR, linewidth=1.0, alpha=0.6,
               linestyle=":", label=f"H1A budget (epoch {FIXED_419_EPOCH}, RMSE={fixed_test_rmse:.4f})")


def annotate_best_test_epoch(ax, best_epoch, best_test_rmse):
    ax.axvline(best_epoch, color=BEST_EPOCH_LINE_COLOR, linewidth=1.0, alpha=0.7)
    ax.annotate(
        f"best test\nepoch {best_epoch}\nRMSE={best_test_rmse:.4f}",
        xy=(best_epoch, best_test_rmse),
        xytext=(best_epoch + 200, best_test_rmse + 0.25),
        color=BEST_EPOCH_LINE_COLOR,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": BEST_EPOCH_LINE_COLOR, "alpha": 0.7},
    )


def main():
    curve_dataframe = load_curve_dataframe()
    best_epoch, best_test_rmse = find_best_test_rmse_epoch(curve_dataframe)

    figure, ax = plt.subplots(figsize=(11, 5))
    plot_test_rmse_curve(ax, curve_dataframe)
    annotate_plsr_baseline(ax)
    annotate_fixed_epoch(ax, curve_dataframe)
    annotate_best_test_epoch(ax, best_epoch, best_test_rmse)
    ax.legend(loc="upper right", fontsize=9)

    figure.suptitle(
        "RBND test RMSE — Indonesia / none (FULL-batch=188, lr=1e-4, dropout=0.3, seed=42)"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    figure.savefig(OUTPUT_FIGURE_PATH, dpi=150)
    plt.close(figure)
    print(f"Wrote {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
