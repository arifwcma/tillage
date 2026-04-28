from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CURVE_CSV_PATH = PROJECT_ROOT / "results" / "probe_indonesia_curve.csv"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "results" / "probe_indonesia_curve_r2.png"

TRAIN_R2_COLOR = "#1f77b4"
TEST_R2_COLOR = "#d62728"
BEST_EPOCH_LINE_COLOR = "#2ca02c"


def load_curve_dataframe():
    return pd.read_csv(CURVE_CSV_PATH)


def find_best_test_r2_epoch(curve_dataframe):
    best_row = curve_dataframe.loc[curve_dataframe["test_r2"].idxmax()]
    return int(best_row["epoch"]), float(best_row["test_r2"]), float(best_row["train_r2"])


def plot_train_and_test_r2(ax, curve_dataframe):
    ax.plot(curve_dataframe["epoch"], curve_dataframe["train_r2"],
            color=TRAIN_R2_COLOR, linestyle="--", linewidth=1.0, label="train R²")
    ax.plot(curve_dataframe["epoch"], curve_dataframe["test_r2"],
            color=TEST_R2_COLOR, linestyle="-", linewidth=1.2, label="test R²")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R²")
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)


def annotate_best_test_r2_epoch(ax, best_epoch, best_test_r2, train_r2_at_best):
    ax.axvline(best_epoch, color=BEST_EPOCH_LINE_COLOR, linewidth=1.0, alpha=0.7)
    ax.annotate(
        f"best test R² @ epoch {best_epoch}\ntest R²={best_test_r2:.4f}\ntrain R²={train_r2_at_best:.4f}",
        xy=(best_epoch, best_test_r2),
        xytext=(best_epoch + 30, best_test_r2 - 0.3),
        color=BEST_EPOCH_LINE_COLOR,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": BEST_EPOCH_LINE_COLOR, "alpha": 0.7},
    )


def main():
    curve_dataframe = load_curve_dataframe()
    best_epoch, best_test_r2, train_r2_at_best = find_best_test_r2_epoch(curve_dataframe)

    figure, ax = plt.subplots(figsize=(11, 5))
    plot_train_and_test_r2(ax, curve_dataframe)
    annotate_best_test_r2_epoch(ax, best_epoch, best_test_r2, train_r2_at_best)

    figure.suptitle("RBN R² curve — Indonesia / none (bs=64, lr=1e-5, wd=0, seed=42)")
    figure.tight_layout(rect=[0, 0, 1, 0.95])

    figure.savefig(OUTPUT_FIGURE_PATH, dpi=150)
    plt.close(figure)
    print(f"Wrote {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
