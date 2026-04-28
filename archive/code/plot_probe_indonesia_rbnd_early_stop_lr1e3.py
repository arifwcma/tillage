from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CURVES_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_early_stop_lr1e3_curves.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_early_stop_lr1e3_summary.csv"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "results" / "probe_indonesia_rbnd_early_stop_lr1e3_test_rmse.png"

PLSR_INDONESIA_TEST_RMSE = 1.1328


def main():
    curves_dataframe = pd.read_csv(CURVES_PATH)
    summary_dataframe = pd.read_csv(SUMMARY_PATH)
    seeds_in_run = sorted(curves_dataframe["seed"].unique().tolist())

    figure, axes_grid = plt.subplots(1, len(seeds_in_run), figsize=(6 * len(seeds_in_run), 4.6),
                                     sharey=True)
    if len(seeds_in_run) == 1:
        axes_grid = [axes_grid]

    for axis, seed in zip(axes_grid, seeds_in_run):
        seed_curves = curves_dataframe[curves_dataframe["seed"] == seed]
        seed_summary = summary_dataframe[summary_dataframe["seed"] == seed].iloc[0]

        axis.plot(seed_curves["epoch"], seed_curves["val_rmse"],
                  label="val RMSE", color="tab:orange", linewidth=1.4)
        axis.plot(seed_curves["epoch"], seed_curves["test_rmse"],
                  label="test RMSE", color="tab:blue", linewidth=1.4)
        axis.axhline(PLSR_INDONESIA_TEST_RMSE, color="tab:red", linestyle="--",
                     linewidth=1.2, label=f"PLSR test RMSE ({PLSR_INDONESIA_TEST_RMSE:.4f})")
        axis.axvline(seed_summary["best_val_epoch"], color="tab:green", linestyle=":",
                     linewidth=1.2, label=f"best-val epoch ({int(seed_summary['best_val_epoch'])})")
        axis.axvline(seed_summary["stopped_at_epoch"], color="grey", linestyle=":",
                     linewidth=1.0, label=f"stopped epoch ({int(seed_summary['stopped_at_epoch'])})")

        axis.set_title(
            f"seed={seed}  |  test RMSE @ best-val = {seed_summary['test_rmse_at_best']:.4f}"
        )
        axis.set_xlabel("epoch")
        axis.set_ylabel("RMSE")
        axis.set_ylim(0.0, max(3.0, float(seed_curves["val_rmse"].max()) * 1.05))
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)

    figure.suptitle(
        "Indonesia / none  |  rbnd FULL-batch, lr=1e-3, dropout=0.3  |  early stop (patience=20)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {PLOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
