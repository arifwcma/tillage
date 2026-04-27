"""Plot mean reflectance of Indonesia train and test sets side-by-side."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_PATH = "data/preprocessed/indonesia_none_train.csv"
TEST_PATH = "data/preprocessed/indonesia_none_test.csv"
OUT_PATH = "results/indonesia_mean_reflectance.png"


def load_spectra(path: str):
    df = pd.read_csv(path)
    spec_cols = [c for c in df.columns if c.startswith("m") and c[1:].replace(".", "", 1).isdigit()]
    wavenumbers = np.array([float(c[1:]) for c in spec_cols])
    order = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[order]
    spectra = df[spec_cols].to_numpy()[:, order]
    return wavenumbers, spectra


def main() -> None:
    wn_tr, X_tr = load_spectra(TRAIN_PATH)
    wn_te, X_te = load_spectra(TEST_PATH)

    mean_tr = X_tr.mean(axis=0)
    std_tr = X_tr.std(axis=0)
    mean_te = X_te.mean(axis=0)
    std_te = X_te.std(axis=0)

    y_lo = min((mean_tr - std_tr).min(), (mean_te - std_te).min())
    y_hi = max((mean_tr + std_tr).max(), (mean_te + std_te).max())
    pad = 0.05 * (y_hi - y_lo)
    ylim = (y_lo - pad, y_hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].plot(wn_tr, mean_tr, color="C0", lw=1.2, label="mean")
    axes[0].fill_between(wn_tr, mean_tr - std_tr, mean_tr + std_tr, color="C0", alpha=0.25, label="±1 SD")
    axes[0].set_title(f"Indonesia train (n={X_tr.shape[0]})")
    axes[0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("Reflectance / Absorbance")
    axes[0].invert_xaxis()
    axes[0].legend(loc="best")

    axes[1].plot(wn_te, mean_te, color="C3", lw=1.2, label="mean")
    axes[1].fill_between(wn_te, mean_te - std_te, mean_te + std_te, color="C3", alpha=0.25, label="±1 SD")
    axes[1].set_title(f"Indonesia test (n={X_te.shape[0]})")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].invert_xaxis()
    axes[1].legend(loc="best")

    for ax in axes:
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Mean MIR spectra: Indonesia (none preprocessing)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
    print(f"Shared ylim: ({ylim[0]:.4f}, {ylim[1]:.4f})")


if __name__ == "__main__":
    main()
