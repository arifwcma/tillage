from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

WAVENUMBER_COLUMN_PATTERN = re.compile(r"^m\d+(?:\.\d+)?$")
DATASET_NAME_FOR_AUDIT = "global"
SNV_PER_ROW_TOLERANCE = 1e-9


def load_spectra_only(file_path):
    table = pd.read_csv(file_path, low_memory=False)
    spectra_columns = [c for c in table.columns if WAVENUMBER_COLUMN_PATTERN.match(c)]
    return table[spectra_columns].to_numpy(dtype=np.float64)


def audit_none_versus_snv(none_spectra, snv_spectra):
    snv_row_means = snv_spectra.mean(axis=1)
    snv_row_stds = snv_spectra.std(axis=1, ddof=1)
    print(f"  SNV row mean range: [{snv_row_means.min():.2e}, {snv_row_means.max():.2e}] (target ~0)")
    print(f"  SNV row std  range: [{snv_row_stds.min():.4f}, {snv_row_stds.max():.4f}] (target ~1)")
    if abs(snv_row_means).max() > SNV_PER_ROW_TOLERANCE:
        print("  WARN: SNV per-row mean not at zero")
    if abs(snv_row_stds - 1.0).max() > 1e-6:
        print("  WARN: SNV per-row std not at one")


def audit_msc(none_spectra, msc_spectra):
    print(f"  raw row range: [{none_spectra.min():.4f}, {none_spectra.max():.4f}]")
    print(f"  MSC row range: [{msc_spectra.min():.4f}, {msc_spectra.max():.4f}]")
    print(f"  shape match: {none_spectra.shape == msc_spectra.shape}")


def audit_sg_versus_none(none_spectra, sg_spectra):
    diff = sg_spectra - none_spectra
    print(f"  SG-None diff: max abs = {abs(diff).max():.6f}, mean abs = {abs(diff).mean():.6f}")


def audit_sgd_first_row_signature(none_spectra, sgd_spectra):
    print(f"  SGD raw range: [{sgd_spectra.min():.4e}, {sgd_spectra.max():.4e}] (should be small around zero)")
    print(f"  SGD mean: {sgd_spectra.mean():.4e}")


def main():
    none_train = load_spectra_only(PREPROCESSED_DIR / f"{DATASET_NAME_FOR_AUDIT}_none_train.csv")
    snv_train = load_spectra_only(PREPROCESSED_DIR / f"{DATASET_NAME_FOR_AUDIT}_snv_train.csv")
    msc_train = load_spectra_only(PREPROCESSED_DIR / f"{DATASET_NAME_FOR_AUDIT}_msc_train.csv")
    sg_train = load_spectra_only(PREPROCESSED_DIR / f"{DATASET_NAME_FOR_AUDIT}_sg_train.csv")
    sgd_train = load_spectra_only(PREPROCESSED_DIR / f"{DATASET_NAME_FOR_AUDIT}_sgd_train.csv")

    print(f"audit on dataset: {DATASET_NAME_FOR_AUDIT} (train, n={len(none_train)})")
    print("\n[SNV]")
    audit_none_versus_snv(none_train, snv_train)
    print("\n[MSC]")
    audit_msc(none_train, msc_train)
    print("\n[SG]")
    audit_sg_versus_none(none_train, sg_train)
    print("\n[SGD]")
    audit_sgd_first_row_signature(none_train, sgd_train)


if __name__ == "__main__":
    main()
