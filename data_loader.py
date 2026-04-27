from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "additionals" / "WD-ICRAF-Spectral_MIR" / "WD-ICRAF-Spectral_MIR"
REFERENCE_CSV_PATH = RAW_DATA_DIR / "ICRAF_ISRIC reference data.csv"
SPECTRA_CSV_PATH = RAW_DATA_DIR / "ICRAF_ISRIC MIR spectra.csv"

OUTPUT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

REFERENCE_KEY_COLUMN = "Batch and labid"
SPECTRA_KEY_COLUMN = "SSN"
SOC_COLUMN = "Org C"
COUNTRY_COLUMN = "Country name"

WAVENUMBER_LOWER_CM = 600.0
WAVENUMBER_UPPER_CM = 4000.0

REFERENCE_COLUMNS_TO_KEEP = [
    "Batch and labid",
    "ICRAF sample codes.SAMPLENO",
    "Country name",
    "Plotcode",
    "HORI",
    "BTOP",
    "BBOT",
    "N / S", "Lat: degr", "Lat: min", "Lat: sec",
    "E / W", "Long: degr", "Long: min", "Long: sec",
    "pH (H2O)",
    "Clay",
    "Org C",
]

COUNTRY_SUBSETS = ["China", "Kenya", "Indonesia"]

EXPECTED_ROW_COUNTS = {
    "Global": 3997,
    "China": 262,
    "Kenya": 245,
    "Indonesia": 236,
}
EXPECTED_WAVENUMBER_COLUMN_COUNT = 1763


def parse_wavenumber_from_column_name(column_name):
    match = re.match(r'^m(\d+(?:\.\d+)?)$', column_name)
    if match is None:
        return None
    return float(match.group(1))


def select_wavenumber_columns_in_window(spectra_columns, lower_cm, upper_cm):
    selected = []
    for column_name in spectra_columns:
        wavenumber = parse_wavenumber_from_column_name(column_name)
        if wavenumber is None:
            continue
        if lower_cm <= wavenumber <= upper_cm:
            selected.append(column_name)
    return selected


def load_reference_table():
    reference = pd.read_csv(REFERENCE_CSV_PATH)
    return reference


def load_spectra_table():
    spectra = pd.read_csv(SPECTRA_CSV_PATH)
    return spectra


def join_reference_and_spectra(reference, spectra):
    spectra_renamed = spectra.rename(columns={SPECTRA_KEY_COLUMN: REFERENCE_KEY_COLUMN})
    joined = reference.merge(spectra_renamed, on=REFERENCE_KEY_COLUMN, how="inner")
    return joined


def filter_to_non_missing_soc(joined):
    has_soc = joined[SOC_COLUMN].notna()
    return joined.loc[has_soc].reset_index(drop=True)


def slice_to_target_wavenumber_window(joined_with_soc, spectra_columns_in_window):
    final_columns = REFERENCE_COLUMNS_TO_KEEP + spectra_columns_in_window
    return joined_with_soc[final_columns].copy()


def build_country_subset(global_dataset, country_name):
    is_in_country = global_dataset[COUNTRY_COLUMN] == country_name
    return global_dataset.loc[is_in_country].reset_index(drop=True)


def report_dataset_shape(label, dataset):
    n_rows = len(dataset)
    n_cols = dataset.shape[1]
    print(f"  {label:>10}: rows={n_rows:>5}  cols={n_cols}")


def main():
    print("Loading reference table...")
    reference = load_reference_table()
    print(f"  reference shape: {reference.shape}")

    print("Loading spectra table...")
    spectra = load_spectra_table()
    print(f"  spectra shape:   {spectra.shape}")

    print("Joining reference and spectra (keeping all duplicate rows to match paper n)...")
    joined = join_reference_and_spectra(reference, spectra)
    print(f"  joined shape:    {joined.shape}")

    print("Filtering to non-missing SOC...")
    joined_with_soc = filter_to_non_missing_soc(joined)
    print(f"  rows with non-missing OC: {len(joined_with_soc)}")

    print(f"Selecting wavenumber columns in [{WAVENUMBER_LOWER_CM}, {WAVENUMBER_UPPER_CM}] cm-1...")
    all_spectra_column_names = [c for c in spectra.columns if c != SPECTRA_KEY_COLUMN]
    spectra_columns_in_window = select_wavenumber_columns_in_window(
        all_spectra_column_names,
        WAVENUMBER_LOWER_CM,
        WAVENUMBER_UPPER_CM,
    )
    print(f"  total spectra columns: {len(all_spectra_column_names)}")
    print(f"  columns kept (4000-600 cm-1): {len(spectra_columns_in_window)}")
    print(f"  first kept: {spectra_columns_in_window[0]}  last kept: {spectra_columns_in_window[-1]}")

    global_dataset = slice_to_target_wavenumber_window(joined_with_soc, spectra_columns_in_window)

    country_datasets = {}
    for country_name in COUNTRY_SUBSETS:
        country_datasets[country_name] = build_country_subset(global_dataset, country_name)

    print("\nFinal row counts:")
    report_dataset_shape("Global", global_dataset)
    for country_name in COUNTRY_SUBSETS:
        report_dataset_shape(country_name, country_datasets[country_name])

    print("\nVerifying counts match paper Table 1 / supplementary Table S1...")
    assert len(spectra_columns_in_window) == EXPECTED_WAVENUMBER_COLUMN_COUNT, (
        f"wavenumber column count {len(spectra_columns_in_window)} != expected {EXPECTED_WAVENUMBER_COLUMN_COUNT}"
    )
    assert len(global_dataset) == EXPECTED_ROW_COUNTS["Global"], (
        f"Global row count {len(global_dataset)} != expected {EXPECTED_ROW_COUNTS['Global']}"
    )
    for country_name in COUNTRY_SUBSETS:
        actual = len(country_datasets[country_name])
        expected = EXPECTED_ROW_COUNTS[country_name]
        assert actual == expected, (
            f"{country_name} row count {actual} != expected {expected}"
        )
    print("  all counts match.")

    OUTPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting raw CSVs to {OUTPUT_RAW_DIR} ...")
    global_dataset.to_csv(OUTPUT_RAW_DIR / "global.csv", index=False)
    for country_name in COUNTRY_SUBSETS:
        file_name = country_name.lower() + ".csv"
        country_datasets[country_name].to_csv(OUTPUT_RAW_DIR / file_name, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
