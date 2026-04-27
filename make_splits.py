from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

DATASET_FILE_NAMES = ["global.csv", "china.csv", "kenya.csv", "indonesia.csv"]

GROUP_KEY_COLUMN = "Batch and labid"
SOC_COLUMN = "Org C"
N_QUARTILES = 4
N_FOLDS_FOR_80_20_SPLIT = 5
RANDOM_SEED = 42

EXPECTED_TEST_FRACTION_TOLERANCE = 0.03


def assign_soc_quartile(soc_values):
    quartile_labels = pd.qcut(soc_values, q=N_QUARTILES, labels=False, duplicates="drop")
    return quartile_labels.astype(int)


def split_into_train_and_test(dataset, group_values, stratum_values):
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS_FOR_80_20_SPLIT,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    train_indices, test_indices = next(splitter.split(dataset, stratum_values, group_values))
    fold_assignment = pd.Series(index=dataset.index, dtype="object")
    fold_assignment.iloc[train_indices] = "train"
    fold_assignment.iloc[test_indices] = "test"
    return fold_assignment


def assert_no_group_leakage_between_train_and_test(group_values, fold_assignment):
    paired_table = pd.DataFrame({"group": group_values.to_numpy(), "fold": fold_assignment.to_numpy()})
    folds_per_group = paired_table.groupby("group")["fold"].nunique()
    leaky_groups = folds_per_group[folds_per_group > 1]
    if len(leaky_groups) > 0:
        raise RuntimeError(f"group leakage: {len(leaky_groups)} groups appear in both train and test")


def assert_test_fraction_close_to_expected(fold_assignment):
    test_fraction = (fold_assignment == "test").mean()
    expected = 1.0 / N_FOLDS_FOR_80_20_SPLIT
    if abs(test_fraction - expected) > EXPECTED_TEST_FRACTION_TOLERANCE:
        raise RuntimeError(
            f"test fraction {test_fraction:.3f} too far from expected {expected:.3f}"
        )


def report_split_quality(dataset_label, fold_assignment, stratum_values):
    n_train = int((fold_assignment == "train").sum())
    n_test = int((fold_assignment == "test").sum())
    test_fraction = n_test / (n_train + n_test)
    print(f"  {dataset_label:>10}: train={n_train:>4}  test={n_test:>4}  test_frac={test_fraction:.3f}")
    paired = pd.DataFrame({"fold": fold_assignment.to_numpy(), "quartile": stratum_values})
    distribution = paired.groupby(["fold", "quartile"]).size().unstack(fill_value=0)
    print(distribution.to_string())


def make_split_for_one_dataset(raw_csv_path):
    dataset_label = raw_csv_path.stem
    print(f"\n--- {dataset_label} ---")
    dataset = pd.read_csv(raw_csv_path, usecols=[GROUP_KEY_COLUMN, SOC_COLUMN])
    print(f"  rows loaded: {len(dataset)}")

    stratum_values = assign_soc_quartile(dataset[SOC_COLUMN])
    group_values = dataset[GROUP_KEY_COLUMN]

    fold_assignment = split_into_train_and_test(dataset, group_values, stratum_values)

    assert_no_group_leakage_between_train_and_test(group_values, fold_assignment)
    assert_test_fraction_close_to_expected(fold_assignment)
    report_split_quality(dataset_label, fold_assignment, stratum_values)

    split_table = pd.DataFrame({
        GROUP_KEY_COLUMN: group_values,
        "fold": fold_assignment,
    })
    output_path = SPLITS_DIR / (dataset_label + "_split.csv")
    split_table.to_csv(output_path, index=False)
    print(f"  wrote {output_path}")


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_file_name in DATASET_FILE_NAMES:
        raw_csv_path = RAW_DIR / dataset_file_name
        make_split_for_one_dataset(raw_csv_path)
    print("\nAll splits written.")


if __name__ == "__main__":
    main()
