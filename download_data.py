from pathlib import Path
import hashlib
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
ADDITIONALS_DIR = PROJECT_ROOT / "additionals"

DATASET_DOWNLOAD_URL = "https://files.isric.org/public/other/WD-ICRAF-Spectral_MIR.zip"
DATASET_ZIP_FILENAME = "WD-ICRAF-Spectral_MIR.zip"
EXPECTED_SHA256 = "5ff3e43fd5bdfdc0a051f34b9b23891790ce2e10168123ae3fa403a8be45ff8f"
EXPECTED_BYTES = 59_327_817

EXTRACTED_FOLDER_NAME = "WD-ICRAF-Spectral_MIR"


def compute_sha256_of_file(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def download_zip_if_missing(zip_path):
    if zip_path.exists():
        print(f"  zip already present: {zip_path}")
        return
    print(f"  downloading from {DATASET_DOWNLOAD_URL}")
    urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, zip_path)
    print(f"  saved to {zip_path}")


def verify_zip_integrity(zip_path):
    actual_size = zip_path.stat().st_size
    if actual_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"zip size mismatch: got {actual_size} bytes, expected {EXPECTED_BYTES}"
        )
    actual_sha256 = compute_sha256_of_file(zip_path)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"zip sha256 mismatch:\n  got      {actual_sha256}\n  expected {EXPECTED_SHA256}"
        )
    print(f"  zip integrity verified (sha256 ok, {actual_size:,} bytes)")


def extract_zip_if_missing(zip_path, extraction_target_dir):
    extracted_subfolder = extraction_target_dir / EXTRACTED_FOLDER_NAME
    if extracted_subfolder.exists():
        print(f"  already extracted: {extracted_subfolder}")
        return
    print(f"  extracting {zip_path.name} to {extraction_target_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extraction_target_dir)
    print(f"  extraction complete")


def main():
    ADDITIONALS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ADDITIONALS_DIR / DATASET_ZIP_FILENAME

    print("Step 1/3: download dataset zip")
    download_zip_if_missing(zip_path)

    print("Step 2/3: verify integrity")
    verify_zip_integrity(zip_path)

    print("Step 3/3: extract zip")
    extract_zip_if_missing(zip_path, ADDITIONALS_DIR)

    print("\nDataset is ready at:")
    print(f"  {ADDITIONALS_DIR / EXTRACTED_FOLDER_NAME}")


if __name__ == "__main__":
    main()
