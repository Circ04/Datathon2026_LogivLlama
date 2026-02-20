"""
Download and extract dataset from Google Drive.

This script downloads the 6 tar.gz files containing the dataset and extracts them
to the Data/ directory.
"""

import os
import gdown
import tarfile
from pathlib import Path

GOOGLE_DRIVE_FILES = [
    {"id": "1HvRxkOVNzLSM368XdcCHXkI41b4cTY0o", "name": "part1.tar.gz"},
    {"id": "1mWlmJeCWGYoU8IbgL6ZVdvvKjAkaNHY6", "name": "part2.tar.gz"},
    {"id": "1gvk7Gt1S9SRB2dpPOUEnq0RfByBqsLq_", "name": "part3.tar.gz"},
    {"id": "1_TU2cEigl9pHcJ1gsVnRoD9IT7Ojjs5v", "name": "part4.tar.gz"},
    {"id": "1G3aHT1fZWGrHFMZQ959eSAUqWhDHJNXG", "name": "part5.tar.gz"},
    {"id": "1hsy5xWLDqc-PP9JAqBb-VqAVpEwyc3rJ", "name": "part6.tar.gz"},
]

DATA_DIR = Path("Data")
TEMP_DIR = DATA_DIR / "temp_downloads"


def download_and_extract():
    """Download tar.gz files from Google Drive and extract them."""
    DATA_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    for i, file_info in enumerate(GOOGLE_DRIVE_FILES, 1):
        output_path = TEMP_DIR / file_info['name']

        # Skip if already downloaded
        if output_path.exists():
            print(f"\n[{i}/6] {file_info['name']} already downloaded, extracting...")
        else:
            print(f"\n[{i}/6] Downloading {file_info['name']}...")
            url = f"https://drive.google.com/uc?id={file_info['id']}"
            gdown.download(url, str(output_path), quiet=False)

        # Extract tar.gz
        print(f"Extracting {file_info['name']}...")
        with tarfile.open(output_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)

        # Clean up tar.gz file
        output_path.unlink()
        print(f"✓ Completed {file_info['name']}")

    # Remove temp directory if empty
    if TEMP_DIR.exists():
        try:
            TEMP_DIR.rmdir()
        except OSError:
            pass  # Directory not empty, leave it

    print("\n✓ All files downloaded and extracted successfully!")


if __name__ == "__main__":
    download_and_extract()
