"""
download_dataset.py - optional helper that downloads an animal dataset from Kaggle.

Kaggle requires a free account and an API token, so automatic download is optional.
If it does not work, follow the MANUAL steps printed below (they take about 5 minutes).

Datasets:
  animals90  - "Animal Image Dataset (90 Different Animals)", 90 classes, ~5,400 images (recommended)
  animals10  - "Animals-10", 10 classes, ~26,000 images (larger files, fewer animals)

Usage:
    pip install kaggle                 (only needed for automatic download)
    python download_dataset.py --dataset animals90
"""

import argparse
import sys
from pathlib import Path

DATASETS = {
    "animals90": {
        "slug": "iamsouravbanerjee/animal-image-dataset-90-different-animals",
        "url": "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals",
        "info": "90 animal classes, about 60 images each (roughly 650 MB).",
    },
    "animals10": {
        "slug": "alessiocorrado99/animals10",
        "url": "https://www.kaggle.com/datasets/alessiocorrado99/animals10",
        "info": "10 animal classes, about 26,000 images (roughly 600 MB). Folder names are Italian.",
    },
}

RAW_DIR = Path(__file__).resolve().parent / "dataset" / "raw"


def print_manual_steps(key):
    d = DATASETS[key]
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD STEPS")
    print("=" * 70)
    print(f"1. Create a free account at https://www.kaggle.com and sign in.")
    print(f"2. Open: {d['url']}")
    print(f"   ({d['info']})")
    print("3. Click 'Download' and save the zip file.")
    print(f"4. Extract the zip into:  {RAW_DIR}")
    print("5. Then run:")
    print(f"     python prepare_dataset.py --source \"{RAW_DIR}\" --output dataset --clean")
    print("6. Then train:")
    print("     python train.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download an animal dataset from Kaggle.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="animals90")
    args = parser.parse_args()
    d = DATASETS[args.dataset]

    print(f"Dataset: {args.dataset} - {d['info']}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Importing 'kaggle' authenticates immediately and raises if no token is configured.
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except ImportError:
        print("\nThe 'kaggle' package is not installed (pip install kaggle).")
        print_manual_steps(args.dataset)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - missing/invalid token etc.
        print(f"\nKaggle authentication failed: {exc}")
        print("Create a token at https://www.kaggle.com/settings -> 'Create New Token' and put")
        print("kaggle.json in C:\\Users\\<your-name>\\.kaggle\\  - or use the manual steps below.")
        print_manual_steps(args.dataset)
        sys.exit(1)

    try:
        print(f"Downloading to {RAW_DIR} (this can take several minutes)...")
        api.dataset_download_files(d["slug"], path=str(RAW_DIR), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001
        print(f"\nDownload failed: {exc}")
        print_manual_steps(args.dataset)
        sys.exit(1)

    print("\nDownload complete.")
    print("Next:")
    print(f"  python prepare_dataset.py --source \"{RAW_DIR}\" --output dataset --clean")
    print("  python train.py")


if __name__ == "__main__":
    main()
