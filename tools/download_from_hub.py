"""
download_from_hub.py — CLI entry point for downloading SGG-Benchmark datasets
from Hugging Face Hub.

Usage
-----
    # Download only annotation JSONs (fastest — requires images already present)
    python tools/download_from_hub.py --dataset PSG

    # Download annotations AND images (large!)
    python tools/download_from_hub.py --dataset PSG --save-images

    # Override the local output directory
    python tools/download_from_hub.py --dataset VG150 --output-dir /data/VG150_coco

The actual download logic lives in sgg_benchmark/data/datasets/download.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sgg_benchmark.data.datasets.download import download_dataset, DATASET_CONFIGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download an SGG-Benchmark dataset from Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        required=True,
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory where the dataset will be saved. "
             "Defaults to the standard SGG-Benchmark path for each dataset.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also download and save images from the Hub dataset. "
             "This can be very large (tens of GB). "
             "By default only annotation JSONs are saved.",
    )
    args = parser.parse_args()

    download_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        save_images=args.save_images,
    )
