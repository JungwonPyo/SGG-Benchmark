"""
download_from_hub.py — Download an SGG-Benchmark dataset from Hugging Face Hub
and reconstruct the local COCO-format directory structure expected by the
SGG-Benchmark codebase.

Usage
-----
    # Download only annotation JSONs (fastest — requires images already present)
    python tools/download_from_hub.py --dataset PSG

    # Download annotations AND images (large!)
    python tools/download_from_hub.py --dataset PSG --save-images

    # Override the local output directory
    python tools/download_from_hub.py --dataset VG150 --output-dir /data/VG150_coco

Datasets available
------------------
    PSG       → maelic/PSG-coco-format
                Images : COCO train2017 + val2017 (VG images subset)
                Output : datasets/PSG/coco_format/{train,val,test}/

    VG150     → maelic/VG150-coco-format
                Images : Visual Genome (VG_100K + VG_100K_2)
                Output : datasets/VG150/VG150_coco_format/{train,val,test}/

    IndoorVG  → maelic/IndoorVG-coco-format
                Images : Visual Genome (VG_100K + VG_100K_2)
                Output : datasets/IndoorVG/IndoorVG_coco_format/{train,val,test}/

The reconstructed directory layout per split is::

    {output_dir}/{split}/_annotations.coco.json
    {output_dir}/{split}/<image files>   (only when --save-images is set)

Notes
-----
* Category and predicate maps are embedded in ``dataset_info.description`` as JSON.
* Without ``--save-images`` the annotation JSON references ``file_name`` fields that
  match the original image filenames; make sure your local image directories match
  the ``img_dir`` entry in ``configs/hydra/default.yaml``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # SGG-Benchmark root

DATASET_CONFIGS: dict[str, dict] = {
    "PSG": {
        "hub_repo":    "maelic/PSG-coco-format",
        "default_dir": BASE_DIR / "datasets/PSG/coco_format",
        "splits":      ("train", "val", "test"),
        "image_note": (
            "PSG uses COCO images.  Download them from:\n"
            "  https://cocodataset.org/#download\n"
            "  (train2017.zip + val2017.zip)\n"
            "and place them under datasets/PSG/coco_format/{train,val,test}/"
        ),
    },
    "VG150": {
        "hub_repo":    "maelic/VG150-coco-format",
        "default_dir": BASE_DIR / "datasets/VG150/VG150_coco_format",
        "splits":      ("train", "val", "test"),
        "image_note": (
            "VG150 uses Visual Genome images.  Download them from:\n"
            "  https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip  (~9 GB)\n"
            "  https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip (~5 GB)\n"
            "and place all *.jpg files under datasets/VG150/VG150_coco_format/{train,val,test}/"
        ),
    },
    "IndoorVG": {
        "hub_repo":    "maelic/IndoorVG-coco-format",
        "default_dir": BASE_DIR / "datasets/IndoorVG/IndoorVG_coco_format",
        "splits":      ("train", "val", "test"),
        "image_note": (
            "IndoorVG uses Visual Genome images.  Download them from:\n"
            "  https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip  (~9 GB)\n"
            "  https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip (~5 GB)\n"
            "and place all *.jpg files under datasets/IndoorVG/IndoorVG_coco_format/{train,val,test}/"
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_metadata(hub_repo: str) -> tuple[list, list]:
    """Load category + predicate lists from the repo's categories.json file."""
    try:
        cats_file = hf_hub_download(hub_repo, "categories.json", repo_type="dataset")
        with open(cats_file) as f:
            meta = json.load(f)
        return meta.get("categories", []), meta.get("rel_categories", [])
    except Exception:
        return [], []


def _build_coco_json(split_ds, categories: list, rel_categories: list) -> dict:
    """
    Reconstruct a COCO-format annotation dict from one HF Dataset split.

    Each row is expected to have the fields produced by push_to_hub.py:
        image, image_id, width, height, file_name, objects, relations
    """
    images = []
    annotations = []
    rel_annotations = []

    for row in split_ds:
        img_id = row["image_id"]
        images.append({
            "id":        img_id,
            "file_name": row["file_name"],
            "width":     row["width"],
            "height":    row["height"],
        })

        for obj in row["objects"]:
            annotations.append({
                "id":           obj["id"],
                "image_id":     img_id,
                "category_id":  obj["category_id"],
                "bbox":         obj["bbox"],
                "area":         obj["area"],
                "iscrowd":      obj["iscrowd"],
                "segmentation": obj.get("segmentation", []),
            })

        for rel in row["relations"]:
            rel_annotations.append({
                "id":           rel["id"],
                "image_id":     img_id,
                "subject_id":   rel["subject_id"],
                "object_id":    rel["object_id"],
                "predicate_id": rel["predicate_id"],
            })

    return {
        "images":          images,
        "annotations":     annotations,
        "rel_annotations": rel_annotations,
        "categories":      categories,
        "rel_categories":  rel_categories,
    }


def _save_images(split_ds, split_dir: Path) -> None:
    """Save PIL images from the HF dataset to *split_dir*."""
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(split_ds)
    for i, row in enumerate(split_ds, 1):
        dst = split_dir / row["file_name"]
        if not dst.exists():
            row["image"].save(dst)
        if i % 500 == 0 or i == total:
            print(f"    saved {i}/{total} images …", end="\r")
    print()


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_dataset(
    dataset_name: str,
    output_dir: Path | None = None,
    save_images: bool = False,
) -> None:
    cfg = DATASET_CONFIGS[dataset_name]
    hub_repo:  str   = cfg["hub_repo"]
    splits:    tuple = cfg["splits"]
    local_dir: Path  = output_dir or cfg["default_dir"]

    print(f"\n{'='*70}")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Hub repo : {hub_repo}")
    print(f"  Output   : {local_dir}")
    print(f"{'='*70}\n")

    print(f"Downloading {hub_repo} from Hugging Face Hub …")
    dataset_dict = load_dataset(hub_repo)

    categories, rel_categories = _extract_metadata(hub_repo)
    print(f"  categories     : {len(categories)}")
    print(f"  rel_categories : {len(rel_categories)}\n")

    for split in splits:
        if split not in dataset_dict:
            print(f"  [SKIP] split '{split}' not found in the Hub dataset.")
            continue

        split_dir = local_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing split '{split}' ({len(dataset_dict[split])} rows) …")

        # Reconstruct annotation JSON
        coco_json = _build_coco_json(dataset_dict[split], categories, rel_categories)
        ann_file = split_dir / "_annotations.coco.json"
        with open(ann_file, "w") as f:
            json.dump(coco_json, f)
        print(f"  Wrote {ann_file}")
        print(f"    images={len(coco_json['images'])}, "
              f"annotations={len(coco_json['annotations'])}, "
              f"relations={len(coco_json['rel_annotations'])}")

        # Optionally save images
        if save_images:
            print(f"  Saving images to {split_dir} …")
            _save_images(dataset_dict[split], split_dir)

    print("\nAnnotation JSON files written successfully.")

    if not save_images:
        print("\n⚠  Images were NOT downloaded.  You need to supply them separately.")
        print(cfg["image_note"])

    print(f"\nThe dataset is ready to use with the catalog key:")
    key_map = {"PSG": "psg_coco", "VG150": "vg150_coco", "IndoorVG": "indoorvg_coco"}
    print(f"  '{key_map[dataset_name]}'")
    print("  See configs/hydra/default.yaml for the full catalog entry.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
