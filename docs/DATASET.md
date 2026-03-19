# DATASETS

| Dataset  | 🤗 HuggingFace (COCO format) | Original annotations (LEGACY) | Images (LEGACY) | Train | Val | Test |
|----------|--------------------------|----------------------|--------|------:|----:|-----:|
| VG-150   | [maelic/VG150-coco-format](https://huggingface.co/datasets/maelic/VG150-coco-format) | [OneDrive](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) | [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) / [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) | 73 538 | 4 844 | 27 032 |
| IndoorVG | [maelic/IndoorVG-coco-format](https://huggingface.co/datasets/maelic/IndoorVG-coco-format) | [Google Drive](https://drive.google.com/file/d/1zfKXzmLxxYMzwlECtSch84oknCBEXTzI/view?usp=sharing) | Uses VG images | 9 538 | 733 | 4 403 |
| PSG      | [maelic/PSG-coco-format](https://huggingface.co/datasets/maelic/PSG-coco-format) | [OneDrive](https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fdata%2Fpsg) | [COCO Download](https://entuedu-my.sharepoint.com/:u:/r/personal/jingkang001_e_ntu_edu_sg/Documents/openpsg/data/coco.zip?csf=1&web=1&e=9Z513T) | 45 564 | 1 000 | 2 186 |

---

## Quickstart — Download from Hugging Face (recommended)

All three datasets are published on Hugging Face in a unified COCO-JSON format,
ready to plug directly into the SGG-Benchmark training pipeline.

### 1. Download annotation JSONs

```bash
# PSG
python tools/download_from_hub.py --dataset PSG

# VG150
python tools/download_from_hub.py --dataset VG150

# IndoorVG
python tools/download_from_hub.py --dataset IndoorVG
```

This writes `_annotations.coco.json` files under the standard paths:

```
datasets/PSG/coco_format/{train,val,test}/_annotations.coco.json
datasets/VG150/VG150_coco_format/{train,val,test}/_annotations.coco.json
datasets/IndoorVG/IndoorVG_coco_format/{train,val,test}/_annotations.coco.json
```

### 2. (Optional) Also download images from the Hub

```bash
Indopython tools/download_from_hub.py --dataset PSG --save-images
```

### 3. Use in training

Set the catalog key in your Hydra config:

```yaml
# configs/hydra/<YourExperiment>.yaml
datasets:
  name: "PSG"
  type: "coco"
  data_dir: "datasets/PSG/PSG_coco_format"
```

The three catalog entries are pre-registered in `configs/hydra/default.yaml`:

| Catalog key       | Hub repo                          | Local path                                      |
|-------------------|-----------------------------------|-------------------------------------------------|
| `psg_coco`        | `maelic/PSG-coco-format`          | `datasets/PSG/coco_format/`                     |
| `vg150_coco`      | `maelic/VG150-coco-format`        | `datasets/VG150/VG150_coco_format/`             |
| `indoorvg_coco`   | `maelic/IndoorVG-coco-format`     | `datasets/IndoorVG/IndoorVG_coco_format/`       |

### 4. Load directly with 🤗 Datasets (Python)

```python
from datasets import load_dataset
import json

ds = load_dataset("maelic/PSG-coco-format")   # or VG150-coco-format / IndoorVG-coco-format

# Recover label maps from embedded metadata
meta = json.loads(ds["train"].info.description)
cat_id2name  = {c["id"]: c["name"] for c in meta["categories"]}
pred_id2name = {c["id"]: c["name"] for c in meta["rel_categories"]}

sample = ds["train"][0]
for obj in sample["objects"]:
    print(cat_id2name[obj["category_id"]], obj["bbox"])
for rel in sample["relations"]:
    print(rel["subject_id"], "--", pred_id2name[rel["predicate_id"]], "->", rel["object_id"])
```

---

# :warning: LEGACY (not supported anymore): classical download

## Image download

### VG images (VG150 & IndoorVG)

```bash
# Part 1 (~9 GB)
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
# Part 2 (~5 GB)
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
```

After extracting, copy / symlink all `*.jpg` files into the split directories:

```bash
# Example for VG150
for split in train val test; do
  mkdir -p datasets/VG150/VG150_coco_format/$split
  # symlink (no disk copy needed)
  python - <<'EOF'
import json, os, pathlib
ann = json.load(open(f"datasets/VG150/VG150_coco_format/{split}/_annotations.coco.json"))
src = pathlib.Path("datasets/VG_100K")   # adjust to your VG image directory
dst = pathlib.Path(f"datasets/VG150/VG150_coco_format/{split}")
for img in ann["images"]:
    s = src / img["file_name"]
    if s.exists():
        (dst / img["file_name"]).symlink_to(s.resolve())
EOF
done
```

### PSG / COCO images

```bash
# COCO 2017 train (~18 GB)
wget http://images.cocodataset.org/zips/train2017.zip
# COCO 2017 val (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
```

Place extracted images under `datasets/PSG/coco_format/{train,val,test}/`.

---

## VG-150

This is the data split used by most SGG papers. It is composed of the top 150 object
classes and 50 predicate classes from the Visual Genome dataset.
The pre-processing of this split is adapted from
[Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md)
and [neural-motifs](https://github.com/rowanz/neural-motifs).
All object bounding boxes corresponding to the 150 most common object classes are
selected, which means that some images have objects but no relations.

:warning: This data split has been heavily criticized for having high biases in the
data distribution and classes semantics (e.g. classes *person, man, men, people*, etc.
highly intersect in the annotations).
See [this paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Liang_VrR-VG_Refocusing_Visually-Relevant_Relationships_ICCV_2019_paper.html)
and [this one](https://openaccess.thecvf.com/content/ICCV2023W/SG2RL/html/Neau_Fine-Grained_is_Too_Coarse_A_Novel_Data-Centric_Approach_for_Efficient_ICCVW_2023_paper.html)
for reference. Use it at your own risks.

Note that VG150 annotations intend to support attributes since the
[work from Kaihua](https://openaccess.thecvf.com/content_CVPR_2020/html/Tang_Unbiased_Scene_Graph_Generation_From_Biased_Training_CVPR_2020_paper.html),
so the `VG-SGG.h5` and `VG-SGG-dicts.json` differ from the versions in
[Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md)
and [neural-motifs](https://github.com/rowanz/neural-motifs).
Attribute information has been added and the files renamed to
`VG-SGG-with-attri.h5` and `VG-SGG-dicts-with-attri.json`.
The code used to generate them is at `process_data/generate_attribute_labels.py`.
Attribute head is disabled by default in the codebase due to poor reported performance.

### Manual download (legacy HDF5 format)

1. Download the VG images
   [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
   [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip).
   Extract to `datasets/vg/VG_100K`.
2. Download the [scene graphs](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed)
   and extract to `datasets/VG150/VG-SGG-with-attri.h5`.

## IndoorVG

This data split is proposed in a
[recent work (2023)](https://link.springer.com/chapter/10.1007/978-3-031-55015-7_25).
It is another split of Visual Genome targeting real-world applications in indoor
settings, composed of 84 object classes and 37 predicate classes manually selected
and refined using semi-automatic merging and processing techniques.
To use it you can download the VG images from the link above and the annotated scene
graphs [from this link](https://drive.google.com/file/d/1zfKXzmLxxYMzwlECtSch84oknCBEXTzI/view?usp=sharing).


## PSG

The PSG dataset is a new approach originally targeting the
[Panoptic Scene Graph Generation](https://arxiv.org/abs/2207.11247) task.
However, its annotations can also be used for traditional SGG.
It is composed of images from COCO and VG which have been re-annotated aiming at
fixing biases from Visual Genome.

> ⚠️ The COCO-format version provided here does **not** contain panoptic segmentation
> masks, only bounding boxes. It cannot be used to train panoptic SGG models.
> For the full panoptic data see the
> [OpenPSG project page](https://github.com/Jingkang50/OpenPSG).

The data (images + full panoptic graphs) can be downloaded using
[the authors' link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh).

Note that for efficient encoding of the class labels it is necessary to change some
names (e.g. remove "-merged" or "-other" suffixes); see our pre-processed class names
in [datasets/psg/obj_classes.txt](datasets/psg/obj_classes.txt).

# Creating Your Own Dataset

Please check the [SGG-Annotate](https://github.com/Maelic/SGG-Annotate) tool to
create your own dataset in COCO format.
Annotations can then be loaded using the `RelationDataset` class; see the
`custom_dataset` example in
[configs/hydra/default.yaml](configs/hydra/default.yaml).
For more information on the annotation format see
[ANNOTATIONS.md](ANNOTATIONS.md).
