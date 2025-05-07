# NYCU Visual Recognition Deep Learning 2025 Homework 3

Student ID: 313551085
Name: 謝振杰

---

## Introduction

This repository contains my solution for **Homework 3 – Instance Segmentation** in the *Visual Recognition using Deep Learning* course.
The goal is to segment four different cell types (class 1 – 4) from colored microscopic images and achieve the highest AP<sub>50</sub> score on the CodaBench leaderboard.

Key highlights

* **Mask R‑CNN + FPN backbone** (ResNet‑50/101) implemented with Detectron2.
* **Strong data augmentation** (multiscale resize, two–way flips, 90 ° rotations, color jitter).
* **Mixed‑precision training** (AMP) and cosine/step LR schedulers.
* **Test‑time augmentation (TTA)** with flip + scale and simple *mask voting*.
* End‑to‑end utilities for dataset conversion, resolution inspection, training, inference, and CodaBench submission packaging.

---

## Repository Structure

```
├── configs/                # Detectron2 YAML configs (R50/R101 variants)
├── convert_to_coco.py      # Convert raw .tif masks ➜ COCO JSON
├── data/                   # ➜ Place dataset here (train/ & test_release/)
├── infer.py                # Inference & submission zip generator
├── train.py                # Training entry point (single‑GPU & DDP)
├── utils/
│   ├── check_image_sizes.py  # Inspect min/max H×W of images
│   └── data_utils.py         # Small helpers & custom DatasetMapper
├── .devcontainer/          # VSCode GPU dev‑container (PyTorch 24.03)
├── hw3-sample-code/        # Sample code provided from course materials
├── requirements.txt        # Python dependencies
├── analysis.ipynb          # iPython notebook for analysis
└── README.md
```

---

## Installation

> **Recommended:** open the project with VSCode ➜ *Reopen in Dev-Container*.

1. **Clone repo & build dev‑container**:

   ```bash
   git clone git@github.com:gnsJhenJie/NYCU_CV2025_HW3.git && cd NYCU_CV2025_HW3
   # VSCode will automatically build the container
   pip install -r requirements.txt # inside container
   ```

---

## Dataset Preparation

1. **Folder structure** (download from E3 ⇒ unzip under `data/`):

   ```
   data/
   ├── train/
   │   └── {uuid}/
   │       ├── image.tif         # RGB image
   │       ├── class1.tif        # instance ids for cell type 1
   │       ├── class2.tif        # …
   │       └── class4.tif
   ├── test_image_name_to_ids.json
   └── test_release/
       └── {uuid}.tif
   ```
2. **Generate COCO annotations** for training:

   ```bash
   python convert_to_coco.py --data_root ./data \
                             --out train_annotations.json
   ```

   The script stitches the per‑class masks into a single COCO‑style JSON (`images`, `annotations`, `categories`).

---

## Usage

### 1. Scan image resolutions (optional)

```bash
python utils/check_image_sizes.py --data_root ./data
```

Prints min/max H×W and suggests `INPUT.MIN_SIZE_TRAIN` & `MAX_SIZE_TRAIN` values.

### 2. Training

```bash
python train.py \
  --data_root   ./data \
  --output_dir  ./outputs_R101 \
  --cfg_file    configs/mask_rcnn_R101_FPN_med.yaml \
  --batch_size  6 \
  --num_gpus    1          # set >1 for DDP
```

Main flags

| flag           | description                                                       |
| -------------- | ----------------------------------------------------------------- |
| `--cfg_file`   | Choose between the provided R50/R101 configs or your custom YAML. |
| `--batch_size` | Per‑GPU images; lower it if you hit OOM.                          |
| `--num_gpus`   | Launches distributed training via Detectron2 `launch()`.          |

Checkpoints and TensorBoard logs are saved to `OUTPUT_DIR`.

### 3. Inference & Submission

```bash
python infer.py \
  --data_root ./data \
  --weights   ./outputs_R101/model_final.pth \
  --cfg_file  configs/mask_rcnn_R101_FPN_med.yaml \
  --output_zip submission.zip \
  --tta       # optional flip+scale Test Time Augmentation
```

* Produces `submission.zip` containing **`test-results.json`** in COCO format.

---

## Performance

| Model               | Config                        | Public LB mAP |
| ------------------- | ----------------------------- | ------------------------- |
| Mask R‑CNN R50‑FPN  | `mask_rcnn_R50_FPN_med.yaml`  | 0.35                      |
| Mask R‑CNN R101‑FPN | `mask_rcnn_R101_FPN_med.yaml` | **0.41**                  |

![image](https://i.imgur.com/YYVuKKn.png)

Training curves and more ablation results are provided in the report.

---

## Homework Report

Please refer to [Homework 3 Report](https://hackmd.io/b8AMTGSdSEa51rItcApjqA) for detailed methodology, additional experiments, and discussion.

---