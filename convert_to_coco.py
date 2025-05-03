"""Convert raw *.tif + class*.tif masks into a single COCO‑format JSON."""
import argparse
from pathlib import Path
import json
import numpy as np
import skimage.io as sio
from tqdm import tqdm
from pycocotools import mask as mask_utils

CATEGORY_NAMES = ["cell_type_1", "cell_type_2", "cell_type_3", "cell_type_4"]


def encode_binary_mask(mask: np.ndarray):
    """Encode binary mask to COCO RLE (compressed)."""
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_image(img_dir: Path, img_id: int, ann_start_id: int):
    image_path = img_dir / "image.tif"
    height, width = sio.imread(image_path).shape[:2]
    image_record = {
        "id": img_id,
        "file_name": str(image_path.resolve()),
        "height": height,
        "width": width,
    }

    annotations = []
    ann_id = ann_start_id

    for class_idx in range(1, 5):  # class1 – class4
        mask_path = img_dir / f"class{class_idx}.tif"
        if not mask_path.exists():
            continue  # some classes missing
        mask_arr = sio.imread(mask_path)
        instance_ids = np.unique(mask_arr)
        instance_ids = instance_ids[instance_ids > 0]
        for inst_id in instance_ids:
            binary_mask = mask_arr == inst_id
            if binary_mask.sum() == 0:
                continue
            rle = encode_binary_mask(binary_mask)
            area = int(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_idx,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return image_record, annotations, ann_id


def main(raw_root: Path, output_json: Path):
    train_dir = raw_root / "train"
    img_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    images, annotations = [], []
    ann_id_counter = 1
    for img_id, img_dir in tqdm(enumerate(img_dirs, 1), total=len(img_dirs)):
        img_rec, ann_recs, ann_id_counter = process_image(
            img_dir, img_id, ann_id_counter
        )
        images.append(img_rec)
        annotations.extend(ann_recs)

    categories = [
        {"id": i + 1, "name": n, "supercategory": "cell"}
        for i, n in enumerate(CATEGORY_NAMES)
    ]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output_json.write_text(json.dumps(coco_dict))
    print(f"Saved COCO annotations ➜ {output_json}  (images:{len(images)}, anns:{len(annotations)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str, help="Root with train/ & test_release/")
    parser.add_argument("--out", default="train_annotations.json", type=str)
    args = parser.parse_args()
    main(Path(args.data_root), Path(args.out))