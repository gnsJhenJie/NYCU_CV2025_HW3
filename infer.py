"""Run inference on test_release/ and write test‑results.json for submission."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from utils.data_utils import rle_encode


CATEGORY_ID_MAP = {0: 1, 1: 2, 2: 3, 3: 4}  # model idx ➜ official id


def load_test_mapping(json_path: Path):
    return {item["file_name"]: item["id"] for item in json.loads(json_path.read_text())}


def build_predictor(cfg_path: str, weights_path: str, score_thresh=0.05):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.freeze()
    return DefaultPredictor(cfg)


def main(args):
    data_root = Path(args.data_root)
    test_dir = data_root / "test_release"
    mapping = load_test_mapping(data_root / "test_image_name_to_ids.json")

    predictor = build_predictor(
        "configs/mask_rcnn_R50_FPN_med.yaml", args.weights, score_thresh=0.3)

    results = []
    for tif_path in tqdm(sorted(test_dir.glob("*.tif"))):
        image = cv2.imread(str(tif_path))
        outputs = predictor(image)
        instances = outputs["instances"].to(torch.device("cpu"))
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for m, s, c in zip(masks, scores, classes):
            rle = rle_encode(m)
            bbox = mask_utils.toBbox(
                mask_utils.frPyObjects(rle, *m.shape)).tolist()
            results.append(
                {
                    "image_id": mapping[tif_path.name],
                    "category_id": CATEGORY_ID_MAP[c],
                    "segmentation": rle,
                    "score": float(s),
                    "bbox": bbox,
                }
            )

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results))
    print(f"Saved submission to {out_path}  (instances {len(results)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument(
        "--weights", default="./outputs/model_final.pth", type=str)
    parser.add_argument("--output_json", default="test-results.json", type=str)
    args = parser.parse_args()
    main(args)
