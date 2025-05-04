"""Inference with optional Test-Time Augmentation (flip+scale) & Mask Voting."""
import argparse
import json
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import GenericMask
from tqdm import tqdm
from pycocotools import mask as mask_utils
from utils.data_utils import rle_encode

CATEGORY_ID_MAP = {0: 1, 1: 2, 2: 3, 3: 4}


def load_test_mapping(json_path: Path):
    return {item["file_name"]: item["id"] for item in json.loads(json_path.read_text())}


def build_cfg(cfg_path: str, weights: str, score_thresh=0.4, enable_tta=False):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 2048

    if enable_tta:
        cfg.defrost()
        cfg.TEST.AUG.ENABLED = True
        cfg.TEST.AUG.FLIP = True
        cfg.TEST.AUG.SCALE = (0.75, 1.0, 1.25)
        cfg.freeze()
    else:
        cfg.freeze()

    return cfg


def predictor_with_tta(cfg):
    model = DefaultPredictor(cfg).model
    # placeholder
    return DefaultPredictor(cfg).trainer.build_model(cfg) if False else DefaultPredictor(cfg)


def merge_masks(masks: List[np.ndarray], scores: List[float]):
    """Simple mask voting: pick the one with highest score (fast)."""
    idx = int(np.argmax(scores))
    return masks[idx], scores[idx]


def run_inference(args):
    data_root = Path(args.data_root)
    test_dir = data_root / "test_release"
    mapping = load_test_mapping(data_root / "test_image_name_to_ids.json")

    cfg = build_cfg(args.cfg_file, args.weights,
                    score_thresh=0.4, enable_tta=args.tta)

    predictor = DefaultPredictor(cfg)
    if args.tta:
        predictor.model = GeneralizedRCNNWithTTA(cfg, predictor.model)

    results = []
    for tif_path in tqdm(sorted(test_dir.glob("*.tif"))):
        image = cv2.imread(str(tif_path))
        outputs = predictor(image)
        if isinstance(outputs, dict):
            instances = outputs["instances"].to(torch.device("cpu"))
        else:  # TTA returns Instances
            instances = outputs.to(torch.device("cpu"))

        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for m, s, c in zip(masks, scores, classes):
            rle = rle_encode(m)
            bbox = mask_utils.toBbox(rle).tolist()
            results.append(
                {
                    "image_id": mapping[tif_path.name],
                    "category_id": CATEGORY_ID_MAP[c],
                    "segmentation": rle,
                    "score": float(s),
                    "bbox": bbox,
                }
            )

    zip_path = Path(args.output_zip)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test-results.json", json.dumps(results))
    print(f"Saved submission ➜ {zip_path}  (instances {len(results)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument(
        "--weights", default="./outputs_R101/model_final.pth", type=str)
    parser.add_argument(
        "--cfg_file", default="configs/mask_rcnn_R101_FPN_med.yaml", type=str)
    parser.add_argument(
        "--output_zip", default="test-results.zip", type=str,
        help="Output ZIP file name (will contain test-results.json)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable flip+scale TTA")
    args = parser.parse_args()
    run_inference(args)
