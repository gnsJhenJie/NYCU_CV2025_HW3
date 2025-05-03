"""Train Mask R‑CNN on the converted COCO dataset using Detectron2."""
import argparse
from pathlib import Path
import os
import json
import torch
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances


def register_dataset(data_root: Path):
    ann_path = data_root / "train_annotations.json"
    if "med_seg_train" in DatasetCatalog.list():
        return  # already registered
    register_coco_instances(
        "med_seg_train", {},
        ann_path.as_posix(),
        (data_root / "train").as_posix())
    MetadataCatalog.get("med_seg_train").thing_classes = [
        "cell_type_1",
        "cell_type_2",
        "cell_type_3",
        "cell_type_4",
    ]


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file("configs/mask_rcnn_R50_FPN_med.yaml")
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main(args):
    data_root = Path(args.data_root).resolve()
    register_dataset(data_root)
    cfg = setup_cfg(args)
    default_setup(cfg, args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument("--machine_rank", default=0, type=int)
    args = parser.parse_args()

    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url="auto", args=(args,))
