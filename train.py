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
import random


def register_and_split(data_root: Path, val_ratio: float = 0.08):
    """Register full dataset once, then split into train/val and register both."""
    ann_path = data_root / "train_annotations.json"
    img_root = data_root / "train"
    full_name = "med_seg_full"

    if full_name not in DatasetCatalog.list():
        register_coco_instances(
            full_name, {},
            ann_path.as_posix(),
            img_root.as_posix())
        MetadataCatalog.get(full_name).thing_classes = [
            "cell_type_1",
            "cell_type_2",
            "cell_type_3",
            "cell_type_4",
        ]

    full_dataset = list(DatasetCatalog.get(full_name))
    random.Random().shuffle(full_dataset)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    val_dataset = full_dataset[:n_val]
    train_dataset = full_dataset[n_val:]

    # wrap list in a lambda to avoid serialization issues
    DatasetCatalog.register(
        "med_seg_train",
        lambda d=train_dataset: d,
    )
    DatasetCatalog.register(
        "med_seg_val",
        lambda d=val_dataset: d,
    )
    for name in ("med_seg_train", "med_seg_val"):
        MetadataCatalog.get(name).thing_classes = MetadataCatalog.get(
            full_name).thing_classes
    print(
        f"Dataset split ➜ train:{len(train_dataset)}  val:{len(val_dataset)}")


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file("configs/mask_rcnn_R50_FPN_med.yaml")
    cfg.OUTPUT_DIR = args.output_dir

    # datasets
    cfg.DATASETS.TRAIN = ("med_seg_train",)
    cfg.DATASETS.TEST = ("med_seg_val",)

    # solver overrides
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size

    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
    return cfg


def main(args):
    data_root = Path(args.data_root).resolve()
    register_and_split(data_root)
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
