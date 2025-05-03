"""Small helpers reused in both training and inference."""
from typing import List, Dict
import torch
import numpy as np
from pycocotools import mask as mask_utils
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper


def rle_encode(mask: np.ndarray) -> Dict:
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def prepare_test_mapper(dataset_dict):
    """Detectron2 DatasetMapper for test images (no GT)."""
    from detectron2.data import detection_utils as utils
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image = utils.resize_image(image, min_size=512, max_size=2048)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
    return dataset_dict


class MedMapper:
    """Custom mapper with strong data augmentation (no DatasetMapper base)."""

    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        if is_train:
            self.augmentations: List[T.Augmentation] = [
                # multi‑scale resize (short edge)
                T.ResizeShortestEdge(
                    short_edge_length=[512, 768, 896, 1024],
                    max_size=2048, sample_style="choice"),
                # horizontal & vertical flip (independent probabilistic)
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                # 90‑degree rotations
                T.RandomRotation(angle=[0, 90, 180, 270]),
                # color jitter
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
            ]
        else:
            self.augmentations = [T.ResizeShortestEdge(
                short_edge_length=512, max_size=2048)]
        self.image_format = "BGR"

    def __call__(self, dataset_dict):
        d = dataset_dict.copy()
        image = utils.read_image(d["file_name"], format=self.image_format)
        image, transforms = T.apply_augmentations(self.augmentations, image)
        d["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())

        if "annotations" in d:
            annos = [utils.transform_instance_annotations(
                a, transforms, image.shape[: 2])
                for a in d["annotations"]]
            d["instances"] = utils.annotations_to_instances(
                annos, image.shape[: 2], mask_format="bitmask")
        return d
