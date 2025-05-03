"""Small helpers reused in both training and inference."""
from typing import List, Dict

import torch
import numpy as np
from pycocotools import mask as mask_utils


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
