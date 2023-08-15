import dataclasses
import json
import os.path
from collections import defaultdict
from typing import List

import numpy as np
from torch.utils.data import Dataset

__all__ = ["DetectionSample", "COCODetectionDatasetReader"]


@dataclasses.dataclass
class DetectionSample:
    image_id: str
    image_path: str
    image_width: int
    image_height: int

    labels: np.ndarray  # [N]
    bboxes: np.ndarray  # [N, 4] in XYXY format
    is_difficult: np.ndarray  # [N]


class COCODetectionDatasetReader(Dataset):
    samples: List[DetectionSample]
    class_names: List[str]
    num_classes: int

    def __init__(self, samples: List[DetectionSample], class_names: List[str]):
        self.samples = samples
        self.class_names = class_names
        self.num_classes = len(class_names)

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def convert_to_dict(annotations):
        result_dict = defaultdict(list)
        for obj in annotations:
            image_id = obj["image_id"]
            result_dict[image_id].append(obj)
        return result_dict

    @classmethod
    def from_directory_and_annotation(cls, images_directory: str, annotation: str):
        samples = []
        with open(annotation, "r") as f:
            data = json.load(f)

        category_ids, class_names = zip(*[(category["id"], category["name"]) for category in data["categories"]])

        annotations = cls.convert_to_dict(data["annotations"])
        category_id_to_index = {category_id: index for index, category_id in enumerate(category_ids)}

        for image in data["images"]:
            image_id = image["id"]
            image_path = os.path.join(images_directory, image["file_name"])
            image_width = image["width"]
            image_height = image["height"]

            labels = []
            bboxes = []
            is_difficult = []

            if image_id in annotations:
                for annotations in annotations[image_id]:
                    class_index = category_id_to_index[annotations["category_id"]]
                    x, y, w, h = annotations["bbox"]
                    bbox_xyxy = [x, y, x + w, y + h]

                    labels.append(class_index)
                    bboxes.append(bbox_xyxy)
                    is_difficult.append(annotations["iscrowd"])

            sample = DetectionSample(
                image_id=image_id,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                labels=np.array(labels, dtype=int).reshape(-1),
                bboxes=np.array(bboxes, dtype=np.float32).reshape(-1, 4),
                is_difficult=np.array(is_difficult, dtype=bool).reshape(-1),
            )
            samples.append(sample)

        return cls(samples, class_names)


if __name__ == "__main__":
    import cv2

    start = cv2.getTickCount()
    train_ds = COCODetectionDatasetReader.from_directory_and_annotation(
        images_directory="e:/coco2017/images/train2017/", annotation="e:/coco2017/annotations/instances_train2017.json"
    )
    end = cv2.getTickCount()
    print((end - start) / cv2.getTickFrequency())

    start = cv2.getTickCount()
    valid_ds = COCODetectionDatasetReader.from_directory_and_annotation(
        images_directory="e:/coco2017/images/val2017/", annotation="e:/coco2017/annotations/instances_val2017.json"
    )
    end = cv2.getTickCount()
    print((end - start) / cv2.getTickFrequency())

    print(len(train_ds))
    print(len(valid_ds))
