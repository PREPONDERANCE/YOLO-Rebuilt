import os
import json
import torch

from config import settings

from torch.utils.data import Dataset
from torchvision import datasets, transforms


class VocDataset:
    """
    A wrapper class around the official `VOCDetection` dataset.

    This class does the following things:

    - Extract and id all classification labels from the datasets.
    - Reshape each label class into the following structure.

    ```
        label: {
            "width": width,
            "height": height,
            "bbox": [
                [name, xmin, ymin, xmax, ymax],
            ]
        }
    ```
    """

    def __init__(self, data: datasets.VOCDetection):
        self.data = data
        self.class_path = f"{self.data.root}/classification.txt"
        self.classifications = None

    def _load_classes(self):
        """
        This will load the classification from path specified by
        `self.class_path`. If this path does not exist, classification
        will be set to `None`.
        """

        if os.path.exists(self.class_path):
            with open(self.class_path, "r") as f:
                self.classifications = json.load(f)
        else:
            self.classifications = None

    def _save_classes(self):
        """
        Save the classification result to the path specified by
        `self.class_path` to enable fast retrieval.
        """

        with open(self.class_path, "w+") as f:
            json.dump(self.classifications, f)

    def get_classes(self):
        """
        Retrieve classification results from all the labels.
        """

        self._load_classes()

        if self.classifications is not None:
            return

        self.classifications = {}

        idx = 0
        for i in range(len(self.data)):
            _, label = self.__getitem__(i)

            for bbox in label["bbox"]:
                if bbox[0] not in self.classifications:
                    self.classifications[bbox[0]] = idx
                    idx += 1

        self._save_classes()

    def __getitem__(self, i: int):
        new_label = {}
        data, label = self.data[i]

        _label = label["annotation"]
        new_label["width"], new_label["height"] = float(_label["size"]["width"]), float(_label["size"]["height"])
        new_label["bbox"] = [[obj["name"], *list(map(float, obj["bndbox"].values()))] for obj in _label["object"]]

        return data, new_label

    def __len__(self):
        return len(self.data)


class YOLOv1DataSet(Dataset):
    """
    Implement a custom dataset for PASCAL VOC 2007 dataset
    """

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        *,
        augment: bool = False,
        normalize: bool = False,
    ):
        self.dataset = VocDataset(
            datasets.VOCDetection(
                root,
                year="2007",
                image_set=image_set,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
                    ]
                ),
            )
        )
        self.dataset.get_classes()

        self.augment = augment
        self.normalize = normalize
        self.cls = self.dataset.classifications

    def __getitem__(self, index):
        gt = torch.zeros(settings.S, settings.S, settings.BOX)
        data, label = self.dataset[index]

        w, h = label["width"], label["height"]
        w_scale, h_scale = w / settings.IMG_SIZE, h / settings.IMG_SIZE

        # parallel operation

        for i, bbox in enumerate(label["bbox"]):
            cname, xmin, ymin, xmax, ymax = bbox

            xmin /= w_scale
            xmax /= w_scale
            ymin /= h_scale
            ymax /= h_scale

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            b_width = xmax - xmin
            b_height = ymax - ymin
            b_cls = torch.zeros(settings.C)
            b_cls[self.cls[cname]] = 1

            gi = int(x_center // (settings.IMG_SIZE / settings.S))
            gj = int(y_center // (settings.IMG_SIZE / settings.S))

            gt[gi, gj, settings.B * 5 : :] = b_cls
            if i < settings.B:
                gt[gi, gj, i * 5 : (i + 1) * 5] = torch.Tensor([x_center, y_center, b_width, b_height, 1.0])

        return data, gt

    def __len__(self):
        return len(self.dataset)
