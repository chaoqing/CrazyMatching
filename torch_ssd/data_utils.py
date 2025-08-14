import os
import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T

# Define the classes based on the animal_XXX.png files
CLASSES = [
    "background",  # SSD models typically require a background class
    "animal_1010",
    "animal_1016",
    "animal_1089",
    "animal_1236",
    "animal_1400",
    "animal_1478",
    "animal_1492",
    "animal_543",
    "animal_639",
    "animal_689",
    "animal_806",
    "animal_859",
    "animal_913",
    "animal_951",
    "animal_986",
]


class CachedDataset(data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.cache = {}

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.original_dataset[idx]
        return self.cache[idx]


class _PascalVOCDataset(data.Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.root = root
        self.transforms = transforms
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

        self.train = train
        if self.train:
            self.images = np.load(os.path.join(root, "X_train.npy"))
            locations = np.load(os.path.join(root, "y_train.npy"))
            locations[:, :, 0] = locations[:, :, 0] - locations[:, :, 2] / 2  # xmin
            locations[:, :, 1] = locations[:, :, 1] - locations[:, :, 3] / 2  # ymin
            locations[:, :, 2] = locations[:, :, 0] + locations[:, :, 2] / 2  # xmax
            locations[:, :, 3] = locations[:, :, 1] + locations[:, :, 3] / 2  # ymax
            self.locations = locations
            labels = np.load(os.path.join(root, "c_train.npy"))
            self.labels = np.array(
                [self.class_to_idx[i] for i in labels.flatten()]
            ).reshape(labels.shape)
        else:
            self.image_dir = os.path.join(root, "images")
            self.annotation_dir = os.path.join(root, "labels")
            self.image_files = sorted(
                [f for f in os.listdir(self.image_dir) if f.endswith((".jpg", ".png"))]
            )

    def __getitem__(self, idx):
        if self.train:
            img = T.ToTensor()(self.images[idx])
            boxes = torch.as_tensor(self.locations[idx], dtype=torch.float32)
            labels = torch.as_tensor(self.labels[idx], dtype=torch.int64)
            target = dict(boxes=boxes, labels=labels, image_id=torch.tensor([idx]))
        else:
            img, target = self.getitem_raw_images(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def getitem_npy_images(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Convert PIL Image to PyTorch Tensor here, before any other transforms
        img = T.ToTensor()(img)

        # Load annotation
        annotation_name = os.path.splitext(img_name)[0] + ".json"
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        boxes = []
        labels = []

        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                for obj in json.load(f):
                    name = obj["label"]
                    if name in self.class_to_idx:  # Only include known classes
                        bndbox = obj["location"]
                        xmin = float(bndbox["center_x"] - bndbox["w"] // 2)
                        ymin = float(bndbox["center_x"] + bndbox["h"] // 2)
                        xmax = float(bndbox["center_x"] - bndbox["w"] // 2)
                        ymax = float(bndbox["center_x"] + bndbox["h"] // 2)

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.class_to_idx[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return img, target

    def getitem_raw_images(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Convert PIL Image to PyTorch Tensor here, before any other transforms
        img = T.ToTensor()(img)

        # Load annotation
        annotation_name = os.path.splitext(img_name)[0] + ".json"
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        boxes = []
        labels = []

        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                for obj in json.load(f):
                    name = obj["label"]
                    if name in self.class_to_idx:  # Only include known classes
                        bndbox = obj["location"]
                        xmin = float(bndbox["center_x"] - bndbox["w"] // 2)
                        ymin = float(bndbox["center_x"] + bndbox["h"] // 2)
                        xmax = float(bndbox["center_x"] - bndbox["w"] // 2)
                        ymax = float(bndbox["center_x"] + bndbox["h"] // 2)

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.class_to_idx[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if self.train:
            return self.images.shape[0]
        else:
            return len(self.image_files)


def PascalVOCDataset(*args, cache=False, **kwargs):
    return (
        CachedDataset(_PascalVOCDataset(*args, **kwargs))
        if cache
        else _PascalVOCDataset(*args, **kwargs)
    )


class RandomHorizontalFlipWithAnnotations:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if np.random.random() >= self.p:
            return image, target

        if "boxes" in target:
            bbox = target["boxes"]

            assert isinstance(image, torch.Tensor)
            w = image.shape[2]

            bbox[:, [0, 2]] = w - bbox[:, [2, 0]]
            target["boxes"] = bbox

        return T.functional.hflip(image), target


def get_transform(train):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlipWithAnnotations(0.5))
    return Compose(transforms)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


if __name__ == "__main__":
    data_root = Path(__file__).parent / ".." / "model" / "data" / "trainning_data"

    # Create a dummy data directory for testing if it doesn't exist
    if not data_root.exists():
        print(
            f"Data directory {data_root} not found. Please run simulate_data.py first."
        )
        print(
            f"Example: uv run python model/data/simulate_data.py --num_samples 100 --output_base_dir {data_root.resolve()}  --save_raw"
        )
    else:
        train_mode = True
        dataset = PascalVOCDataset(
            str(data_root),
            get_transform(train=train_mode),
            train=train_mode,
            cache=not train_mode,
        )
        print(f"Dataset size: {len(dataset)}")

        # Test loading a sample
        if len(dataset) > 0:
            img, target = dataset[0]
            print(f"Image tensor shape: {img.shape} - {img.dtype}")
            print(
                f"Target boxes shape: {target['boxes'].shape} - {target['boxes'].dtype}"
            )
            print(
                f"Target labels shape: {target['labels'].shape} - {target['labels'].dtype}"
            )
            print(f"Target labels: {target['labels']}")
            print(f"Class names: {[CLASSES[l] for l in target['labels']]}")
        else:
            print("No samples in the dataset to test.")
