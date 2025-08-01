import os
import torch
import torch.utils.data as data
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as T

# Define the classes based on the animal_XXX.png files
CLASSES = [
    'background', # SSD models typically require a background class
    'animal_1010', 'animal_1016', 'animal_1089', 'animal_1236', 'animal_1400',
    'animal_1478', 'animal_1492', 'animal_543', 'animal_639', 'animal_689',
    'animal_806', 'animal_859', 'animal_913', 'animal_951', 'animal_986'
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
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, "images")
        self.annotation_dir = os.path.join(root, "annotations")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Convert PIL Image to PyTorch Tensor here, before any other transforms
        img = T.ToTensor()(img)

        # Load annotation
        annotation_name = os.path.splitext(img_name)[0] + ".xml"
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.class_to_idx: # Only include known classes
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
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
        return len(self.image_files)

def PascalVOCDataset(*args, cache=False, **kwargs):
    return CachedDataset(_PascalVOCDataset(*args, **kwargs)) if cache else _PascalVOCDataset(*args, **kwargs)


def get_transform(train):
    transforms = []
    if train:
        # Data augmentation for training
        # transforms.append(T.RandomHorizontalFlip(0.5)) # Example augmentation
        pass # For now, keep it simple
    return Compose(transforms)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

if __name__ == '__main__':
    # Example usage:
    # First, ensure you have generated data using generate_tfod_data.py
    # For example: uv run python tfod_coco_ssd/data/generate_tfod_data.py --num_samples 10 --output_base_dir torch_ssd/data
    
    data_root = "./torch_ssd/data"
    
    # Create a dummy data directory for testing if it doesn't exist
    if not os.path.exists(data_root):
        print(f"Data directory {data_root} not found. Please run generate_tfod_data.py first.")
        print("Example: uv run python tfod_coco_ssd/data/generate_tfod_data.py --original_assets_dir model/data --num_samples 100 --output_base_dir torch_ssd/data")
    else:
        dataset = PascalVOCDataset(data_root, get_transform(train=True))
        print(f"Dataset size: {len(dataset)}")

        # Test loading a sample
        if len(dataset) > 0:
            img, target = dataset[0]
            print(f"Image tensor shape: {img.shape}")
            print(f"Target boxes shape: {target['boxes'].shape}")
            print(f"Target labels shape: {target['labels'].shape}")
            print(f"Target labels: {target['labels']}")
            print(f"Class names: {[CLASSES[l] for l in target['labels']]}")
        else:
            print("No samples in the dataset to test.")
