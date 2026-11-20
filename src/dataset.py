
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATA_DIR = Path("data")
METADATA_PATH = DATA_DIR / "metadata.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "paper_cardboard",
    "plastic_packaging",
    "glass_packaging",
    "metal_packaging",
    "organic_waste",
    "general_waste",
    "other_or_not_an_item",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

SPLITS = ["train", "validation", "test"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class WasteDataset(Dataset):

    def __init__(self, split):
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")

        self.samples = []
        self.transform = train_transform if split == "train" else eval_transform

        with open(METADATA_PATH, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] != split:
                    continue

                image_path = DATA_DIR / row["image_path"]
                label = CLASS_TO_IDX[row["class_label"]]

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No images found for split: {split}")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label



def create_loader(split, batch_size=BATCH_SIZE):
    dataset = WasteDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=3,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


if __name__ == "__main__":
    train_loader = create_loader("train")
    val_loader = create_loader("validation")

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"CUDA available: {torch.cuda.is_available()}")
