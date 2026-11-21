
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.dataset import CLASS_NAMES


DATA_DIR = Path("data")
METADATA_PATH = DATA_DIR / "metadata.csv"

SPLITS = ["train", "validation", "test", "stress"]


def validate_dataset():
    counts = defaultdict(Counter)
    hashes = {}
    errors = []
    warnings = []

    with open(METADATA_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:

            split = row["split"].strip()
            label = row["class_label"].strip()
            image_path = DATA_DIR / row["image_path"].strip()

            # Check split and class
            if split not in SPLITS:
                errors.append(f"Invalid split: {split}")
                continue

            if label not in CLASS_NAMES:
                errors.append(f"Invalid class: {label}")
                continue

            counts[split][label] += 1

            # Check image path
            if not image_path.is_file():
                errors.append(f"Missing image: {image_path}")
                continue

            # Check corrupted images
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                errors.append(f"Corrupted image: {image_path}")
                continue

            # Check duplicate images
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

            if image_hash in hashes:
                previous_path, previous_split = hashes[image_hash]

                message = f"Duplicate: {previous_path} <-> {image_path}"

                if previous_split != split:
                    errors.append(message)
                else:
                    warnings.append(message)
            else:
                hashes[image_hash] = (image_path, split)

    ### Dataset statistics

    print("\nDATASET STATISTICS")
    print("-" * 50)

    for split in SPLITS:
        total = sum(counts[split].values())
        print(f"{split}: {total} images")

    print("\nIMAGES PER CLASS")
    print("-" * 50)

    for label in CLASS_NAMES:

        total = sum(
            counts[split][label]
            for split in ["train", "validation", "test"]
        )

        print(f"{label}: {total}")

        if total < 150:
            warnings.append(f"{label} has fewer than 150 images")

    if sum(counts["stress"].values()) < 50:
        warnings.append("Stress dataset has fewer than 50 images")


    ### Validation results

    print("\nVALIDATION RESULTS")
    print("-" * 50)

    for error in errors:
        print(f"ERROR: {error}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if not errors and not warnings:
        print("Dataset validation PASSED")
    else:
        print(f"\nErrors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    validate_dataset()
