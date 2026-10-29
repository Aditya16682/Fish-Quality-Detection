
from pathlib import Path
import random
import shutil

SOURCE = Path(
    "data/Digital image analysis aided spoilage of puti fish (puntius sophore) copy"
)
OUTPUT = Path("data/split")

random.seed(42)
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

for class_name in ["fresh", "spoiled"]:
    source_class = SOURCE / class_name

    if not source_class.exists():
        raise FileNotFoundError(f"Missing class folder: {source_class}")

    images = [
        p for p in source_class.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    random.shuffle(images)

    if len(images) < 3:
        raise ValueError(f"Not enough images in {class_name}")

    # 70% train, 15% validation, 15% test
    train_end = int(len(images) * 0.70)
    val_end = int(len(images) * 0.85)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, files in splits.items():
        destination = OUTPUT / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(files):
            new_name = f"{index}_{image_path.name}"
            shutil.copy2(image_path, destination / new_name)

    print(
        f"{class_name}: "
        f"{len(splits['train'])} train, "
        f"{len(splits['validation'])} validation, "
        f"{len(splits['test'])} test"
    )

print("Split complete. Original images were not moved or deleted.")
