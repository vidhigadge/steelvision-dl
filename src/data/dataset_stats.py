from pathlib import Path

DATA_DIR = Path("data/raw/NEU-CLS")

for folder in sorted(DATA_DIR.iterdir()):
    if folder.is_dir():
        count = len(list(folder.glob("*.*")))
        print(f"{folder.name:<20} {count}")

from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/NEU-CLS")

sizes = set()

for folder in DATA_DIR.iterdir():
    if not folder.is_dir():
        continue

    for image_path in folder.glob("*.*"):
        with Image.open(image_path) as image:
            sizes.add(image.size)

print("Unique image sizes:")
print(sizes)