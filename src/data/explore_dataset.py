from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/NEU-CLS")

class_folders = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir()
)

print("Classes found:")

for folder in class_folders:
    image_files = list(folder.glob("*.*"))
    print(f"{folder.name}: {len(image_files)} images")

from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/NEU-CLS")

class_folders = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir()
)

print("\nClasses:")

for folder in class_folders:
    images = list(folder.glob("*.*"))
    print(f"{folder.name}: {len(images)}")

first_image_path = list(class_folders[0].glob("*.*"))[0]

image = Image.open(first_image_path)

print("\nSample image information:")
print("Path:", first_image_path)
print("Size:", image.size)
print("Mode:", image.mode)