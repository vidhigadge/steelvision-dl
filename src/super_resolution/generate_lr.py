from pathlib import Path
from PIL import Image

SOURCE = Path("data/splits/test")
OUTPUT = Path("data/sr/lr")

RESOLUTIONS = [128, 64, 32]


def main():
    for class_dir in SOURCE.iterdir():
        if not class_dir.is_dir():
            continue

        for resolution in RESOLUTIONS:
            output_dir = (
                OUTPUT
                / f"{resolution}x{resolution}"
                / class_dir.name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                }:
                    continue

                image = Image.open(image_path)

                low_res = image.resize(
                    (resolution, resolution),
                    Image.Resampling.BICUBIC,
                )

                low_res.save(
                    output_dir / image_path.name
                )


if __name__ == "__main__":
    main()