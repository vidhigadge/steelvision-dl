from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
}


IMAGE_PATH = Path(
    "data/raw/NEU-CLS-YOLO/valid/images/crazing_1.jpg"
)

LABEL_PATH = Path(
    "data/raw/NEU-CLS-YOLO/valid/labels/crazing_1.txt"
)

OUTPUT_PATH = Path(
    "outputs/yolo_ground_truth_preview.jpg"
)


def yolo_to_xyxy(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    center_x_px = center_x * image_width
    center_y_px = center_y * image_height

    box_width_px = box_width * image_width
    box_height_px = box_height * image_height

    x1 = int(center_x_px - box_width_px / 2)
    y1 = int(center_y_px - box_height_px / 2)
    x2 = int(center_x_px + box_width_px / 2)
    y2 = int(center_y_px + box_height_px / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width - 1, x2)
    y2 = min(image_height - 1, y2)

    return x1, y1, x2, y2


def load_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    boxes = []

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            center_x = float(parts[1])
            center_y = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            x1, y1, x2, y2 = yolo_to_xyxy(
                center_x=center_x,
                center_y=center_y,
                box_width=box_width,
                box_height=box_height,
                image_width=image_width,
                image_height=image_height,
            )

            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    "box": (x1, y1, x2, y2),
                }
            )

    return boxes


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label not found: {LABEL_PATH}"
        )

    image = Image.open(IMAGE_PATH).convert("RGB")

    image_width, image_height = image.size

    boxes = load_ground_truth(
        label_path=LABEL_PATH,
        image_width=image_width,
        image_height=image_height,
    )

    draw = ImageDraw.Draw(image)

    for item in boxes:
        x1, y1, x2, y2 = item["box"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        label = item["class_name"]

        text_x = x1 + 4
        text_y = max(0, y1 + 4)

        draw.text(
            (text_x, text_y),
            label,
            fill="red",
        )

        print(
            f"Class: {item['class_name']}"
        )

        print(
            f"Ground-truth box: "
            f"({x1}, {y1}, {x2}, {y2})"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(OUTPUT_PATH)

    print()
    print(
        f"Saved preview to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()