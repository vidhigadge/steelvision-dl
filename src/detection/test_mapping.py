from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from src.detection.coordinate_mapping import map_box_to_original


MODEL_PATH = "runs/detect/steel_defect_yolo/weights/best.pt"
IMAGE_DIR = Path("data/raw/NEU-CLS-YOLO/valid/images")
PROCESSED_PATH = Path("data/processed/mapping_test.jpg")


def main():
    model = YOLO(MODEL_PATH)

    image_path = next(IMAGE_DIR.glob("*.jpg"))

    with Image.open(image_path) as image:
        original_size = image.size

        processed = image.resize((800, 800))

        PROCESSED_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed.save(PROCESSED_PATH)

    results = model.predict(
        source=str(PROCESSED_PATH),
        device="mps",
        conf=0.25,
        verbose=False,
    )

    result = results[0]

    processed_size = (
        result.orig_shape[1],
        result.orig_shape[0],
    )

    print("Image:", image_path.name)
    print("Original size:", original_size)
    print("Processed size:", processed_size)

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().tolist()

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        mapped_box = map_box_to_original(
            box=xyxy,
            processed_size=processed_size,
            original_size=original_size,
        )

        print()
        print("Class:", model.names[class_id])
        print(f"Confidence: {confidence:.3f}")

        print(
            "Processed box:",
            tuple(round(value, 2) for value in xyxy),
        )

        print(
            "Mapped box:",
            tuple(round(value, 2) for value in mapped_box),
        )


if __name__ == "__main__":
    main()