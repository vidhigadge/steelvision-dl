from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
}

IMAGE_PATH = Path(
    "data/raw/NEU-CLS-YOLO/valid/images/inclusion_1.jpg"
)

LABEL_PATH = Path(
    "data/raw/NEU-CLS-YOLO/valid/labels/inclusion_1.txt"
)

MODEL_PATH = Path(
    "runs/detect/steel_defect_yolo/weights/best.pt"
)

OUTPUT_PATH = Path(
    "outputs/yolo_gt_vs_prediction.jpg"
)

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


def yolo_to_xyxy(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    center_x *= image_width
    center_y *= image_height
    box_width *= image_width
    box_height *= image_height

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    return x1, y1, x2, y2


def calculate_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def load_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict]:
    ground_truth = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            box = yolo_to_xyxy(
                center_x=float(parts[1]),
                center_y=float(parts[2]),
                box_width=float(parts[3]),
                box_height=float(parts[4]),
                image_width=image_width,
                image_height=image_height,
            )

            ground_truth.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    "box": box,
                }
            )

    return ground_truth


def load_predictions(
    image_path: Path,
) -> list[dict]:
    model = YOLO(
        str(MODEL_PATH)
    )

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    predictions = []

    if not results:
        return predictions

    boxes = results[0].boxes

    if boxes is None:
        return predictions

    for box in boxes:
        coords = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        predictions.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}",
                ),
                "confidence": confidence,
                "box": tuple(coords),
            }
        )

    return predictions


def find_true_positive_matches(
    ground_truth: list[dict],
    predictions: list[dict],
) -> tuple[
    list[dict],
    set[int],
    set[int],
]:
    candidates = []

    for gt_index, gt in enumerate(
        ground_truth
    ):
        for pred_index, prediction in enumerate(
            predictions
        ):
            if (
                gt["class_id"]
                != prediction["class_id"]
            ):
                continue

            iou = calculate_iou(
                gt["box"],
                prediction["box"],
            )

            if iou < IOU_THRESHOLD:
                continue

            candidates.append(
                {
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": iou,
                }
            )

    candidates.sort(
        key=lambda item: item["iou"],
        reverse=True,
    )

    matched_gt = set()
    matched_predictions = set()
    true_positive_matches = []

    for candidate in candidates:
        gt_index = candidate["gt_index"]
        pred_index = candidate["pred_index"]

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(
            gt_index
        )

        matched_predictions.add(
            pred_index
        )

        true_positive_matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "ground_truth": ground_truth[
                    gt_index
                ],
                "prediction": predictions[
                    pred_index
                ],
                "iou": candidate["iou"],
            }
        )

    return (
        true_positive_matches,
        matched_gt,
        matched_predictions,
    )


def diagnose_unmatched_cases(
    ground_truth: list[dict],
    predictions: list[dict],
    matched_gt: set[int],
    matched_predictions: set[int],
) -> list[dict]:
    remaining_gt = [
        index
        for index in range(
            len(ground_truth)
        )
        if index not in matched_gt
    ]

    remaining_predictions = [
        index
        for index in range(
            len(predictions)
        )
        if index not in matched_predictions
    ]

    candidates = []

    for gt_index in remaining_gt:
        for pred_index in remaining_predictions:
            iou = calculate_iou(
                ground_truth[
                    gt_index
                ]["box"],
                predictions[
                    pred_index
                ]["box"],
            )

            if iou <= 0:
                continue

            candidates.append(
                {
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": iou,
                }
            )

    candidates.sort(
        key=lambda item: item["iou"],
        reverse=True,
    )

    used_gt = set()
    used_predictions = set()
    diagnostics = []

    for candidate in candidates:
        gt_index = candidate["gt_index"]
        pred_index = candidate["pred_index"]

        if gt_index in used_gt:
            continue

        if pred_index in used_predictions:
            continue

        gt = ground_truth[
            gt_index
        ]

        prediction = predictions[
            pred_index
        ]

        class_match = (
            gt["class_id"]
            == prediction["class_id"]
        )

        iou = candidate["iou"]

        if (
            not class_match
            and iou >= IOU_THRESHOLD
        ):
            reason = "WRONG CLASS"

        elif (
            class_match
            and iou < IOU_THRESHOLD
        ):
            reason = "POOR LOCALIZATION"

        elif (
            not class_match
            and iou < IOU_THRESHOLD
        ):
            reason = (
                "WRONG CLASS + "
                "POOR LOCALIZATION"
            )

        else:
            reason = "UNMATCHED"

        diagnostics.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "ground_truth": gt,
                "prediction": prediction,
                "iou": iou,
                "reason": reason,
            }
        )

        used_gt.add(
            gt_index
        )

        used_predictions.add(
            pred_index
        )

    return diagnostics


def draw_boxes(
    image: Image.Image,
    ground_truth: list[dict],
    predictions: list[dict],
) -> Image.Image:
    output = image.copy()

    draw = ImageDraw.Draw(
        output
    )

    for index, gt in enumerate(
        ground_truth,
        start=1,
    ):
        x1, y1, x2, y2 = gt["box"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        draw.text(
            (
                x1 + 3,
                y1 + 3,
            ),
            (
                f"GT{index}: "
                f"{gt['class_name']}"
            ),
            fill="red",
        )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        x1, y1, x2, y2 = prediction["box"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="cyan",
            width=3,
        )

        draw.text(
            (
                x1 + 3,
                max(
                    0,
                    y1 + 16,
                ),
            ),
            (
                f"P{index}: "
                f"{prediction['class_name']} "
                f"{prediction['confidence']:.2f}"
            ),
            fill="cyan",
        )

    return output


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth label not found: "
            f"{LABEL_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: "
            f"{MODEL_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert(
        "RGB"
    )

    image_width, image_height = (
        image.size
    )

    ground_truth = load_ground_truth(
        LABEL_PATH,
        image_width,
        image_height,
    )

    predictions = load_predictions(
        IMAGE_PATH
    )

    (
        true_positive_matches,
        matched_gt,
        matched_predictions,
    ) = find_true_positive_matches(
        ground_truth,
        predictions,
    )

    diagnostics = diagnose_unmatched_cases(
        ground_truth,
        predictions,
        matched_gt,
        matched_predictions,
    )

    true_positives = len(
        true_positive_matches
    )

    false_negatives = (
        len(ground_truth)
        - true_positives
    )

    false_positives = (
        len(predictions)
        - true_positives
    )

    precision_denominator = (
        true_positives
        + false_positives
    )

    recall_denominator = (
        true_positives
        + false_negatives
    )

    precision = (
        true_positives
        / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positives
        / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    matched_ious = [
        match["iou"]
        for match in true_positive_matches
    ]

    mean_true_positive_iou = (
        sum(matched_ious)
        / len(matched_ious)
        if matched_ious
        else 0.0
    )

    print()
    print(
        "YOLO Ground-Truth Evaluation"
    )

    print(
        "=" * 60
    )

    print(
        f"Image: {IMAGE_PATH.name}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"IoU threshold: "
        f"{IOU_THRESHOLD:.2f}"
    )

    print()

    print(
        f"Ground-truth boxes: "
        f"{len(ground_truth)}"
    )

    print(
        f"Predicted boxes: "
        f"{len(predictions)}"
    )

    print()

    if true_positive_matches:
        print(
            "VALID TRUE-POSITIVE MATCHES"
        )

        print(
            "-" * 60
        )

        for index, match in enumerate(
            true_positive_matches,
            start=1,
        ):
            gt = match[
                "ground_truth"
            ]

            prediction = match[
                "prediction"
            ]

            print(
                f"Match #{index}"
            )

            print(
                f"  Ground Truth: "
                f"{gt['class_name']}"
            )

            print(
                f"  Prediction: "
                f"{prediction['class_name']}"
            )

            print(
                f"  Confidence: "
                f"{prediction['confidence']:.4f}"
            )

            print(
                f"  IoU: "
                f"{match['iou']:.4f}"
            )

            print(
                "  Result: TRUE POSITIVE"
            )

            print()

    if diagnostics:
        print(
            "FAILURE DIAGNOSTICS"
        )

        print(
            "-" * 60
        )

        for index, item in enumerate(
            diagnostics,
            start=1,
        ):
            gt = item[
                "ground_truth"
            ]

            prediction = item[
                "prediction"
            ]

            print(
                f"Failure #{index}"
            )

            print(
                f"  Ground Truth: "
                f"{gt['class_name']}"
            )

            print(
                f"  Prediction: "
                f"{prediction['class_name']}"
            )

            print(
                f"  Confidence: "
                f"{prediction['confidence']:.4f}"
            )

            print(
                f"  IoU: "
                f"{item['iou']:.4f}"
            )

            print(
                f"  Reason: "
                f"{item['reason']}"
            )

            print(
                "  Standard evaluation:"
            )

            print(
                "    GT contributes to FN"
            )

            print(
                "    Prediction contributes "
                "to FP"
            )

            print()

    diagnostic_gt = {
        item["gt_index"]
        for item in diagnostics
    }

    diagnostic_predictions = {
        item["pred_index"]
        for item in diagnostics
    }

    completely_missed_gt = [
        index
        for index in range(
            len(ground_truth)
        )
        if (
            index not in matched_gt
            and index not in diagnostic_gt
        )
    ]

    completely_unmatched_predictions = [
        index
        for index in range(
            len(predictions)
        )
        if (
            index not in matched_predictions
            and index
            not in diagnostic_predictions
        )
    ]

    if completely_missed_gt:
        print(
            "COMPLETELY MISSED DEFECTS"
        )

        print(
            "-" * 60
        )

        for gt_index in completely_missed_gt:
            gt = ground_truth[
                gt_index
            ]

            print(
                f"GT #{gt_index + 1}: "
                f"{gt['class_name']} "
                f"→ FALSE NEGATIVE"
            )

        print()

    if completely_unmatched_predictions:
        print(
            "EXTRA PREDICTIONS"
        )

        print(
            "-" * 60
        )

        for pred_index in (
            completely_unmatched_predictions
        ):
            prediction = predictions[
                pred_index
            ]

            print(
                f"Prediction "
                f"#{pred_index + 1}: "
                f"{prediction['class_name']} "
                f"(confidence "
                f"{prediction['confidence']:.4f}) "
                f"→ FALSE POSITIVE"
            )

        print()

    print(
        "SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"True Positives:       "
        f"{true_positives}"
    )

    print(
        f"False Positives:      "
        f"{false_positives}"
    )

    print(
        f"False Negatives:      "
        f"{false_negatives}"
    )

    print(
        f"Precision:            "
        f"{precision:.4f}"
    )

    print(
        f"Recall:               "
        f"{recall:.4f}"
    )

    print(
        f"Mean TP IoU:          "
        f"{mean_true_positive_iou:.4f}"
    )

    print()

    output_image = draw_boxes(
        image=image,
        ground_truth=ground_truth,
        predictions=predictions,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image.save(
        OUTPUT_PATH
    )

    print(
        "Legend:"
    )

    print(
        "  RED  = Ground Truth"
    )

    print(
        "  CYAN = YOLO Prediction"
    )

    print()

    print(
        f"Saved comparison image to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()