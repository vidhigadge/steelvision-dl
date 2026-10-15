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

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.50


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

    if not label_path.exists():
        return ground_truth

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
    model: YOLO,
    image_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    results = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
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
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
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

            if iou < iou_threshold:
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
    matches = []

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

        matches.append(
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
        matches,
        matched_gt,
        matched_predictions,
    )


def diagnose_failures(
    ground_truth: list[dict],
    predictions: list[dict],
    matched_gt: set[int],
    matched_predictions: set[int],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
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
            class_match
            and iou < iou_threshold
        ):
            reason = "poor_localization"

        elif (
            not class_match
            and iou >= iou_threshold
        ):
            reason = "wrong_class"

        elif (
            not class_match
            and iou < iou_threshold
        ):
            reason = (
                "wrong_class_and_poor_localization"
            )

        else:
            reason = "unmatched"

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


def evaluate_image(
    model: YOLO,
    image_path: Path,
    label_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image_width, image_height = (
        image.size
    )

    ground_truth = load_ground_truth(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
    )

    predictions = load_predictions(
        model=model,
        image_path=image_path,
        confidence_threshold=confidence_threshold,
    )

    (
        matches,
        matched_gt,
        matched_predictions,
    ) = find_true_positive_matches(
        ground_truth=ground_truth,
        predictions=predictions,
        iou_threshold=iou_threshold,
    )

    diagnostics = diagnose_failures(
        ground_truth=ground_truth,
        predictions=predictions,
        matched_gt=matched_gt,
        matched_predictions=matched_predictions,
        iou_threshold=iou_threshold,
    )

    true_positives = len(
        matches
    )

    false_positives = (
        len(predictions)
        - true_positives
    )

    false_negatives = (
        len(ground_truth)
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

    mean_iou = (
        sum(
            match["iou"]
            for match in matches
        )
        / len(matches)
        if matches
        else 0.0
    )

    diagnostic_gt = {
        item["gt_index"]
        for item in diagnostics
    }

    diagnostic_predictions = {
        item["pred_index"]
        for item in diagnostics
    }

    missed_ground_truth = [
        {
            "index": index,
            **ground_truth[index],
        }
        for index in range(
            len(ground_truth)
        )
        if (
            index not in matched_gt
            and index not in diagnostic_gt
        )
    ]

    extra_predictions = [
        {
            "index": index,
            **predictions[index],
        }
        for index in range(
            len(predictions)
        )
        if (
            index not in matched_predictions
            and index not in diagnostic_predictions
        )
    ]

    return {
        "image": image,
        "image_path": image_path,
        "label_path": label_path,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "matches": matches,
        "diagnostics": diagnostics,
        "missed_ground_truth": missed_ground_truth,
        "extra_predictions": extra_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }


def draw_ground_truth(
    image: Image.Image,
    ground_truth: list[dict],
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

    return output


def draw_predictions(
    image: Image.Image,
    predictions: list[dict],
) -> Image.Image:
    output = image.copy()

    draw = ImageDraw.Draw(
        output
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
                y1 + 3,
            ),
            (
                f"P{index}: "
                f"{prediction['class_name']} "
                f"{prediction['confidence']:.2f}"
            ),
            fill="cyan",
        )

    return output


def draw_overlay(
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