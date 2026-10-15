from collections import defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
}

IMAGE_DIR = Path(
    "data/raw/NEU-CLS-YOLO/valid/images"
)

LABEL_DIR = Path(
    "data/raw/NEU-CLS-YOLO/valid/labels"
)

MODEL_PATH = Path(
    "runs/detect/steel_defect_yolo/weights/best.pt"
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
) -> list[dict]:
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
            and iou < IOU_THRESHOLD
        ):
            reason = "poor_localization"

        elif (
            not class_match
            and iou >= IOU_THRESHOLD
        ):
            reason = "wrong_class"

        elif (
            not class_match
            and iou < IOU_THRESHOLD
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


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found: "
            f"{LABEL_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: "
            f"{MODEL_PATH}"
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    image_paths = sorted(
        list(
            IMAGE_DIR.glob("*.jpg")
        )
        + list(
            IMAGE_DIR.glob("*.jpeg")
        )
        + list(
            IMAGE_DIR.glob("*.png")
        )
    )

    total_gt = 0
    total_predictions = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_tp_ious = []

    failure_counts = defaultdict(
        int
    )

    class_stats = defaultdict(
        lambda: {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": [],
        }
    )

    image_results = []

    for image_path in image_paths:
        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image_width, image_height = (
            image.size
        )

        ground_truth = load_ground_truth(
            label_path,
            image_width,
            image_height,
        )

        predictions = load_predictions(
            model,
            image_path,
        )

        (
            matches,
            matched_gt,
            matched_predictions,
        ) = find_true_positive_matches(
            ground_truth,
            predictions,
        )

        diagnostics = diagnose_failures(
            ground_truth,
            predictions,
            matched_gt,
            matched_predictions,
        )

        tp = len(
            matches
        )

        fp = (
            len(predictions)
            - tp
        )

        fn = (
            len(ground_truth)
            - tp
        )

        total_gt += len(
            ground_truth
        )

        total_predictions += len(
            predictions
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for gt in ground_truth:
            class_stats[
                gt["class_name"]
            ]["gt"] += 1

        for prediction in predictions:
            class_stats[
                prediction["class_name"]
            ]["pred"] += 1

        for match in matches:
            class_name = (
                match[
                    "ground_truth"
                ][
                    "class_name"
                ]
            )

            class_stats[
                class_name
            ]["tp"] += 1

            class_stats[
                class_name
            ]["ious"].append(
                match["iou"]
            )

            all_tp_ious.append(
                match["iou"]
            )

        for class_name, stats in (
            class_stats.items()
        ):
            stats["fp"] = (
                stats["pred"]
                - stats["tp"]
            )

            stats["fn"] = (
                stats["gt"]
                - stats["tp"]
            )

        for item in diagnostics:
            failure_counts[
                item["reason"]
            ] += 1

        image_results.append(
            {
                "image": image_path.name,
                "gt": len(
                    ground_truth
                ),
                "pred": len(
                    predictions
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    precision_denominator = (
        total_tp
        + total_fp
    )

    recall_denominator = (
        total_tp
        + total_fn
    )

    precision = (
        total_tp
        / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        total_tp
        / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    mean_iou = (
        sum(
            all_tp_ious
        )
        / len(
            all_tp_ious
        )
        if all_tp_ious
        else 0.0
    )

    print()
    print(
        "YOLO FULL GROUND-TRUTH EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Images evaluated:      "
        f"{len(image_paths)}"
    )

    print(
        f"Confidence threshold:  "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"IoU threshold:         "
        f"{IOU_THRESHOLD:.2f}"
    )

    print()

    print(
        "OVERALL"
    )

    print(
        "-" * 70
    )

    print(
        f"Ground-truth boxes:    "
        f"{total_gt}"
    )

    print(
        f"Predicted boxes:       "
        f"{total_predictions}"
    )

    print(
        f"True Positives:        "
        f"{total_tp}"
    )

    print(
        f"False Positives:       "
        f"{total_fp}"
    )

    print(
        f"False Negatives:       "
        f"{total_fn}"
    )

    print(
        f"Precision:             "
        f"{precision:.4f}"
    )

    print(
        f"Recall:                "
        f"{recall:.4f}"
    )

    print(
        f"Mean TP IoU:           "
        f"{mean_iou:.4f}"
    )

    print()

    print(
        "FAILURE BREAKDOWN"
    )

    print(
        "-" * 70
    )

    print(
        f"Poor localization:     "
        f"{failure_counts['poor_localization']}"
    )

    print(
        f"Wrong class:           "
        f"{failure_counts['wrong_class']}"
    )

    print(
        "Wrong class + poor "
        "localization: "
        f"{failure_counts['wrong_class_and_poor_localization']}"
    )

    print()

    print(
        "PER-CLASS RESULTS"
    )

    print(
        "-" * 70
    )

    header = (
        f"{'Class':<20}"
        f"{'GT':>6}"
        f"{'Pred':>7}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'Prec':>9}"
        f"{'Recall':>9}"
        f"{'MeanIoU':>10}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for class_id in sorted(
        CLASS_NAMES
    ):
        class_name = CLASS_NAMES[
            class_id
        ]

        stats = class_stats[
            class_name
        ]

        class_precision_denominator = (
            stats["tp"]
            + stats["fp"]
        )

        class_recall_denominator = (
            stats["tp"]
            + stats["fn"]
        )

        class_precision = (
            stats["tp"]
            / class_precision_denominator
            if class_precision_denominator > 0
            else 0.0
        )

        class_recall = (
            stats["tp"]
            / class_recall_denominator
            if class_recall_denominator > 0
            else 0.0
        )

        class_mean_iou = (
            sum(
                stats["ious"]
            )
            / len(
                stats["ious"]
            )
            if stats["ious"]
            else 0.0
        )

        print(
            f"{class_name:<20}"
            f"{stats['gt']:>6}"
            f"{stats['pred']:>7}"
            f"{stats['tp']:>6}"
            f"{stats['fp']:>6}"
            f"{stats['fn']:>6}"
            f"{class_precision:>9.4f}"
            f"{class_recall:>9.4f}"
            f"{class_mean_iou:>10.4f}"
        )

    print()

    print(
        "PER-IMAGE SUMMARY"
    )

    print(
        "-" * 70
    )

    for item in image_results:
        print(
            f"{item['image']:<30}"
            f"GT={item['gt']:<3} "
            f"Pred={item['pred']:<3} "
            f"TP={item['tp']:<3} "
            f"FP={item['fp']:<3} "
            f"FN={item['fn']:<3}"
        )


if __name__ == "__main__":
    main()