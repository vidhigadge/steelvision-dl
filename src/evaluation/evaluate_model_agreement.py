from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TEST_DIR = Path("data/splits/test")

CLASSIFIER_PATH = Path(
    "models/resnet50_finetune.onnx"
)

DETECTOR_PATH = Path(
    "runs/detect/steel_defect_yolo/weights/best.pt"
)

YOLO_CONFIDENCE_THRESHOLD = 0.25

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


# -----------------------------------------------------------------------------
# Classification preprocessing
# -----------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

classifier_session = ort.InferenceSession(
    str(CLASSIFIER_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

detector = YOLO(
    str(DETECTOR_PATH)
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def softmax(values):
    values = (
        values
        - np.max(
            values,
            axis=1,
            keepdims=True,
        )
    )

    exp_values = np.exp(
        values
    )

    return (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )
    )


def classify_image(image):
    tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    input_array = np.ascontiguousarray(
        tensor.numpy(),
        dtype=np.float32,
    )

    input_name = (
        classifier_session
        .get_inputs()[0]
        .name
    )

    logits = classifier_session.run(
        None,
        {
            input_name:
            input_array
        },
    )[0]

    probabilities = softmax(
        logits
    )[0]

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    return {
        "class_name":
            CLASS_NAMES[
                predicted_index
            ],

        "confidence":
            float(
                probabilities[
                    predicted_index
                ]
            ),
    }


def detect_image(image):
    results = detector.predict(
        source=np.array(
            image.convert("RGB")
        ),
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        detections.append(
            {
                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "confidence":
                    confidence,
            }
        )

    detections.sort(
        key=lambda detection:
            detection["confidence"],
        reverse=True,
    )

    return detections


def create_class_stats():
    stats = {}

    for class_name in CLASS_NAMES:
        stats[class_name] = {
            "total": 0,
            "resnet_correct": 0,
            "yolo_top_correct": 0,
            "yolo_any_correct": 0,
            "top_agreement": 0,
            "any_agreement": 0,
            "no_detection": 0,
            "multi_class": 0,
        }

    return stats


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def main():
    total_images = 0

    classifier_correct = 0
    yolo_top_correct = 0
    yolo_any_correct = 0

    top_agreement = 0
    any_agreement = 0

    disagreement_count = 0
    no_detection_count = 0
    multi_class_detection_count = 0

    resnet_correct_yolo_wrong = 0
    yolo_correct_resnet_wrong = 0
    both_wrong = 0

    disagreement_examples = []

    class_stats = create_class_stats()

    for class_dir in sorted(
        TEST_DIR.iterdir()
    ):
        if not class_dir.is_dir():
            continue

        ground_truth = class_dir.name

        if ground_truth not in CLASS_NAMES:
            continue

        for image_path in sorted(
            class_dir.glob("*")
        ):
            if (
                image_path.suffix.lower()
                not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }
            ):
                continue

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            classifier_result = (
                classify_image(
                    image
                )
            )

            detections = (
                detect_image(
                    image
                )
            )

            total_images += 1

            class_stats[
                ground_truth
            ][
                "total"
            ] += 1

            classifier_class = (
                classifier_result[
                    "class_name"
                ]
            )

            classifier_confidence = (
                classifier_result[
                    "confidence"
                ]
            )

            classifier_is_correct = (
                classifier_class
                == ground_truth
            )

            if classifier_is_correct:
                classifier_correct += 1

                class_stats[
                    ground_truth
                ][
                    "resnet_correct"
                ] += 1

            # -------------------------------------------------------------
            # No YOLO detections
            # -------------------------------------------------------------

            if not detections:
                no_detection_count += 1
                disagreement_count += 1

                class_stats[
                    ground_truth
                ][
                    "no_detection"
                ] += 1

                if classifier_is_correct:
                    resnet_correct_yolo_wrong += 1

                else:
                    both_wrong += 1

                disagreement_examples.append(
                    {
                        "image":
                            image_path.name,

                        "ground_truth":
                            ground_truth,

                        "resnet":
                            classifier_class,

                        "resnet_confidence":
                            classifier_confidence,

                        "yolo_top":
                            None,

                        "yolo_top_confidence":
                            None,

                        "yolo_classes":
                            [],
                    }
                )

                continue

            # -------------------------------------------------------------
            # YOLO results
            # -------------------------------------------------------------

            yolo_top_class = (
                detections[0][
                    "class_name"
                ]
            )

            yolo_top_confidence = (
                detections[0][
                    "confidence"
                ]
            )

            yolo_classes = [
                detection[
                    "class_name"
                ]
                for detection
                in detections
            ]

            unique_yolo_classes = set(
                yolo_classes
            )

            if (
                len(
                    unique_yolo_classes
                )
                > 1
            ):
                multi_class_detection_count += 1

                class_stats[
                    ground_truth
                ][
                    "multi_class"
                ] += 1

            yolo_top_is_correct = (
                yolo_top_class
                == ground_truth
            )

            yolo_any_is_correct = (
                ground_truth
                in yolo_classes
            )

            if yolo_top_is_correct:
                yolo_top_correct += 1

                class_stats[
                    ground_truth
                ][
                    "yolo_top_correct"
                ] += 1

            if yolo_any_is_correct:
                yolo_any_correct += 1

                class_stats[
                    ground_truth
                ][
                    "yolo_any_correct"
                ] += 1

            # -------------------------------------------------------------
            # ResNet-YOLO agreement
            # -------------------------------------------------------------

            top_models_agree = (
                classifier_class
                == yolo_top_class
            )

            any_models_agree = (
                classifier_class
                in yolo_classes
            )

            if top_models_agree:
                top_agreement += 1

                class_stats[
                    ground_truth
                ][
                    "top_agreement"
                ] += 1

            else:
                disagreement_count += 1

            if any_models_agree:
                any_agreement += 1

                class_stats[
                    ground_truth
                ][
                    "any_agreement"
                ] += 1

            # -------------------------------------------------------------
            # Disagreement analysis
            # -------------------------------------------------------------

            if not top_models_agree:
                if (
                    classifier_is_correct
                    and not yolo_top_is_correct
                ):
                    resnet_correct_yolo_wrong += 1

                elif (
                    yolo_top_is_correct
                    and not classifier_is_correct
                ):
                    yolo_correct_resnet_wrong += 1

                elif (
                    not classifier_is_correct
                    and not yolo_top_is_correct
                ):
                    both_wrong += 1

                disagreement_examples.append(
                    {
                        "image":
                            image_path.name,

                        "ground_truth":
                            ground_truth,

                        "resnet":
                            classifier_class,

                        "resnet_confidence":
                            classifier_confidence,

                        "yolo_top":
                            yolo_top_class,

                        "yolo_top_confidence":
                            yolo_top_confidence,

                        "yolo_classes":
                            yolo_classes,
                    }
                )

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "STEELVISION CROSS-MODEL AGREEMENT EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Total test images: "
        f"{total_images}"
    )

    print()

    print(
        "MODEL ACCURACY"
    )

    print(
        "-" * 80
    )

    print(
        f"ResNet correct: "
        f"{classifier_correct} / {total_images} "
        f"({classifier_correct / total_images * 100:.2f}%)"
    )

    print(
        f"YOLO top-class correct: "
        f"{yolo_top_correct} / {total_images} "
        f"({yolo_top_correct / total_images * 100:.2f}%)"
    )

    print(
        f"YOLO any-detection correct: "
        f"{yolo_any_correct} / {total_images} "
        f"({yolo_any_correct / total_images * 100:.2f}%)"
    )

    print()

    print(
        "MODEL AGREEMENT"
    )

    print(
        "-" * 80
    )

    print(
        f"Top-class agreement: "
        f"{top_agreement} / {total_images} "
        f"({top_agreement / total_images * 100:.2f}%)"
    )

    print(
        f"Any-detection agreement: "
        f"{any_agreement} / {total_images} "
        f"({any_agreement / total_images * 100:.2f}%)"
    )

    print(
        f"Top-class disagreements: "
        f"{disagreement_count}"
    )

    print()

    print(
        "DISAGREEMENT ANALYSIS"
    )

    print(
        "-" * 80
    )

    print(
        f"ResNet correct / YOLO wrong: "
        f"{resnet_correct_yolo_wrong}"
    )

    print(
        f"YOLO correct / ResNet wrong: "
        f"{yolo_correct_resnet_wrong}"
    )

    print(
        f"Both wrong: "
        f"{both_wrong}"
    )

    print(
        f"YOLO no detection: "
        f"{no_detection_count}"
    )

    print(
        f"YOLO multiple classes detected: "
        f"{multi_class_detection_count}"
    )

    print()

    # ---------------------------------------------------------------------
    # Per-class summary
    # ---------------------------------------------------------------------

    print(
        "PER-CLASS BREAKDOWN"
    )

    print(
        "-" * 80
    )

    header = (
        f"{'Class':<20}"
        f"{'Total':>7}"
        f"{'ResNet':>10}"
        f"{'YOLO Top':>11}"
        f"{'YOLO Any':>11}"
        f"{'Top Agree':>12}"
        f"{'Any Agree':>12}"
        f"{'No Det':>9}"
        f"{'Multi':>8}"
    )

    print(
        header
    )

    print(
        "-" * 80
    )

    for class_name in CLASS_NAMES:
        stats = class_stats[
            class_name
        ]

        total = stats[
            "total"
        ]

        print(
            f"{class_name:<20}"
            f"{total:>7}"
            f"{stats['resnet_correct']:>10}"
            f"{stats['yolo_top_correct']:>11}"
            f"{stats['yolo_any_correct']:>11}"
            f"{stats['top_agreement']:>12}"
            f"{stats['any_agreement']:>12}"
            f"{stats['no_detection']:>9}"
            f"{stats['multi_class']:>8}"
        )

    print()

    print(
        "PER-CLASS PERCENTAGES"
    )

    print(
        "-" * 80
    )

    for class_name in CLASS_NAMES:
        stats = class_stats[
            class_name
        ]

        total = stats[
            "total"
        ]

        if total == 0:
            continue

        print(
            f"\n{class_name}"
        )

        print(
            f"  ResNet accuracy: "
            f"{stats['resnet_correct'] / total * 100:.2f}%"
        )

        print(
            f"  YOLO top accuracy: "
            f"{stats['yolo_top_correct'] / total * 100:.2f}%"
        )

        print(
            f"  YOLO any accuracy: "
            f"{stats['yolo_any_correct'] / total * 100:.2f}%"
        )

        print(
            f"  Top agreement: "
            f"{stats['top_agreement'] / total * 100:.2f}%"
        )

        print(
            f"  Any-detection agreement: "
            f"{stats['any_agreement'] / total * 100:.2f}%"
        )

        print(
            f"  No-detection rate: "
            f"{stats['no_detection'] / total * 100:.2f}%"
        )

    print()

    # ---------------------------------------------------------------------
    # All disagreement examples
    # ---------------------------------------------------------------------

    print(
        "ALL DISAGREEMENTS"
    )

    print(
        "-" * 80
    )

    for example in disagreement_examples:
        print(
            f"\nImage: "
            f"{example['image']}"
        )

        print(
            f"Ground truth: "
            f"{example['ground_truth']}"
        )

        print(
            f"ResNet: "
            f"{example['resnet']} "
            f"({example['resnet_confidence']:.4f})"
        )

        if (
            example[
                "yolo_top"
            ]
            is None
        ):
            print(
                "YOLO top: "
                "NO DETECTION"
            )

        else:
            print(
                f"YOLO top: "
                f"{example['yolo_top']} "
                f"({example['yolo_top_confidence']:.4f})"
            )

        print(
            f"YOLO classes: "
            f"{example['yolo_classes']}"
        )

    print()

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()