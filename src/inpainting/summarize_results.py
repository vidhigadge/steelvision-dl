"""Summarize EXP-016 inpainting reliability results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def percentage(series: pd.Series) -> float:
    """Convert mean of binary values to percentage."""
    return float(series.mean() * 100.0)


def build_method_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby("method")
        .agg(
            cases=("method", "count"),
            class_flips=("class_flip", "sum"),
            flip_rate=("class_flip", percentage),
            reconstructed_accuracy=(
                "reconstructed_correct",
                percentage,
            ),
            confidence_change=(
                "confidence_change",
                "mean",
            ),
            entropy_change=(
                "entropy_change",
                "mean",
            ),
            psnr=("psnr", "mean"),
            ssim=("ssim", "mean"),
            mae=("mae", "mean"),
        )
        .reset_index()
    )


def build_placement_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby("placement")
        .agg(
            cases=("placement", "count"),
            class_flips=("class_flip", "sum"),
            flip_rate=("class_flip", percentage),
            reconstructed_accuracy=(
                "reconstructed_correct",
                percentage,
            ),
            confidence_change=(
                "confidence_change",
                "mean",
            ),
            entropy_change=(
                "entropy_change",
                "mean",
            ),
            psnr=("psnr", "mean"),
            ssim=("ssim", "mean"),
            mae=("mae", "mean"),
        )
        .reset_index()
    )


def build_method_placement_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "placement",
                "method",
            ]
        )
        .agg(
            cases=("method", "count"),
            class_flips=("class_flip", "sum"),
            flip_rate=("class_flip", percentage),
            reconstructed_accuracy=(
                "reconstructed_correct",
                percentage,
            ),
            confidence_change=(
                "confidence_change",
                "mean",
            ),
            entropy_change=(
                "entropy_change",
                "mean",
            ),
            psnr=("psnr", "mean"),
            ssim=("ssim", "mean"),
            mae=("mae", "mean"),
        )
        .reset_index()
    )


def build_mask_ratio_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby(
            "mask_ratio_target"
        )
        .agg(
            cases=("mask_ratio_target", "count"),
            class_flips=("class_flip", "sum"),
            flip_rate=("class_flip", percentage),
            reconstructed_accuracy=(
                "reconstructed_correct",
                percentage,
            ),
            confidence_change=(
                "confidence_change",
                "mean",
            ),
            entropy_change=(
                "entropy_change",
                "mean",
            ),
        )
        .reset_index()
    )


def build_mask_type_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby("mask_type")
        .agg(
            cases=("mask_type", "count"),
            class_flips=("class_flip", "sum"),
            flip_rate=("class_flip", percentage),
            reconstructed_accuracy=(
                "reconstructed_correct",
                percentage,
            ),
            confidence_change=(
                "confidence_change",
                "mean",
            ),
            entropy_change=(
                "entropy_change",
                "mean",
            ),
        )
        .reset_index()
    )


def build_class_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.groupby(
            "ground_truth_class"
        )
        .agg(
            cases=("ground_truth_class", "count"),
            class_flips=("class_flip", "sum"),
            flip_rate=("class_flip", percentage),
            reconstructed_accuracy=(
                "reconstructed_correct",
                percentage,
            ),
            confidence_change=(
                "confidence_change",
                "mean",
            ),
            entropy_change=(
                "entropy_change",
                "mean",
            ),
        )
        .reset_index()
    )


def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_csv(
        path,
        index=False,
    )


def plot_flip_rate_by_method(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        summary["method"],
        summary["flip_rate"],
    )

    axis.set_title(
        "Class Flip Rate by Inpainting Method"
    )

    axis.set_xlabel(
        "Method"
    )

    axis.set_ylabel(
        "Flip Rate (%)"
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "flip_rate_by_method.png",
        dpi=180,
    )

    plt.close(
        figure
    )


def plot_flip_rate_by_placement(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        summary["placement"],
        summary["flip_rate"],
    )

    axis.set_title(
        "Class Flip Rate by Mask Placement"
    )

    axis.set_xlabel(
        "Placement"
    )

    axis.set_ylabel(
        "Flip Rate (%)"
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "flip_rate_by_placement.png",
        dpi=180,
    )

    plt.close(
        figure
    )


def plot_flip_rate_by_ratio(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        summary[
            "mask_ratio_target"
        ],
        summary[
            "flip_rate"
        ],
        marker="o",
    )

    axis.set_title(
        "Class Flip Rate vs Mask Coverage"
    )

    axis.set_xlabel(
        "Target Mask Coverage"
    )

    axis.set_ylabel(
        "Flip Rate (%)"
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "flip_rate_by_mask_ratio.png",
        dpi=180,
    )

    plt.close(
        figure
    )


def plot_quality_vs_task(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    grouped = (
        df.groupby("method")
        .agg(
            flip_rate=(
                "class_flip",
                percentage,
            ),
            psnr=(
                "psnr",
                "mean",
            ),
        )
        .reset_index()
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.scatter(
        grouped["psnr"],
        grouped["flip_rate"],
    )

    for _, row in grouped.iterrows():
        axis.annotate(
            row["method"],
            (
                row["psnr"],
                row["flip_rate"],
            ),
        )

    axis.set_title(
        "Reconstruction Quality vs Task Preservation"
    )

    axis.set_xlabel(
        "Mean PSNR"
    )

    axis.set_ylabel(
        "Class Flip Rate (%)"
    )

    figure.tight_layout()

    figure.savefig(
        output_dir
        / "quality_vs_task.png",
        dpi=180,
    )

    plt.close(
        figure
    )


def write_markdown_summary(
    output_path: Path,
    source_csv: Path,
    df: pd.DataFrame,
    method_summary: pd.DataFrame,
    placement_summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    mask_type_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> None:
    overall_flip_rate = (
        float(
            df[
                "class_flip"
            ].mean()
        )
        * 100.0
    )

    overall_accuracy = (
        float(
            df[
                "reconstructed_correct"
            ].mean()
        )
        * 100.0
    )

    overall_confidence_change = float(
        df[
            "confidence_change"
        ].mean()
    )

    overall_entropy_change = float(
        df[
            "entropy_change"
        ].mean()
    )

    lines = [
        "# EXP-016 Inpainting Reliability Summary",
        "",
        f"Source CSV: `{source_csv}`",
        "",
        "## Overall",
        "",
        f"- Cases: **{len(df)}**",
        f"- Class flips: **{int(df['class_flip'].sum())}**",
        f"- Flip rate: **{overall_flip_rate:.2f}%**",
        f"- Reconstructed-case accuracy: **{overall_accuracy:.2f}%**",
        (
            "- Mean confidence change: "
            f"**{overall_confidence_change:.4f}**"
        ),
        (
            "- Mean entropy change: "
            f"**{overall_entropy_change:.4f}**"
        ),
        "",
        "## By Method",
        "",
        method_summary.to_markdown(
            index=False
        ),
        "",
        "## By Placement",
        "",
        placement_summary.to_markdown(
            index=False
        ),
        "",
        "## By Mask Ratio",
        "",
        ratio_summary.to_markdown(
            index=False
        ),
        "",
        "## By Mask Type",
        "",
        mask_type_summary.to_markdown(
            index=False
        ),
        "",
        "## By Class",
        "",
        class_summary.to_markdown(
            index=False
        ),
        "",
        "## Interpretation Notes",
        "",
        (
            "- Lower class-flip rate indicates stronger "
            "downstream decision preservation."
        ),
        (
            "- Negative confidence change means the classifier "
            "became less confident after reconstruction."
        ),
        (
            "- Positive entropy change means classifier "
            "uncertainty increased."
        ),
        (
            "- Higher PSNR / SSIM and lower MAE indicate "
            "better pixel reconstruction."
        ),
        (
            "- Better reconstruction metrics do not necessarily "
            "imply better downstream classification reliability."
        ),
        (
            "- High-attention regions are defined using Grad-CAM "
            "model influence, not physical defect ground truth."
        ),
        (
            "- Inpainting is evaluated on synthetically removed "
            "regions where the original image is retained as reference."
        ),
        (
            "- Results should not be interpreted as proof that "
            "missing real-world defect information can be recovered."
        ),
        "",
    ]

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/inpainting/summary"
        ),
    )

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {args.csv}"
        )

    df = pd.read_csv(
        args.csv
    )

    if df.empty:
        raise RuntimeError(
            "Results CSV is empty."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    method_summary = (
        build_method_summary(
            df
        )
    )

    placement_summary = (
        build_placement_summary(
            df
        )
    )

    method_placement_summary = (
        build_method_placement_summary(
            df
        )
    )

    ratio_summary = (
        build_mask_ratio_summary(
            df
        )
    )

    mask_type_summary = (
        build_mask_type_summary(
            df
        )
    )

    class_summary = (
        build_class_summary(
            df
        )
    )

    save_csv(
        method_summary,
        args.output_dir
        / "method_summary.csv",
    )

    save_csv(
        placement_summary,
        args.output_dir
        / "placement_summary.csv",
    )

    save_csv(
        method_placement_summary,
        args.output_dir
        / "method_placement_summary.csv",
    )

    save_csv(
        ratio_summary,
        args.output_dir
        / "mask_ratio_summary.csv",
    )

    save_csv(
        mask_type_summary,
        args.output_dir
        / "mask_type_summary.csv",
    )

    save_csv(
        class_summary,
        args.output_dir
        / "class_summary.csv",
    )

    plot_flip_rate_by_method(
        method_summary,
        args.output_dir,
    )

    plot_flip_rate_by_placement(
        placement_summary,
        args.output_dir,
    )

    plot_flip_rate_by_ratio(
        ratio_summary,
        args.output_dir,
    )

    plot_quality_vs_task(
        df,
        args.output_dir,
    )

    write_markdown_summary(
        output_path=(
            args.output_dir
            / "summary.md"
        ),
        source_csv=args.csv,
        df=df,
        method_summary=method_summary,
        placement_summary=placement_summary,
        ratio_summary=ratio_summary,
        mask_type_summary=mask_type_summary,
        class_summary=class_summary,
    )

    print(
        "Summary complete."
    )

    print(
        "Input cases:",
        len(df),
    )

    print(
        "Output directory:",
        args.output_dir,
    )

    print()

    print(
        "Placement summary:"
    )

    print(
        placement_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Method summary:"
    )

    print(
        method_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()