"""Reconstruction metrics for EXP-016 inpainting reliability."""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def _validate_images(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert images to float32 in [0, 1]."""
    original = np.asarray(
        original,
        dtype=np.float32,
    )

    reconstructed = np.asarray(
        reconstructed,
        dtype=np.float32,
    )

    if original.shape != reconstructed.shape:
        raise ValueError(
            "original and reconstructed images "
            "must have the same shape."
        )

    if original.ndim != 2:
        raise ValueError(
            "EXP-016 reconstruction metrics expect "
            "2D grayscale images."
        )

    if original.max() > 1.0:
        original = original / 255.0

    if reconstructed.max() > 1.0:
        reconstructed = reconstructed / 255.0

    original = np.clip(
        original,
        0.0,
        1.0,
    )

    reconstructed = np.clip(
        reconstructed,
        0.0,
        1.0,
    )

    return (
        original,
        reconstructed,
    )


def _validate_mask(
    mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return the missing-region mask as a boolean array."""
    mask = np.asarray(
        mask
    )

    if mask.shape != shape:
        raise ValueError(
            "mask must have the same height and width "
            "as the evaluated images."
        )

    return (
        mask > 0
    )


def masked_mae(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean absolute error inside the missing region only."""
    (
        original,
        reconstructed,
    ) = _validate_images(
        original,
        reconstructed,
    )

    hole = _validate_mask(
        mask,
        original.shape,
    )

    if not hole.any():
        return 0.0

    return float(
        np.mean(
            np.abs(
                original[
                    hole
                ]
                - reconstructed[
                    hole
                ]
            )
        )
    )


def masked_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean squared error inside the missing region only."""
    (
        original,
        reconstructed,
    ) = _validate_images(
        original,
        reconstructed,
    )

    hole = _validate_mask(
        mask,
        original.shape,
    )

    if not hole.any():
        return 0.0

    difference = (
        original[
            hole
        ]
        - reconstructed[
            hole
        ]
    )

    return float(
        np.mean(
            difference ** 2
        )
    )


def masked_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Peak signal-to-noise ratio inside the missing region.

    Higher is better.

    Returns infinity when reconstruction is numerically perfect.
    """
    mse = masked_mse(
        original,
        reconstructed,
        mask,
    )

    if mse <= 1e-12:
        return float(
            "inf"
        )

    return float(
        10.0
        * math.log10(
            1.0
            / mse
        )
    )


def masked_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Approximate SSIM for the masked region.

    SSIM normally operates on a rectangular image rather than an
    arbitrary irregular mask. For EXP-016, we compute SSIM on the
    bounding rectangle containing the missing region.

    This keeps the metric interpretable while ensuring that the
    evaluated crop contains the synthetically removed pixels.

    Higher is better.
    """
    (
        original,
        reconstructed,
    ) = _validate_images(
        original,
        reconstructed,
    )

    hole = _validate_mask(
        mask,
        original.shape,
    )

    if not hole.any():
        return 1.0

    rows, columns = np.where(
        hole
    )

    y1 = int(
        rows.min()
    )

    y2 = int(
        rows.max()
    ) + 1

    x1 = int(
        columns.min()
    )

    x2 = int(
        columns.max()
    ) + 1

    original_crop = original[
        y1:y2,
        x1:x2,
    ]

    reconstructed_crop = reconstructed[
        y1:y2,
        x1:x2,
    ]

    min_dimension = min(
        original_crop.shape
    )

    if min_dimension < 3:
        return float(
            "nan"
        )

    window_size = min(
        7,
        min_dimension,
    )

    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 3:
        return float(
            "nan"
        )

    return float(
        structural_similarity(
            original_crop,
            reconstructed_crop,
            data_range=1.0,
            win_size=window_size,
        )
    )


def full_image_mae(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Mean absolute error across the full image."""
    (
        original,
        reconstructed,
    ) = _validate_images(
        original,
        reconstructed,
    )

    return float(
        np.mean(
            np.abs(
                original
                - reconstructed
            )
        )
    )


def evaluate_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Calculate the core EXP-016 reconstruction metrics.

    The masked metrics are the primary reconstruction measurements
    because unmasked pixels should remain unchanged by an inpainter.
    """
    return {
        "psnr":
            masked_psnr(
                original,
                reconstructed,
                mask,
            ),

        "ssim":
            masked_ssim(
                original,
                reconstructed,
                mask,
            ),

        "mae":
            masked_mae(
                original,
                reconstructed,
                mask,
            ),

        "full_image_mae":
            full_image_mae(
                original,
                reconstructed,
            ),
    }