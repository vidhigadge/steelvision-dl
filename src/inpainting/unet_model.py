"""Small U-Net for EXP-016 steel-defect inpainting."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(
    nn.Module
):
    """Two convolution blocks with ReLU activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            x
        )


class DownBlock(
    nn.Module
):
    """Downsample followed by feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pool(
            x
        )

        return self.conv(
            x
        )


class UpBlock(
    nn.Module
):
    """Upsample and fuse with the corresponding encoder feature map."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels
            + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(
            x
        )

        if (
            x.shape[
                -2:
            ]
            != skip.shape[
                -2:
            ]
        ):
            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[
                    -2:
                ],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [
                skip,
                x,
            ],
            dim=1,
        )

        return self.conv(
            x
        )


class InpaintingUNet(
    nn.Module
):
    """Compact U-Net for grayscale steel-surface reconstruction.

    Input channels:
        0 -> masked grayscale image
        1 -> binary missing-region mask

    Output:
        reconstructed grayscale image in [0, 1]
    """

    def __init__(
        self,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.encoder_1 = DoubleConv(
            2,
            base_channels,
        )

        self.encoder_2 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.encoder_3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.encoder_4 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.decoder_4 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.decoder_3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.decoder_2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.decoder_1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(
                base_channels,
                1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        encoder_1 = self.encoder_1(
            x
        )

        encoder_2 = self.encoder_2(
            encoder_1
        )

        encoder_3 = self.encoder_3(
            encoder_2
        )

        encoder_4 = self.encoder_4(
            encoder_3
        )

        bottleneck = self.bottleneck(
            encoder_4
        )

        decoder_4 = self.decoder_4(
            bottleneck,
            encoder_4,
        )

        decoder_3 = self.decoder_3(
            decoder_4,
            encoder_3,
        )

        decoder_2 = self.decoder_2(
            decoder_3,
            encoder_2,
        )

        decoder_1 = self.decoder_1(
            decoder_2,
            encoder_1,
        )

        return self.output_layer(
            decoder_1
        )


def create_inpainting_unet(
    base_channels: int = 32,
) -> InpaintingUNet:
    """Create the default EXP-016 U-Net."""
    return InpaintingUNet(
        base_channels=base_channels
    )