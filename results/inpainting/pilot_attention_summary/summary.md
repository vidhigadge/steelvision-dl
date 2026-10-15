# EXP-016 Inpainting Reliability Summary

Source CSV: `results/inpainting/pilot_attention.csv`

## Overall

- Cases: **72**
- Class flips: **4**
- Flip rate: **5.56%**
- Reconstructed-case accuracy: **94.44%**
- Mean confidence change: **-0.1702**
- Mean entropy change: **0.4255**

## By Method

| method        |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |    psnr |     ssim |       mae |
|:--------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|--------:|---------:|----------:|
| mean_fill     |      18 |             0 |     0       |                 100      |           -0.10656  |         0.303838 | 19.2117 | 0.609612 | 0.112388  |
| navier_stokes |      18 |             2 |    11.1111  |                  88.8889 |           -0.209116 |         0.489043 | 25.3075 | 0.619457 | 0.0490271 |
| telea         |      18 |             1 |     5.55556 |                  94.4444 |           -0.189569 |         0.463982 | 24.1578 | 0.622394 | 0.0522333 |
| unet          |      18 |             1 |     5.55556 |                  94.4444 |           -0.175742 |         0.445267 | 24.846  | 0.634075 | 0.0497482 |

## By Placement

| placement      |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |    psnr |     ssim |       mae |
|:---------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|--------:|---------:|----------:|
| high_attention |      24 |             4 |     16.6667 |                  83.3333 |           -0.262912 |         0.589689 | 23.4641 | 0.59541  | 0.0639399 |
| low_attention  |      24 |             0 |      0      |                 100      |           -0.108541 |         0.310432 | 23.3376 | 0.63199  | 0.0669767 |
| random         |      24 |             0 |      0      |                 100      |           -0.139289 |         0.376476 | 23.3407 | 0.636753 | 0.066631  |

## By Mask Ratio

|   mask_ratio_target |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|--------------------:|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
|                 0.2 |      72 |             4 |     5.55556 |                  94.4444 |           -0.170247 |         0.425533 |

## By Mask Type

| mask_type   |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|:------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
| block       |      72 |             4 |     5.55556 |                  94.4444 |           -0.170247 |         0.425533 |

## By Class

| ground_truth_class   |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|:---------------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
| crazing              |      12 |             2 |     16.6667 |                  83.3333 |         -0.338759   |        0.931463  |
| inclusion            |      12 |             2 |     16.6667 |                  83.3333 |         -0.264715   |        0.59807   |
| patches              |      12 |             0 |      0      |                 100      |         -0.0137464  |        0.0714376 |
| pitted_surface       |      12 |             0 |      0      |                 100      |         -0.18779    |        0.352848  |
| rolled_in_scale      |      12 |             0 |      0      |                 100      |         -0.213517   |        0.579772  |
| scratches            |      12 |             0 |      0      |                 100      |         -0.00295409 |        0.0196054 |

## Interpretation Notes

- Lower class-flip rate indicates stronger downstream decision preservation.
- Negative confidence change means the classifier became less confident after reconstruction.
- Positive entropy change means classifier uncertainty increased.
- Higher PSNR / SSIM and lower MAE indicate better pixel reconstruction.
- Better reconstruction metrics do not necessarily imply better downstream classification reliability.
- High-attention regions are defined using Grad-CAM model influence, not physical defect ground truth.
- Inpainting is evaluated on synthetically removed regions where the original image is retained as reference.
- Results should not be interpreted as proof that missing real-world defect information can be recovered.
