# EXP-016 Inpainting Reliability Summary

Source CSV: `results/inpainting/medium_30_results.csv`

## Overall

- Cases: **3240**
- Class flips: **467**
- Flip rate: **14.41%**
- Reconstructed-case accuracy: **85.59%**
- Mean confidence change: **-0.1733**
- Mean entropy change: **0.4802**

## By Method

| method        |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |    psnr |     ssim |       mae |
|:--------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|--------:|---------:|----------:|
| mean_fill     |     810 |            88 |     10.8642 |                  89.1358 |           -0.18411  |         0.514339 | 20.1876 | 0.664523 | 0.0973696 |
| navier_stokes |     810 |           131 |     16.1728 |                  83.8272 |           -0.180426 |         0.496321 | 25.6557 | 0.713201 | 0.0483321 |
| telea         |     810 |           127 |     15.679  |                  84.321  |           -0.163177 |         0.454786 | 25.2951 | 0.714005 | 0.0493454 |
| unet          |     810 |           121 |     14.9383 |                  85.0617 |           -0.165494 |         0.455253 | 24.7601 | 0.710344 | 0.0530102 |

## By Placement

| placement      |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |    psnr |     ssim |       mae |
|:---------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|--------:|---------:|----------:|
| high_attention |    1080 |           256 |    23.7037  |                  76.2963 |           -0.247087 |         0.654448 | 23.6799 | 0.696229 | 0.063131  |
| low_attention  |    1080 |            88 |     8.14815 |                  91.8519 |           -0.114192 |         0.33672  | 24.2302 | 0.699798 | 0.0609543 |
| random         |    1080 |           123 |    11.3889  |                  88.6111 |           -0.158626 |         0.449356 | 24.0138 | 0.705528 | 0.0619576 |

## By Mask Ratio

|   mask_ratio_target |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|--------------------:|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
|                 0.1 |    1080 |            20 |     1.85185 |                  98.1481 |          -0.0685291 |         0.213512 |
|                 0.2 |    1080 |           104 |     9.62963 |                  90.3704 |          -0.151508  |         0.440482 |
|                 0.4 |    1080 |           343 |    31.7593  |                  68.2407 |          -0.299868  |         0.786531 |

## By Mask Type

| mask_type   |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|:------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
| blob        |    1080 |           186 |     17.2222 |                  82.7778 |           -0.187345 |         0.508743 |
| block       |    1080 |           131 |     12.1296 |                  87.8704 |           -0.153616 |         0.44119  |
| line        |    1080 |           150 |     13.8889 |                  86.1111 |           -0.178943 |         0.490592 |

## By Class

| ground_truth_class   |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|:---------------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
| crazing              |     540 |           201 |    37.2222  |                  62.7778 |          -0.348361  |        0.952769  |
| inclusion            |     540 |            36 |     6.66667 |                  93.3333 |          -0.160749  |        0.442316  |
| patches              |     540 |             7 |     1.2963  |                  98.7037 |          -0.0584207 |        0.221124  |
| pitted_surface       |     540 |            21 |     3.88889 |                  96.1111 |          -0.132606  |        0.399858  |
| rolled_in_scale      |     540 |           164 |    30.3704  |                  69.6296 |          -0.30688   |        0.793899  |
| scratches            |     540 |            38 |     7.03704 |                  92.963  |          -0.0327944 |        0.0710836 |

## Interpretation Notes

- Lower class-flip rate indicates stronger downstream decision preservation.
- Negative confidence change means the classifier became less confident after reconstruction.
- Positive entropy change means classifier uncertainty increased.
- Higher PSNR / SSIM and lower MAE indicate better pixel reconstruction.
- Better reconstruction metrics do not necessarily imply better downstream classification reliability.
- High-attention regions are defined using Grad-CAM model influence, not physical defect ground truth.
- Inpainting is evaluated on synthetically removed regions where the original image is retained as reference.
- Results should not be interpreted as proof that missing real-world defect information can be recovered.
