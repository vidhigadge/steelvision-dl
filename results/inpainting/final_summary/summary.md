# EXP-016 Inpainting Reliability Summary

Source CSV: `results/inpainting/final_results.csv`

## Overall

- Cases: **27540**
- Class flips: **4247**
- Flip rate: **15.42%**
- Reconstructed-case accuracy: **84.43%**
- Mean confidence change: **-0.1823**
- Mean entropy change: **0.4908**

## By Method

| method        |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |    psnr |     ssim |       mae |
|:--------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|--------:|---------:|----------:|
| mean_fill     |    6885 |           766 |     11.1256 |                  88.947  |           -0.185786 |         0.511068 | 21.0501 | 0.648832 | 0.089015  |
| navier_stokes |    6885 |          1186 |     17.2259 |                  82.5853 |           -0.193031 |         0.512066 | 25.6138 | 0.690154 | 0.0488069 |
| telea         |    6885 |          1195 |     17.3566 |                  82.3965 |           -0.179078 |         0.480705 | 25.2338 | 0.690539 | 0.0498732 |
| unet          |    6885 |          1100 |     15.9768 |                  83.8054 |           -0.17132  |         0.459376 | 24.636  | 0.687804 | 0.0525572 |

## By Placement

| placement      |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |    psnr |     ssim |       mae |
|:---------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|--------:|---------:|----------:|
| high_attention |    9180 |          2076 |    22.6144  |                  77.3203 |           -0.24496  |         0.641812 | 23.7998 | 0.676163 | 0.0611253 |
| low_attention  |    9180 |           892 |     9.71678 |                  90.1307 |           -0.128615 |         0.358267 | 24.4532 | 0.680369 | 0.0590439 |
| random         |    9180 |          1279 |    13.9325  |                  85.8497 |           -0.173337 |         0.472332 | 24.1472 | 0.681464 | 0.06002   |

## By Mask Ratio

|   mask_ratio_target |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|--------------------:|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
|                 0.1 |    9180 |           348 |     3.79085 |                  96.0131 |          -0.0789138 |         0.237918 |
|                 0.2 |    9180 |           923 |    10.0545  |                  89.8148 |          -0.160434  |         0.448431 |
|                 0.4 |    9180 |          2976 |    32.4183  |                  67.4728 |          -0.307564  |         0.786063 |

## By Mask Type

| mask_type   |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|:------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
| blob        |    9180 |          1702 |     18.5403 |                  81.1329 |           -0.18986  |         0.50536  |
| block       |    9180 |          1246 |     13.573  |                  86.3834 |           -0.164208 |         0.454845 |
| line        |    9180 |          1299 |     14.1503 |                  85.7843 |           -0.192843 |         0.512207 |

## By Class

| ground_truth_class   |   cases |   class_flips |   flip_rate |   reconstructed_accuracy |   confidence_change |   entropy_change |
|:---------------------|--------:|--------------:|------------:|-------------------------:|--------------------:|-----------------:|
| crazing              |    4860 |          1937 |    39.856   |                  60.144  |          -0.341852  |        0.898647  |
| inclusion            |    4860 |           410 |     8.43621 |                  91.5638 |          -0.159368  |        0.414801  |
| patches              |    4536 |            97 |     2.13845 |                  97.8616 |          -0.0821423 |        0.262732  |
| pitted_surface       |    4320 |           153 |     3.54167 |                  96.4583 |          -0.133562  |        0.392625  |
| rolled_in_scale      |    4860 |          1312 |    26.9959  |                  73.0041 |          -0.316024  |        0.814226  |
| scratches            |    4104 |           338 |     8.23587 |                  90.7895 |          -0.0241849 |        0.0702603 |

## Interpretation Notes

- Lower class-flip rate indicates stronger downstream decision preservation.
- Negative confidence change means the classifier became less confident after reconstruction.
- Positive entropy change means classifier uncertainty increased.
- Higher PSNR / SSIM and lower MAE indicate better pixel reconstruction.
- Better reconstruction metrics do not necessarily imply better downstream classification reliability.
- High-attention regions are defined using Grad-CAM model influence, not physical defect ground truth.
- Inpainting is evaluated on synthetically removed regions where the original image is retained as reference.
- Results should not be interpreted as proof that missing real-world defect information can be recovered.
