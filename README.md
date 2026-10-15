```markdown
# SteelVision

### AI-Assisted Steel Surface Defect Inspection and Reliability Analysis

SteelVision is an end-to-end deep learning research prototype for **steel surface defect analysis**.

The project goes beyond simple image classification. It combines:

- supported-domain validation
- defect classification
- uncertainty estimation
- explainability
- defect localization
- coordinate mapping
- low-resolution robustness analysis
- super-resolution evaluation
- YOLO ground-truth comparison
- inpainting reliability analysis
- inference optimization
- FastAPI deployment
- Streamlit dashboard integration

The main research focus of SteelVision is not only:

> **Can the model classify a steel defect?**

but also:

> **How reliable is that decision when image information becomes degraded, uncertain, reconstructed, or partially missing?**

---

# Project Motivation

Visual quality inspection is an important part of industrial manufacturing.

Manual surface inspection can be:

- repetitive
- subjective
- difficult to scale
- affected by image quality
- sensitive to lighting and viewing conditions
- difficult to quantify consistently

SteelVision explores how deep learning can support visual inspection while also exposing the limitations and reliability of the AI system.

The project is designed as a **research prototype**, not as a production-certified inspection system.

---

# Supported Defect Classes

SteelVision currently supports six steel surface defect categories:

| Class | Description |
|---|---|
| Crazing | Crack-like surface patterns |
| Inclusion | Embedded or foreign-material irregularities |
| Patches | Patch-like surface defects |
| Pitted Surface | Small pits or depressions |
| Rolled-in Scale | Scale pressed into the surface during rolling |
| Scratches | Linear scratch defects |

---

# System Architecture

```text
                     ┌────────────────────────┐
                     │     Uploaded Image     │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │   Domain Validation    │
                     │   MobileNetV3-Small    │
                     └───────────┬────────────┘
                                 │
                       Supported│Input
                                 ▼
                     ┌────────────────────────┐
                     │     Classification     │
                     │       ResNet-50        │
                     └───────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐   ┌────────────────────┐   ┌─────────────────┐
│ Confidence &    │   │      Grad-CAM      │   │      YOLO       │
│ Entropy         │   │   Explainability   │   │  Localization   │
└─────────────────┘   └────────────────────┘   └────────┬────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │ Coordinate       │
                                             │ Mapping          │
                                             └──────────────────┘
```

Additional research workflows evaluate:

```text
Resolution Degradation
        │
        ▼
Super-Resolution Evaluation

Synthetic Missing Information
        │
        ▼
Inpainting Reconstruction
        │
        ▼
Decision Reliability Analysis
```

---

# Dataset

For classification, SteelVision uses a six-class steel surface defect dataset.

Classification split:

| Split | Images |
|---|---:|
| Training | 1,172 |
| Validation | 250 |
| Test | 255 |
| **Total** | **1,677** |

The held-out test set is kept separate from model training.

---

# Image Preprocessing

The classification pipeline uses:

```text
Grayscale image
      ↓
Replicate to 3 channels
      ↓
Resize to 224 × 224
      ↓
Convert to tensor
      ↓
ImageNet normalization
      ↓
ResNet-50
```

ResNet-50 expects three input channels because the model is initialized from ImageNet-pretrained representations.

---

# 1. Defect Classification

## Baseline CNN

A custom CNN was first trained as a baseline.

The baseline established whether basic convolutional features could learn the six steel defect categories.

---

## Fine-Tuned ResNet-50

The main classifier uses **transfer learning with ResNet-50**.

Transfer learning allows a model pretrained on a large dataset to be adapted to a smaller domain-specific dataset.

### Final Test Performance

```text
Test Accuracy: 99.22%
Correct:       253 / 255
```

This result applies to the held-out project test set.

It should not be interpreted as guaranteed industrial production accuracy.

---

# 2. Supported-Domain Validation

A closed-set classifier will always select one of its known classes, even when given an unrelated image.

To reduce this problem, SteelVision includes a separate:

> **MobileNetV3-Small supported-domain validator**

The validator determines whether the uploaded image sufficiently resembles the image domain supported by SteelVision.

### Deployment threshold

```text
Supported-domain probability ≥ 0.80
```

If the image does not pass the domain gate, downstream classification and localization are stopped.

The validator should not be interpreted as a universal steel/non-steel detector.

---

# 3. Confidence and Uncertainty

SteelVision reports:

- classification confidence
- entropy

## Confidence

Confidence is based on the classifier softmax output.

Example:

```text
Scratches
Confidence: 96.4%
```

---

## Entropy

Entropy is used as an uncertainty indicator.

A concentrated probability distribution produces lower entropy.

A more distributed probability output produces higher entropy.

Conceptually:

```text
Low entropy  → more concentrated model belief
High entropy → greater prediction uncertainty
```

Entropy does not guarantee whether a prediction is correct.

---

# 4. Calibration

SteelVision also evaluates confidence calibration using:

> **Expected Calibration Error — ECE**

Observed result:

```text
ECE = 0.0482
```

Calibration measures how closely model confidence corresponds to observed correctness.

---

# 5. Resolution Robustness Experiment

SteelVision evaluates how classification behaves when image detail is deliberately reduced.

Images were evaluated at:

```text
224 × 224
128 × 128
64 × 64
32 × 32
```

### Classification results

| Resolution | Accuracy |
|---|---:|
| 224 | 99.61% |
| 128 | 83.92% |
| 64 | 49.80% |
| 32 | 31.76% |

The experiment demonstrates that classification reliability decreases significantly when visual information is removed.

The resolution experiment does not estimate physical camera distance.

Resizing a degraded image back to 224 × 224 does not restore the lost information.

---

# 6. Super-Resolution Experiment

SteelVision tested whether neural super-resolution could improve downstream classification.

The experiment used:

> **Real-ESRGAN**

and compared it with standard bicubic upscaling.

### Results

| Source Resolution | Bicubic Accuracy | Real-ESRGAN Accuracy |
|---|---:|---:|
| 128 | 83.92% | 63.53% |
| 64 | 49.80% | 41.18% |
| 32 | 31.76% | 31.37% |

Real-ESRGAN did not improve downstream classification in these experiments.

### Important finding

> **A visually sharper image is not necessarily a more reliable input for an AI inspection model.**

Super-resolution may generate plausible visual detail, but that detail is not guaranteed to represent the original physical surface.

---

# 7. Defect Localization with YOLO

ResNet-50 performs image-level classification.

YOLO provides spatial defect localization.

YOLO outputs:

```text
Class
Confidence
Bounding Box
```

SteelVision therefore performs:

> **Object detection**

not segmentation.

No pixel-level segmentation masks are generated.

---

# 8. YOLO Detection Performance

Overall detector metrics:

```text
Precision:     0.672
Recall:        0.830
mAP@50:        0.801
mAP@50-95:     0.505
```

### Precision

```text
TP / (TP + FP)
```

Measures how many predicted detections are correct.

### Recall

```text
TP / (TP + FN)
```

Measures how many annotated defects are detected.

### IoU

Intersection over Union measures bounding-box overlap.

```text
IoU = 1.0 → perfect overlap
IoU = 0.0 → no overlap
```

---

# 9. YOLO Ground-Truth Evaluation

SteelVision includes a custom fixed-threshold detector evaluation.

Evaluation settings:

```text
Confidence Threshold = 0.25
IoU Threshold        = 0.50
```

Observed evaluation result:

```text
Ground-Truth Boxes: 64
Predicted Boxes:    62
True Positives:     43
False Positives:    19
False Negatives:    21

Precision:           0.6935
Recall:              0.6719
Mean TP IoU:         0.8022
```

The custom fixed-threshold analysis complements the detector mAP evaluation.

It does not replace mAP.

---

# Ground Truth in the Dashboard

The standalone YOLO evaluation page was removed from the final presentation dashboard.

Ground-truth comparison is now integrated directly into:

> **Standard Analysis → Defect Localization**

For labelled YOLO validation images:

```text
CYAN = Ground Truth
RED  = YOLO Prediction
```

Ground truth is displayed only when a genuine annotation exists for the uploaded validation image.

SteelVision does not generate or invent ground truth for arbitrary uploaded images.

---

# 10. Coordinate Mapping

The localization pipeline can process the image at a different resolution from the uploaded image.

Bounding boxes are therefore mapped back to original-image coordinates.

Conceptually:

```text
scale_x = original_width  / processed_width
scale_y = original_height / processed_height
```

The predicted coordinates are rescaled using these factors.

This allows localization results to be visualized correctly on the uploaded image.

---

# 11. Cross-Model Agreement

SteelVision also investigates consistency between:

- ResNet-50 classification
- YOLO detection

Observed results:

```text
ResNet-50 classification accuracy: 99.22%
YOLO top-class accuracy:           87.84%
```

YOLO produced at least one detection for:

```text
225 / 255 test images
```

Among those detected cases:

```text
225 / 225
```

contained the ResNet-predicted class somewhere among YOLO detections.

This is a conditional agreement result.

A missing YOLO detection does not automatically mean the ResNet prediction is wrong.

---

# 12. Grad-CAM Explainability

SteelVision uses:

> **Gradient-weighted Class Activation Mapping — Grad-CAM**

Grad-CAM produces a heatmap showing which image regions influenced the ResNet-50 prediction.

Important interpretation:

> Grad-CAM shows **model influence**, not physical defect ground truth.

A highlighted region should not be interpreted as causal proof of a real defect mechanism.

---

# 13. SHAP Explainability

SteelVision also explores:

> **SHAP — SHapley Additive exPlanations**

SHAP provides another way to study feature contribution and understand how image regions affect classifier predictions.

Grad-CAM and SHAP are explainability methods.

They are not additional training targets.

---

# 14. Inpainting Reliability — EXP-016

One of the major SteelVision extensions studies model robustness when image information is synthetically removed.

## Research Question

> **When visual information is synthetically removed from a steel-defect image, can inpainting reconstruct it well enough to preserve downstream classification behavior, and does visual reconstruction quality correlate with decision reliability?**

This experiment does not claim that SteelVision can recover unknown physical information from a genuinely damaged image.

Instead:

1. start with the intact image
2. deliberately remove known pixels
3. reconstruct the removed region
4. compare against the original image
5. test whether the classifier decision remains stable

---

# Inpainting Mask Types

Three missing-region shapes are evaluated:

```text
Block
Line
Blob
```

Mask convention:

```text
255 = missing region
0   = retained region
```

---

# Missing Area

Three approximate missing-area levels are evaluated:

```text
10%
20%
40%
```

Larger missing areas remove more visual information and therefore test stronger corruption conditions.

---

# Mask Placement

Three placement strategies are evaluated:

```text
Random
High Attention
Low Attention
```

## High Attention

The synthetic missing region is placed near an area with strong Grad-CAM activation.

## Low Attention

The region is placed near lower Grad-CAM activation.

These terms mean:

> high or low **model influence**

not:

> defect / non-defect ground truth.

---

# Inpainting Methods

Four reconstruction approaches are compared.

## Mean Fill

A simple non-inpainting baseline.

The missing area is filled using a mean-intensity reference.

## Telea

A classical OpenCV inpainting method.

## Navier-Stokes

A classical OpenCV inpainting method based on propagating surrounding image structure.

## U-Net

A learned domain-specific image reconstruction model.

The U-Net receives:

```text
Masked grayscale image
+
Binary mask
```

and predicts the reconstructed image.

---

# U-Net Training

The compact U-Net was trained for 10 epochs.

Validation loss improved from approximately:

```text
0.1327
```

to:

```text
0.0575
```

The best checkpoint is stored as:

```text
models/inpainting_unet.pth
```

---

# Inpainting Evaluation Metrics

SteelVision evaluates both image reconstruction and model decision reliability.

## Reconstruction Metrics

### PSNR

Peak Signal-to-Noise Ratio.

Higher values generally indicate closer pixel reconstruction.

### SSIM

Structural Similarity Index.

Higher values generally indicate greater structural similarity.

### MAE

Mean Absolute Error.

Lower values indicate smaller average pixel differences.

---

# Decision Reliability Metrics

## Class Flip

A class flip occurs when:

```text
Original predicted class
≠
Reconstructed-image predicted class
```

For example:

```text
Original:      Scratches
Reconstructed: Inclusion

→ Class Flip
```

---

## Flip Rate

```text
Flip Rate =
Number of Class Flips
---------------------
Total Cases
× 100
```

Lower flip rate means greater decision stability.

Flip rate is not the same as classification accuracy.

A prediction can flip from an incorrect class to the correct class and still count as a class flip.

---

# Full Inpainting Experiment

The final experiment evaluated:

```text
255 test images
× 3 mask types
× 3 mask ratios
× 3 placements
× 4 reconstruction methods
```

Total:

```text
27,540 evaluation cases
```

Each reconstruction method receives the same exact mask for a given experimental condition.

---

# Final Placement Results

| Placement | Cases | Flip Rate | Reconstructed Accuracy |
|---|---:|---:|---:|
| High Attention | 9,180 | 22.61% | 77.32% |
| Random | 9,180 | 13.93% | 85.85% |
| Low Attention | 9,180 | 9.72% | 90.13% |

Additional average changes:

| Placement | Confidence Change | Entropy Change |
|---|---:|---:|
| High Attention | -0.2450 | +0.6418 |
| Random | -0.1733 | +0.4723 |
| Low Attention | -0.1286 | +0.3583 |

### Interpretation

High-attention masking caused:

- the highest class-flip rate
- the largest confidence decrease
- the largest entropy increase
- the lowest reconstructed accuracy

This suggests that:

> **Where visual information is removed matters, not only how much information is removed.**

Because placement is based on Grad-CAM, this should be interpreted as sensitivity to model-influential regions rather than confirmed physical defect regions.

---

# Final Reconstruction-Method Results

| Method | Cases | Flip Rate | Accuracy | PSNR | SSIM | MAE |
|---|---:|---:|---:|---:|---:|---:|
| Mean Fill | 6,885 | **11.13%** | **88.95%** | 21.05 | 0.649 | 0.0890 |
| U-Net | 6,885 | 15.98% | 83.81% | 24.64 | 0.688 | 0.0526 |
| Navier-Stokes | 6,885 | 17.23% | 82.59% | **25.61** | 0.690 | **0.0488** |
| Telea | 6,885 | 17.36% | 82.40% | 25.23 | **0.691** | 0.0499 |

### Important Result

Mean Fill produced worse image-reconstruction metrics but the lowest class-flip rate.

The more sophisticated methods often reconstructed pixels more closely, but changed classifier decisions more often.

This demonstrates:

> **Better reconstruction quality does not automatically imply better downstream classification reliability.**

---

# Combined Restoration Finding

The super-resolution and inpainting experiments point to the same broader conclusion:

> **A visually improved image is not automatically a more reliable input for an AI inspection model.**

Super-resolution:

```text
Sharper-looking image
≠
Higher classification accuracy
```

Inpainting:

```text
Better PSNR / SSIM / MAE
≠
Better classifier stability
```

SteelVision therefore evaluates:

> image quality

and

> downstream AI decision reliability

as separate objectives.

---

# 15. Model Optimization

SteelVision evaluates deployment performance using ONNX Runtime.

Observed CPU inference:

```text
PyTorch:      ~19.883 ms / image
ONNX Runtime: ~12.590 ms / image
```

Approximate speedup:

```text
1.58×
```

FP32 ONNX Runtime is therefore used for deployment-oriented classification inference.

---

# INT8 Quantization

INT8 quantization was also evaluated.

Observed classification accuracy:

```text
92.94%
```

Because this produced a substantial loss in accuracy, the INT8 model was not selected as the primary deployment model.

---

# 16. FastAPI Backend

SteelVision separates model logic from the user interface using FastAPI.

The backend provides services for tasks such as:

- supported-domain validation
- defect classification
- Grad-CAM explainability
- defect localization

Run the API from the project root:

```bash
uvicorn src.api.main:app --reload
```

Default backend:

```text
http://127.0.0.1:8000
```

---

# 17. Streamlit Dashboard

The SteelVision frontend is implemented with Streamlit.

Current workflows:

```text
Standard Analysis
Resolution Experiment
Inpainting Reliability
```

---

# Standard Analysis

The main inspection workflow performs:

```text
Upload Image
     ↓
Domain Validation
     ↓
ResNet-50 Classification
     ↓
Confidence + Entropy
     ↓
Grad-CAM
     ↓
YOLO Localization
     ↓
Ground-Truth Comparison
when annotation is available
```

Ground-truth display convention:

```text
Cyan = Ground Truth
Red  = YOLO Prediction
```

---

# Resolution Experiment

This workflow allows controlled evaluation of different resolution levels.

It is designed for robustness analysis and does not estimate real camera distance.

---

# Inpainting Reliability

This workflow allows the user to interactively select:

- mask shape
- missing area
- placement
- reconstruction method

The dashboard displays:

- original image
- masked image
- reconstructed image
- Grad-CAM attention
- before/after prediction
- class preservation or class flip
- confidence change
- entropy change
- PSNR
- SSIM
- MAE
- all-method comparison

---

# Project Structure

```text
steelvision-dl/
│
├── data/
│   ├── raw/
│   └── splits/
│
├── models/
│   ├── resnet50_finetune.pth
│   └── inpainting_unet.pth
│
├── results/
│   └── inpainting/
│       ├── final_results.csv
│       ├── final_summary/
│       ├── medium_30_results.csv
│       └── pilot experiments
│
├── runs/
│   └── detect/
│
├── src/
│   │
│   ├── api/
│   │   └── main.py
│   │
│   ├── evaluation/
│   │   ├── compare_yolo_ground_truth.py
│   │   ├── evaluate_yolo_ground_truth.py
│   │   ├── visualize_yolo_ground_truth.py
│   │   └── yolo_ground_truth_utils.py
│   │
│   ├── inpainting/
│   │   ├── common.py
│   │   ├── evaluate_inpainting.py
│   │   ├── inpainters.py
│   │   ├── masks.py
│   │   ├── metrics.py
│   │   ├── summarize_results.py
│   │   ├── train_unet_inpaint.py
│   │   └── unet_model.py
│   │
│   └── ui/
│       └── app.py
│
├── experiments/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Running SteelVision

## 1. Clone the repository

```bash
git clone https://github.com/vidhigadge/steelvision-dl.git
cd steelvision-dl
```

---

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start the FastAPI backend

Open one terminal:

```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

---

## 5. Start the Streamlit dashboard

Open another terminal:

```bash
source .venv/bin/activate
streamlit run src/ui/app.py
```

---

# Main Technologies

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Classification | ResNet-50 |
| Domain Validation | MobileNetV3-Small |
| Object Detection | YOLO |
| Explainability | Grad-CAM, SHAP |
| Learned Inpainting | U-Net |
| Classical Inpainting | OpenCV Telea, Navier-Stokes |
| Super-Resolution | Real-ESRGAN |
| Optimized Inference | ONNX Runtime |
| API | FastAPI |
| Dashboard | Streamlit |
| Image Processing | OpenCV, Pillow, NumPy |
| Evaluation | scikit-image, custom metrics |

---

# Key Results

## Classification

```text
ResNet-50 Test Accuracy
99.22%
```

---

## Resolution Robustness

```text
224 → 99.61%
128 → 83.92%
64  → 49.80%
32  → 31.76%
```

---

## Super-Resolution

```text
Real-ESRGAN did not outperform bicubic upscaling
for downstream classification.
```

---

## YOLO

```text
mAP@50:     0.801
mAP@50-95:  0.505
Recall:      0.830
```

---

## Fixed-Threshold YOLO Evaluation

```text
Precision:   0.6935
Recall:      0.6719
Mean TP IoU: 0.8022
```

---

## Inpainting Placement

```text
High Attention Flip Rate: 22.61%
Random Flip Rate:         13.93%
Low Attention Flip Rate:   9.72%
```

---

## Inpainting Reconstruction Methods

```text
Mean Fill Flip Rate:       11.13%
U-Net Flip Rate:           15.98%
Navier-Stokes Flip Rate:   17.23%
Telea Flip Rate:           17.36%
```

---

## Optimization

```text
ONNX CPU Speedup: ~1.58×
```

---

# Key Research Findings

SteelVision produced several important findings.

### 1. Image resolution strongly affects classification reliability.

Reducing the available visual information caused substantial accuracy degradation.

### 2. Neural super-resolution did not improve downstream classification.

A sharper reconstructed image was not necessarily a more useful classifier input.

### 3. Model-influential image regions matter strongly.

Removing high-Grad-CAM-attention regions produced more class flips and greater uncertainty than removing low-attention regions.

### 4. Reconstruction quality and decision reliability are different objectives.

Mean Fill had poorer image-quality metrics but better classifier stability than more sophisticated inpainting methods.

### 5. AI inspection systems should expose uncertainty and failure conditions.

A high headline accuracy alone is not sufficient for understanding model reliability.

---

# Scientific Scope

SteelVision is a research prototype.

The following claims are intentionally avoided:

- production-ready inspection
- universal steel/non-steel detection
- guaranteed recovery of missing physical information
- causal defect explanations from Grad-CAM
- segmentation capability
- guaranteed reliability on unseen manufacturing environments

The current results are based on the project datasets and experimental conditions.

---

# Current Limitations

- limited to six supervised defect classes
- based primarily on public steel-surface defect data
- no pixel-level segmentation
- no universal normal/no-defect recognition system
- Grad-CAM measures model influence rather than defect ground truth
- inpainting corruption is synthetic
- super-resolution can hallucinate plausible detail
- deployment has not been validated on a real production line
- industrial camera, illumination and environmental variation require further evaluation
- engineering defect acceptance criteria are not currently integrated

---

# Future Work

Potential extensions include:

- company-specific industrial datasets
- normal/no-defect recognition
- pixel-level segmentation
- additional object detectors
- calibrated rejection mechanisms
- stronger out-of-distribution detection
- uncertainty-aware human review
- improved inpainting architectures
- LaMa-style inpainting
- transformer-based restoration
- SwinIR super-resolution
- model distillation
- improved quantization
- edge-device deployment
- live industrial camera integration
- production-line latency studies
- defect severity models using engineering-labelled severity data
- larger-scale cross-domain validation

---

# Manufacturing Use Case

A future industrial workflow could resemble:

```text
Industrial Component
        ↓
Inspection Camera
        ↓
Supported-Domain Validation
        ↓
Defect Classification
        ↓
Localization
        ↓
Confidence / Uncertainty Check
        ↓
AI-Assisted Quality Review
        ↓
Accept / Rework / Reject
```

SteelVision is intended to support engineers, not replace engineering judgement.

---

# Project Summary

SteelVision demonstrates how an AI-assisted inspection project can move beyond a single classification accuracy number.

The system evaluates:

```text
What defect?
Where is it?
How confident is the model?
What influenced the decision?
Does another model support the result?
What happens when image quality drops?
Does reconstruction actually help?
How stable is the decision when information is missing?
```

The central research lesson from SteelVision is:

> **A visually better image is not automatically a better input for an AI inspection model. Image restoration quality and downstream decision reliability must be evaluated separately.**

---

# Disclaimer

SteelVision is an academic and research prototype.

It has not been certified for industrial safety, quality-control acceptance, or production decision-making.

All predictions should be interpreted within the scope of the datasets, models, thresholds, and experiments used in this project.

---

## Repository

**SteelVision**

GitHub:  
https://github.com/vidhigadge/steelvision-dl
```