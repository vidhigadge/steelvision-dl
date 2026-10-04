# SteelVision

Deep learning research system for steel surface defect classification, explainability, localization, uncertainty analysis, and controlled resolution experiments.

SteelVision combines a fine-tuned ResNet-50 classifier, YOLO defect localization, Grad-CAM explainability, predictive uncertainty, coordinate mapping, ONNX deployment, a FastAPI backend, and a Streamlit interface into one end-to-end research prototype.

---

## Project Overview

Steel surface defect inspection becomes more difficult when visual detail is reduced by lower image resolution.

This project investigates how deep learning models behave under degraded-resolution conditions and whether enhancement techniques such as neural super-resolution improve downstream defect recognition.

The project also extends beyond classification by integrating:

- defect localization
- explainable AI
- uncertainty analysis
- model optimization
- coordinate mapping
- API deployment
- an interactive user interface

The final application is called:

```text
SteelVision
```

The GitHub repository remains:

```text
steelvision-dl
```

---

## Main Research Question

The central research question is:

> Does neural super-resolution improve downstream steel defect classification under degraded image resolution compared with standard bicubic upscaling?

The experiments also investigate:

- how classification accuracy changes as image resolution decreases
- whether visually sharper super-resolution images improve model performance
- how defect localization changes under controlled resolution reduction
- whether model confidence reflects actual image quality
- how explainability and uncertainty can improve interpretation of model outputs
- whether ONNX optimization can improve inference speed without reducing accuracy

---

## Supported Defect Classes

SteelVision works with six steel surface defect categories:

```text
crazing
inclusion
patches
pitted_surface
rolled_in_scale
scratches
```

---

# System Capabilities

The final SteelVision system provides:

```text
Classification
Uncertainty Estimation
Grad-CAM Explainability
Defect Localization
Coordinate Mapping
Resolution Experimentation
```

The application contains two separate workflows:

```text
Standard Analysis
Resolution Experiment
```

---

# Standard Analysis

Standard Analysis is the normal user-facing workflow.

The uploaded image is analyzed as provided.

The pipeline performs:

```text
image upload
        ↓
ResNet-50 classification
        ↓
confidence
        ↓
predictive entropy
        ↓
top defect predictions
        ↓
Grad-CAM explanation
        ↓
YOLO defect localization
        ↓
coordinate mapping
        ↓
mapped defect regions
```

The user does not need to manually specify the image resolution.

---

# Resolution Experiment

Resolution Experiment is a research workflow.

It intentionally adjusts the same uploaded image to a controlled resolution and compares localization behavior with the baseline image.

Available experimental resolutions are:

```text
224
128
64
32
```

The experiment evaluates how changes in available visual detail affect:

```text
detection count
detector confidence
bounding-box localization
```

The selected resolution is an experimental setting.

It does not estimate physical camera distance.

---

# Dataset

The project uses steel surface defect data organized into six defect classes.

For classification, the final dataset contained:

```text
1677 images
```

The classification split was created using:

```text
70% training
15% validation
15% testing
```

with:

```text
random seed = 42
```

Final split sizes:

| Split | Images |
|---|---:|
| Training | 1172 |
| Validation | 250 |
| Test | 255 |
| Total | 1677 |

Class-wise distribution:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Crazing | 206 | 44 | 45 | 295 |
| Inclusion | 210 | 45 | 45 | 300 |
| Patches | 189 | 40 | 42 | 271 |
| Pitted Surface | 182 | 39 | 40 | 261 |
| Rolled-in Scale | 210 | 45 | 45 | 300 |
| Scratches | 175 | 37 | 38 | 250 |

---

# Classification Preprocessing

Images are processed using:

```text
grayscale conversion
↓
3-channel representation
↓
resize to 224 × 224
↓
tensor conversion
↓
ImageNet normalization
```

Normalization parameters:

```text
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]
```

The same preprocessing pipeline is used during training, evaluation, and deployment.

---

# Classification Models

## Baseline CNN

A small custom convolutional neural network was first trained as a baseline.

The model uses convolutional blocks followed by pooling and fully connected layers.

This provided an initial comparison point before transfer learning.

---

## ResNet-50 Transfer Learning

A pretrained ResNet-50 was then evaluated using ImageNet weights.

The original backbone was initially frozen and the final classification layer was replaced for the six steel defect classes.

Frozen ResNet-50 test accuracy:

```text
98.82%
```

---

## ResNet-50 Partial Fine-Tuning

The final classifier uses partial fine-tuning.

Training strategy:

```text
freeze most pretrained layers
unfreeze layer4
replace final classification layer
train selected parameters
```

Final test result:

```text
253 / 255 correct
```

Test accuracy:

```text
99.22%
```

This model became the main classification model used throughout the later phases.

---

# Resolution Degradation Study

The project evaluated classification performance as image resolution was reduced.

The experiment followed:

```text
original test image
↓
bicubic downsample
↓
bicubic resize back to 224 × 224
↓
ResNet-50 classification
```

Results:

| Resolution | Classification Accuracy |
|---|---:|
| 224 × 224 | 99.61% |
| 128 × 128 | 83.92% |
| 64 × 64 | 49.80% |
| 32 × 32 | 31.76% |

Accuracy drop relative to 224:

| Resolution | Accuracy Drop |
|---|---:|
| 128 | 15.69 percentage points |
| 64 | 49.81 percentage points |
| 32 | 67.85 percentage points |

The results show that classification performance decreases sharply as visual detail is removed.

Upscaling the image dimensions afterward does not restore the information lost during downsampling.

---

# Super-Resolution Study

Neural super-resolution was evaluated using Real-ESRGAN.

The controlled pipeline compared:

```text
low-resolution image
↓
bicubic reconstruction

versus

low-resolution image
↓
Real-ESRGAN reconstruction
```

The same ResNet-50 classifier was then used for downstream evaluation.

---

## Super-Resolution Results

| Resolution | Bicubic Accuracy | Real-ESRGAN Accuracy | Difference |
|---|---:|---:|---:|
| 128 | 83.92% | 63.53% | -20.39 pp |
| 64 | 49.80% | 41.18% | -8.62 pp |
| 32 | 31.76% | 31.37% | -0.39 pp |

Real-ESRGAN did not improve downstream classification performance in this experiment.

At every tested resolution, the neural super-resolution output performed worse than standard bicubic reconstruction.

---

## Super-Resolution Interpretation

A visually sharper image does not necessarily contain more useful discriminative information for a classifier.

Possible explanations include:

```text
domain shift
artificial texture generation
feature distortion
hallucinated detail
```

Therefore:

```text
visual sharpness
≠
better classification accuracy
```

and:

```text
super-resolution
≠
guaranteed ground-truth recovery
```

The original image remains the authoritative observation.

---

# Defect Localization

SteelVision uses a YOLO detector for defect localization.

The detector was trained on a YOLO-formatted steel surface defect dataset.

Training data:

```text
1770 training images
30 validation images
```

Validation set:

```text
64 annotated defect instances
```

Model:

```text
YOLO26n
```

Training configuration:

```text
50 epochs
image size = 640
Apple MPS acceleration
```

---

## Detection Results

Overall validation results:

| Metric | Result |
|---|---:|
| Precision | 0.672 |
| Recall | 0.830 |
| mAP@50 | 0.801 |
| mAP@50-95 | 0.505 |

Class-wise results:

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Crazing | 0.645 | 0.625 | 0.756 | 0.441 |
| Inclusion | 0.526 | 0.888 | 0.693 | 0.353 |
| Patches | 0.822 | 0.941 | 0.951 | 0.618 |
| Pitted Surface | 0.959 | 1.000 | 0.995 | 0.603 |
| Rolled-in Scale | 0.501 | 0.667 | 0.584 | 0.317 |
| Scratches | 0.578 | 0.857 | 0.828 | 0.697 |

Observed detector inference time:

```text
approximately 6.9 ms / image
```

---

# Coordinate Mapping

YOLO detections are produced in processed-image coordinates.

SteelVision maps these bounding boxes back to the uploaded image coordinate space.

For:

```text
processed_width
processed_height
original_width
original_height
```

the scaling factors are:

```text
scale_x = original_width / processed_width

scale_y = original_height / processed_height
```

Processed coordinates:

```text
(x1, y1, x2, y2)
```

are mapped using:

```text
mapped_x1 = x1 × scale_x
mapped_y1 = y1 × scale_y

mapped_x2 = x2 × scale_x
mapped_y2 = y2 × scale_y
```

This allows detected regions to be visualized correctly on the uploaded image.

The current method assumes resize-based geometric scaling.

It does not automatically handle arbitrary:

```text
cropping
rotation
perspective transformation
complex geometric warping
```

---

# Explainable AI

SteelVision includes two explainability techniques:

```text
Grad-CAM
SHAP
```

---

## Grad-CAM

Grad-CAM is applied to the fine-tuned ResNet-50 classifier.

Target layer:

```text
model.layer4[-1]
```

Grad-CAM highlights image regions that influenced the predicted class.

Example validation images from all six classes produced correct predictions during the explainability experiment.

Grad-CAM should be interpreted as:

```text
model influence
```

not:

```text
physical proof of a defect mechanism
```

---

## SHAP

SHAP was also evaluated using:

```text
shap.GradientExplainer
```

A balanced background set was created from training images.

The SHAP experiment provides feature-attribution analysis for classifier behavior.

SHAP is used as an explainability method and is not part of model training.

---

# Uncertainty and Reliability

SteelVision evaluates predictive confidence and entropy.

On the 255-image test set:

```text
Correct predictions:   253

Incorrect predictions: 2
```

Overall:

| Metric | Result |
|---|---:|
| Accuracy | 99.22% |
| Mean Confidence | 0.9471 |
| Mean Entropy | 0.2083 |
| Expected Calibration Error | 0.0482 |

Correct predictions:

```text
mean confidence = 0.9503
mean entropy    = 0.2019
```

Incorrect predictions:

```text
mean confidence = 0.5440
mean entropy    = 1.0096
```

The incorrect predictions were associated with substantially higher uncertainty.

The model was slightly underconfident overall because its mean confidence was below its observed accuracy.

---

## Entropy Interpretation

The Streamlit interface uses simple interface-level thresholds:

```text
Entropy < 0.5
→ Low uncertainty

0.5 ≤ Entropy ≤ 1.0
→ Moderate uncertainty

Entropy > 1.0
→ Elevated uncertainty
```

These thresholds are intended only for user-interface interpretation.

They are not universal scientific uncertainty thresholds.

---

# ONNX Optimization

The final ResNet-50 model was exported to ONNX.

PyTorch and ONNX outputs were validated on the same input.

Example comparison:

```text
PyTorch confidence:
0.967897

ONNX confidence:
0.967895

Maximum probability difference:
0.00000232
```

Prediction agreement:

```text
True
```

---

## CPU Inference Benchmark

Benchmark configuration:

```text
20 warm-up runs
100 measured runs
CPU inference
```

Results:

| Runtime | Latency |
|---|---:|
| PyTorch | 19.883 ms / image |
| ONNX Runtime | 12.590 ms / image |

Observed ONNX speedup:

```text
1.58×
```

The FP32 ONNX model preserved classifier accuracy while improving CPU inference speed.

---

# INT8 Quantization

Static INT8 quantization was also evaluated.

The INT8 model reduced model size substantially but produced unacceptable accuracy degradation.

Tested configurations included:

```text
QInt8 / QInt8
balanced calibration
U8 / U8 preprocessing-aware calibration
```

Observed accuracies included:

```text
92.94%
54.90%
61.57%
```

compared with:

```text
99.22%
```

for the FP32 classifier.

Therefore, the final deployment decision was:

```text
retain FP32 ONNX
```

rather than use INT8 quantization.

---

# FastAPI Backend

SteelVision uses FastAPI as the inference backend.

Main file:

```text
src/api/main.py
```

Available endpoints:

```text
GET  /
GET  /health
POST /predict
POST /explain
POST /localize
```

---

## `/predict`

Uses:

```text
FP32 ONNX ResNet-50
```

Returns:

```text
predicted class
confidence
entropy
top-3 predictions
```

---

## `/explain`

Uses:

```text
PyTorch ResNet-50
```

to generate live Grad-CAM explanations.

The PyTorch model is retained because Grad-CAM requires access to internal convolutional activations.

---

## `/localize`

Uses:

```text
YOLO
```

for defect localization.

It also performs:

```text
coordinate mapping
resolution experimentation
```

Supported resolution parameters:

```text
original
224
128
64
32
```

---

# Streamlit Interface

The user-facing application is implemented in:

```text
src/ui/app.py
```

The dashboard integrates:

```text
classification
confidence
entropy
Grad-CAM
defect localization
coordinate mapping
resolution experiments
backend status
model information
```

The interface uses a dark industrial design intended for technical inspection workflows.

---

# Application Architecture

```text
                    SteelVision
                         │
                         ▼
                Streamlit Frontend
                         │
                         ▼
                  FastAPI Backend
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   ONNX ResNet-50   PyTorch ResNet-50   YOLO
          │              │              │
          ▼              ▼              ▼
 Classification      Grad-CAM       Localization
          │                              │
          ▼                              ▼
 Confidence                     Bounding Boxes
 Entropy                              │
 Top Predictions                      ▼
                               Coordinate Mapping
          │                              │
          └──────────────┬───────────────┘
                         ▼
                SteelVision Dashboard
```

---

# Project Structure

```text
steelvision-dl/
│
├── data/
│   ├── detection.yaml
│   └── splits/
│
├── experiments/
│   ├── EXP-001.md
│   ├── EXP-002.md
│   ├── EXP-003.md
│   ├── EXP-004.md
│   ├── EXP-005.md
│   ├── EXP-006.md
│   ├── EXP-007.md
│   ├── EXP-008.md
│   ├── EXP-009.md
│   ├── EXP-010.md
│   ├── EXP-011.md
│   ├── EXP-012.md
│   └── EXP-013.md
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── data/
│   │   ├── create_splits.py
│   │   ├── dataset_stats.py
│   │   ├── degradation.py
│   │   └── loaders.py
│   │
│   ├── detection/
│   │   ├── coordinate_mapping.py
│   │   ├── localization_service.py
│   │   └── test_mapping.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_cnn.py
│   │   ├── evaluate_resnet.py
│   │   ├── evaluate_resnet_finetune.py
│   │   ├── evaluate_resolution.py
│   │   └── evaluate_sr.py
│   │
│   ├── explainability/
│   │   ├── gradcam.py
│   │   ├── gradcam_service.py
│   │   └── shap_explain.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   ├── resnet50.py
│   │   └── resnet50_finetune.py
│   │
│   ├── optimization/
│   │   ├── benchmark_inference.py
│   │   ├── evaluate_int8.py
│   │   ├── export_onnx.py
│   │   ├── quantize_onnx.py
│   │   └── validate_onnx.py
│   │
│   ├── reliability/
│   │   ├── calibration.py
│   │   └── uncertainty.py
│   │
│   ├── super_resolution/
│   │   └── generate_lr.py
│   │
│   ├── training/
│   │   ├── train_cnn.py
│   │   ├── train_resnet.py
│   │   └── train_resnet_finetune.py
│   │
│   └── ui/
│       └── app.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Experiment Log

The project maintains a complete experiment record.

| Experiment | Description |
|---|---|
| EXP-001 | Baseline CNN |
| EXP-002 | ResNet-50 Transfer Learning |
| EXP-003 | ResNet-50 Partial Fine-Tuning |
| EXP-004 | Resolution Degradation |
| EXP-005 | Bicubic vs Real-ESRGAN at 64×64 |
| EXP-006 | Super-Resolution Downstream Evaluation |
| EXP-007 | Steel Surface Defect Detection with YOLO26n |
| EXP-008 | Bounding Box Coordinate Mapping to Original Image Space |
| EXP-009 | Explainable AI with Grad-CAM and SHAP |
| EXP-010 | Uncertainty and Reliability Analysis |
| EXP-011 | Model Export, Optimization, and Quantization |
| EXP-012 | FastAPI Inference Backend |
| EXP-013 | Streamlit Application Integration |

Detailed methodology and results are available in the `experiments/` directory.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/vidhigadge/steelvision-dl.git
cd steelvision-dl
```

---

## 2. Create a Virtual Environment

The project was developed and tested using Python 3.13.

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Required Model Files

Large trained model artifacts are not stored in GitHub.

The `.gitignore` excludes:

```text
*.pth
*.pt
*.onnx
runs/
```

To run the complete application, the following model files must exist locally.

---

## Classification Checkpoint

Required for Grad-CAM:

```text
models/resnet50_finetune.pth
```

---

## ONNX Classifier

Required for `/predict`:

```text
models/resnet50_finetune.onnx
```

It can be generated using:

```bash
python -m src.optimization.export_onnx
```

provided that the PyTorch checkpoint is already available.

---

## YOLO Detector

Required for localization:

```text
runs/detect/steel_defect_yolo/weights/best.pt
```

This checkpoint is generated after YOLO training.

Because `runs/` is ignored by Git, the checkpoint must be restored separately when cloning the repository onto another machine.

---

# Running SteelVision

The application requires two terminals.

---

## Terminal 1 — Start FastAPI

From the project root with the virtual environment activated:

```bash
uvicorn src.api.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Terminal 2 — Start Streamlit

From the project root with the same virtual environment activated:

```bash
streamlit run src/ui/app.py
```

Streamlit will display a local browser address after startup.

---

# Main Results Summary

| Experiment | Main Result |
|---|---|
| Fine-tuned ResNet-50 | 99.22% test accuracy |
| 128 px degradation | 83.92% accuracy |
| 64 px degradation | 49.80% accuracy |
| 32 px degradation | 31.76% accuracy |
| Real-ESRGAN at 128 px | 63.53% |
| Real-ESRGAN at 64 px | 41.18% |
| Real-ESRGAN at 32 px | 31.37% |
| YOLO mAP@50 | 0.801 |
| YOLO mAP@50-95 | 0.505 |
| Expected Calibration Error | 0.0482 |
| PyTorch CPU inference | 19.883 ms/image |
| ONNX CPU inference | 12.590 ms/image |
| ONNX speedup | 1.58× |

---

# Key Research Findings

## 1. Transfer learning was highly effective

The fine-tuned ResNet-50 achieved:

```text
99.22% test accuracy
```

which substantially outperformed the need for a purely custom baseline model.

---

## 2. Reduced resolution strongly affected classification

Classification accuracy decreased from:

```text
99.61%
```

at full resolution to:

```text
31.76%
```

at 32 × 32 resolution.

This confirms that important discriminative information is lost as image resolution decreases.

---

## 3. Neural super-resolution did not improve classification

Real-ESRGAN produced lower downstream classification accuracy than bicubic reconstruction at every tested resolution.

The experiment therefore does not support the assumption that visually enhanced super-resolution images necessarily improve defect recognition.

---

## 4. Localization remained possible under controlled resolution changes

YOLO localization could still detect defect regions after resolution adjustment, but:

```text
confidence changed
bounding boxes changed
detections could change
```

These changes demonstrate that localization behavior is sensitive to input resolution.

---

## 5. Higher confidence does not imply better image quality

Some degraded images produced higher detector confidence than their original counterparts.

Therefore:

```text
higher model confidence
≠
better image quality
```

---

## 6. Explainability improves interpretation

Grad-CAM and SHAP provide insight into which image regions and features influence classification.

These methods improve model interpretability but do not provide causal physical explanations.

---

## 7. Uncertainty helped distinguish difficult predictions

Incorrect classifier predictions showed:

```text
lower confidence
higher entropy
```

than correct predictions on average.

This supports using uncertainty indicators alongside class predictions.

---

## 8. ONNX improved deployment efficiency

FP32 ONNX inference preserved classification performance and reduced CPU inference latency by approximately:

```text
1.58×
```

---

## 9. INT8 quantization was not suitable for the final deployment

Although quantization reduced model size, the tested INT8 configurations produced substantial accuracy degradation.

The final deployment therefore retains:

```text
FP32 ONNX
```

---

# Limitations

SteelVision is a research prototype and has several limitations.

- The models were trained on a limited steel defect dataset.
- Performance on unseen industrial environments may differ.
- The classifier predicts one primary class per image.
- YOLO localization depends on the quality of the available bounding-box annotations.
- The detector checkpoint is not stored in the repository.
- The classification and ONNX checkpoints are not stored in the repository.
- Coordinate mapping currently assumes resize-based geometric transformation.
- Complex cropping, perspective changes, and rotation are not automatically handled.
- Grad-CAM indicates model influence rather than physical causality.
- SHAP explanations remain model-derived.
- Confidence does not guarantee correctness.
- Predictive entropy does not guarantee reliability.
- Resolution simulation does not estimate real-world camera distance.
- Upscaling image dimensions does not restore guaranteed lost detail.
- Real-ESRGAN can generate visually plausible detail that is not guaranteed to represent ground truth.
- Neural super-resolution was not shown to improve downstream classification in the current experiments.
- The application is currently designed for local execution.
- Authentication and production security controls are not implemented.

---

# Responsible Interpretation

SteelVision should not be interpreted as an autonomous industrial safety system.

The system can:

```text
evaluate steel surface imagery
classify visible defect patterns
estimate uncertainty
highlight model-influential regions
localize visible defect regions
map detections to uploaded-image coordinates
study resolution effects
```

It does not:

```text
recover guaranteed detail lost because of distance
estimate physical defect depth or severity
replace engineering inspection
prove causal defect mechanisms
guarantee prediction correctness
```

The correct interpretation is:

> SteelVision can evaluate low-resolution imagery, optionally study controlled resolution changes, detect visible defect regions, and map localization coordinates back to the original image space.

---

# Technology Stack

Core deep learning:

```text
Python
PyTorch
torchvision
ResNet-50
Ultralytics YOLO
```

Explainability:

```text
Grad-CAM
SHAP
```

Optimization:

```text
ONNX
ONNX Runtime
```

Backend:

```text
FastAPI
Uvicorn
```

Frontend:

```text
Streamlit
```

Data and analysis:

```text
NumPy
Pandas
Matplotlib
scikit-learn
OpenCV
Pillow
Jupyter
```

---

# Development Environment

The project was developed on:

```text
MacBook Air
Apple M4
```

with:

```text
Python 3.13.5
PyTorch 2.14.0
torchvision 0.29.0
```

Apple Metal Performance Shaders were used where supported during model training and inference experiments.

---

# Future Work

Potential extensions include:

- evaluation on larger industrial datasets
- domain-specific super-resolution training
- SwinIR or other transformer-based restoration models
- segmentation-based defect localization
- larger and more diverse YOLO training data
- confidence calibration improvements
- automated model-weight distribution
- ONNX optimization for additional components
- cloud deployment
- production authentication and monitoring
- robustness evaluation under blur, noise, compression, lighting variation, and perspective changes
- physically validated severity estimation using expert-labelled engineering data

---

# Conclusion

SteelVision demonstrates an end-to-end deep learning workflow for steel surface defect analysis.

The project progressed from a simple CNN baseline to a complete research and application pipeline containing:

```text
transfer learning
fine-tuning
resolution degradation analysis
super-resolution evaluation
defect localization
coordinate mapping
explainable AI
uncertainty analysis
ONNX optimization
FastAPI deployment
Streamlit integration
```

The strongest classification model achieved:

```text
99.22% test accuracy
```

However, classification performance degraded substantially as image resolution decreased.

The super-resolution experiments provided an important negative result:

> Real-ESRGAN produced visually enhanced images but did not improve downstream classification compared with bicubic reconstruction.

This demonstrates that perceptual image quality and model utility are not necessarily equivalent.

The final SteelVision application integrates classification, uncertainty, explainability, localization, coordinate mapping, and controlled resolution experiments into one interface while preserving the scientific limitations identified during the research process.

---

## Repository

```text
https://github.com/vidhigadge/steelvision-dl
```