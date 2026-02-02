# PROJECT-NEUROBIOMARK-1: Complete Knowledge Transfer Document

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Input Data](#2-input-data)
3. [Three Main Pipelines](#3-three-main-pipelines)
4. [Pipeline 1: dataset_cartography (ALS vs Control - Original)](#4-pipeline-1-dataset_cartography)
5. [Pipeline 2: grouped_cv_eb0 (ALS vs Control - Production)](#5-pipeline-2-grouped_cv_eb0)
6. [Pipeline 3: als_vs_alsftd (ALS vs ALS-FTD)](#6-pipeline-3-als_vs_alsftd)
7. [Cross-Validation Strategies](#7-cross-validation-strategies)
8. [Model Architecture](#8-model-architecture)
9. [Training Process](#9-training-process)
10. [Dataset Cartography Method](#10-dataset-cartography-method)
11. [Metrics and Evaluation](#11-metrics-and-evaluation)
12. [Data Augmentation](#12-data-augmentation)
13. [Hyperparameter Tuning (Optuna)](#13-hyperparameter-tuning-optuna)
14. [Output Files](#14-output-files)
15. [How to Run](#15-how-to-run)
16. [Data Leakage Prevention](#16-data-leakage-prevention)
17. [Key Configuration Parameters](#17-key-configuration-parameters)

---

## 1. Project Overview

This project implements **deep learning classification of brain tissue images** for ALS (Amyotrophic Lateral Sclerosis) research. The goal is to:

1. **Classify brain tissue images** as ALS vs Control (healthy)
2. **Classify ALS subtypes**: ALS vs ALS-FTD (ALS with Frontotemporal Dementia)
3. **Identify reliable vs problematic samples** using Dataset Cartography

### Directory Structure

```
PROJECT-NEUROBIOMARK-1/
├── data/                      # Input data (images + metadata)
│   ├── raw_images/           # TIFF images (1.tif, 2.tif, ...)
│   └── image_keys.xlsx       # Metadata mapping
├── dataset_cartography/       # Pipeline 1: Original LOPO + Cartography
├── grouped_cv_eb0/           # Pipeline 2: 5-Fold CV with Optuna tuning
├── als_vs_alsftd/            # Pipeline 3: ALS vs ALS-FTD classification
└── image_keys.xlsx           # Master metadata file
```

---

## 2. Input Data

### Images
- **Location**: `data/raw_images/`
- **Format**: TIFF files named `1.tif`, `2.tif`, `3.tif`, etc.
- **Total**: ~190 images from ~15 patients
- **Resolution**: Variable (resized to 224×224 during preprocessing)

### Metadata (image_keys.xlsx)

| Column | Description | Example |
|--------|-------------|---------|
| `Image No` | Image filename number | 1, 2, 3 |
| `Case ID` | Patient identifier | "SD028-12 Concord BA46" |
| `Condition` | ALS vs Control | "Case" or "Control" |
| `Category` | Subtype info | "Concordant", "Discordant" |
| `Region` | Brain region | "BA46" |

### Label Mapping

**For ALS vs Control (dataset_cartography, grouped_cv_eb0):**
- `Condition == "Case"` → y = 1 (ALS)
- `Condition == "Control"` → y = 0 (Control)

**For ALS vs ALS-FTD (als_vs_alsftd):**
- `Category == "Concordant"` → y = 1 (ALS-FTD)
- `Category == "Discordant"` → y = 0 (ALS)
- Controls are **excluded entirely**

---

## 3. Three Main Pipelines

| Pipeline | Task | CV Strategy | Purpose |
|----------|------|-------------|---------|
| `dataset_cartography/` | ALS vs Control | LOPO | Original research, cartography analysis |
| `grouped_cv_eb0/` | ALS vs Control | 5-Fold Patient-Grouped | Production, Optuna tuning, CAM |
| `als_vs_alsftd/` | ALS vs ALS-FTD | LOPO | Subtype classification |

---

## 4. Pipeline 1: dataset_cartography

### Purpose
Original Dataset Cartography implementation for identifying easy/hard/ambiguous samples.

### Key Files
```
dataset_cartography/
├── config.py                  # Configuration
├── data_loader.py            # Data loading
├── model.py                  # EfficientNetB0 model
├── trainer.py                # Training loop
├── run_cartography.py        # Main cartography runner
├── run_lopo_evaluation.py    # LOPO evaluation
├── windowed_cartography.py   # Windowed metrics computation
├── visualize.py              # Plotting
└── patient_analysis.py       # Patient-level analysis
```

### How to Run
```bash
cd dataset_cartography
python run_cartography.py      # Train + compute cartography
python run_lopo_evaluation.py  # Evaluate LOPO performance
```

---

## 5. Pipeline 2: grouped_cv_eb0

### Purpose
Production-ready pipeline with:
- 5-fold patient-grouped cross-validation
- Optuna hyperparameter tuning
- CAM (Class Activation Maps) interpretability
- Clinical threshold tuning

### Key Files
```
grouped_cv_eb0/
├── config.py                 # All configuration in one place
├── main.py                   # Interactive menu interface
├── data/
│   ├── dataset.py           # PyTorch Dataset class
│   ├── dataloader.py        # DataLoader creation
│   ├── splits.py            # Patient-grouped fold creation
│   └── transforms.py        # Augmentation pipelines
├── models/
│   ├── efficientnet.py      # Model architecture
│   └── trainer.py           # Training loop with Trainer class
├── evaluation/
│   ├── metrics.py           # Sensitivity, Specificity, Wilson CI
│   ├── calibration.py       # Brier score, reliability diagrams
│   ├── reliability.py       # Report generation
│   └── threshold.py         # Clinical threshold tuning
├── tuning/
│   └── optuna_tuner.py      # Hyperparameter optimization
└── interpretability/
    ├── layercam.py          # LayerCAM implementation
    └── guided_gradcam.py    # Guided Grad-CAM
```

### How to Run
```bash
cd grouped_cv_eb0
python main.py
```

Interactive menu:
```
[1] Validate data and create CV splits
[2] Run Optuna hyperparameter tuning
[3] Run full cross-validation training
[4] Evaluate trained models
[5] Generate CAM visualizations
[6] Run complete pipeline (1→3→4→5)
[0] Exit
```

---

## 6. Pipeline 3: als_vs_alsftd

### Purpose
ALS vs ALS-FTD subtype classification using LOPO with windowed cartography.

### Key Files
```
als_vs_alsftd/
├── config.py              # Configuration
├── main.py                # Main entry point
├── utils.py               # Data loading, filtering
├── lopo_splits.py         # LOPO split creation
├── train.py               # Training loop
├── cartography.py         # Windowed cartography
├── reliability_checks.py  # Patient categorization
├── visualization.py       # 5 diagnostic plots
└── README.md              # Pipeline documentation
```

### How to Run
```bash
python -m als_vs_alsftd.main
```

Or with specific stages:
```bash
python -m als_vs_alsftd.main --stage splits      # Only create splits
python -m als_vs_alsftd.main --stage train       # Only train
python -m als_vs_alsftd.main --stage reliability # Reliability checks
python -m als_vs_alsftd.main --stage visualize   # Generate plots
```

---

## 7. Cross-Validation Strategies

### LOPO (Leave-One-Patient-Out)
Used in: `dataset_cartography/`, `als_vs_alsftd/`

```
N patients → N folds
Fold 1: Train on patients [2,3,...,N], Test on patient [1]
Fold 2: Train on patients [1,3,...,N], Test on patient [2]
...
Fold N: Train on patients [1,2,...,N-1], Test on patient [N]
```

**Pros**: Each patient tested exactly once, maximum use of data
**Cons**: N training runs (slow for large N)

### 5-Fold Patient-Grouped CV
Used in: `grouped_cv_eb0/`

```
15 patients → 5 folds (~3 patients per fold)
Fold 1: Train on folds [2,3,4,5], Test on fold [1]
Fold 2: Train on folds [1,3,4,5], Test on fold [2]
...
```

**Pros**: Faster than LOPO, standard for reporting
**Cons**: Each patient tested once, but fewer folds

### Why Patient-Level Splitting?

**CRITICAL**: Never split at image level!

If patient P has 12 images and you randomly split:
- 10 images in train, 2 in test
- Model learns patient-specific features → **DATA LEAKAGE**
- Metrics are artificially inflated

Patient-grouped splitting ensures:
- ALL images from a patient are in same split
- Model never sees test patient during training

---

## 8. Model Architecture

### EfficientNetB0

All pipelines use **EfficientNetB0** pretrained on ImageNet.

```python
Architecture:
├── Backbone: EfficientNetB0 (ImageNet weights)
│   └── features.0 through features.8
├── Global Average Pooling
├── Dropout (p=0.3)
└── Classifier: Linear(1280, 1)  # Single logit for binary
```

**Why EfficientNetB0?**
- Efficient architecture (compound scaling)
- Good balance of accuracy vs compute
- Works well with small datasets when pretrained
- 224×224 input size (standard)

### Model Creation Code
```python
# From grouped_cv_eb0/models/efficientnet.py
import timm

def create_efficientnet_b0(dropout_rate=0.3, pretrained=True):
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=1  # Binary classification
    )
    # Replace classifier with dropout + linear
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(1280, 1)
    )
    return model
```

---

## 9. Training Process

### Two-Phase Training (grouped_cv_eb0)

**Phase A: Head-Only (Freeze Epochs = 3)**
- Backbone frozen, only classifier head trains
- Higher learning rate for head (1e-3)
- Allows head to adapt to new task

**Phase B: Fine-Tuning (Finetune Epochs = 20)**
- Entire network unfrozen
- Lower learning rate for backbone (HEAD_LR × 0.1)
- Fine-tunes pretrained features

### Training Loop Components

```python
# Optimizer
optimizer = AdamW(params, lr=HEAD_LR, weight_decay=1e-4)

# Loss Function
criterion = BCEWithLogitsLoss()  # No pos_weight for balanced data

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize val_loss
    factor=0.5,      # Reduce LR by half
    patience=3       # After 3 epochs without improvement
)

# Early Stopping
if val_loss hasn't improved for PATIENCE epochs:
    stop training, restore best model
```

### Per-Epoch Output
```
Epoch  1/25 | Train Loss: 0.6821 | Val Loss: 0.6234 | LR: 5.00e-05 ✓ (best)
Epoch  2/25 | Train Loss: 0.5124 | Val Loss: 0.5678 | LR: 5.00e-05  
Epoch  3/25 | Train Loss: 0.4532 | Val Loss: 0.5123 | LR: 5.00e-05 ✓ (best)
...
Early stopping at epoch 15
Best epoch: 12, Val loss: 0.3456
```

---

## 10. Dataset Cartography Method

### What is Dataset Cartography?

A technique from [Swayamdipta et al., 2020](https://arxiv.org/abs/2009.10795) that characterizes training samples by their behavior during training.

### Key Metrics (Per Image)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Confidence** | P(correct label) | How certain the model is about the true class |
| **Correctness** | Mean accuracy over epochs | How often the model predicts correctly |
| **Variability** | Std(probability) over epochs | How stable the prediction is |

### Windowed Cartography (Our Innovation)

Standard cartography averages over ALL epochs. Problem:
- Early epochs: Model is random
- Late epochs: Model may overfit

**Our solution**: Window around best validation epoch

```
t* = argmin(val_loss)  # Best epoch
Window = [t* - 2, t* + 2]  # 5 epochs total

Metrics computed only within this window
```

### Patient Categories

Based on cartography metrics, patients are categorized:

| Category | Criteria | Meaning |
|----------|----------|---------|
| **Easy** | correctness ≥ 0.8 AND confidence ≥ 0.8 | Reliably classified |
| **Medium** | 0.6 ≤ correctness < 0.8, stable | Moderately reliable |
| **Ambiguous** | 0.4 ≤ correctness < 0.6 | Near chance level |
| **Hard** | correctness < 0.4 | Consistently wrong |

### Additional Flags

| Flag | Criteria | Action Needed |
|------|----------|---------------|
| **Borderline** | |prob - 0.5| ≤ 0.1 | Review threshold |
| **Unstable (seeds)** | High variance across random seeds | Unreliable |
| **Outlier** | Far from median in confidence-correctness space | Investigate |

---

## 11. Metrics and Evaluation

### Primary Metrics (Patient-Level)

| Metric | Formula | Clinical Meaning |
|--------|---------|------------------|
| **Sensitivity** | TP / (TP + FN) | How many ALS cases are detected |
| **Specificity** | TN / (TN + FP) | How many controls are correctly identified |
| **Balanced Accuracy** | (Sens + Spec) / 2 | Overall performance (handles imbalance) |
| **ROC AUC** | Area under ROC curve | Discrimination ability |

### Wilson Confidence Intervals

For small sample sizes (~15 patients), we use **Wilson score intervals** instead of normal approximation:

```python
def compute_wilson_ci(successes, total, confidence=0.95):
    p = successes / total
    z = 1.96  # for 95% CI
    
    denominator = 1 + z²/n
    center = (p + z²/(2n)) / denominator
    margin = z * sqrt((p(1-p) + z²/(4n)) / n) / denominator
    
    return (center - margin, center + margin)
```

### Report Output Format

```
CROSS-VALIDATION RESULTS
========================================

Per-Fold Metrics:
Fold   Sens     Spec     Bal Acc    AUC
0      0.850    0.780    0.815      0.890
1      0.820    0.800    0.810      0.875
2      0.750    0.850    0.800      0.860
3      0.900    0.700    0.800      0.880
4      0.800    0.820    0.810      0.870
---------------------------------------

Mean Metrics Across Folds:
  Sensitivity: 0.824 ± 0.055
  Specificity: 0.790 ± 0.057
  Balanced Acc: 0.807 ± 0.007

Pooled 95% Wilson Confidence Intervals:
  Sensitivity: 0.824 [0.680, 0.915]
  Specificity: 0.790 [0.640, 0.890]
```

---

## 12. Data Augmentation

### Augmentation Levels

| Level | Transforms | Use Case |
|-------|------------|----------|
| **Light** | Flips, Rotation (±20°) | Small datasets, prevent overfitting |
| **Medium** | Light + ShiftScaleRotate, RandomCrop | Default, balanced |
| **Strong** | Medium + Blur, Noise, Elastic | Large datasets, heavy regularization |

### Augmentation Pipeline (Medium)

```python
import albumentations as A

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.1),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

### Critical Rule

**Training**: Augmentation ON (random transforms each epoch)
**Validation/Test**: Augmentation OFF (only resize + normalize)

---

## 13. Hyperparameter Tuning (Optuna)

### Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| `head_lr` | [1e-4, 1e-2] | Log-uniform |
| `backbone_lr_multiplier` | [0.01, 0.3] | Uniform |
| `weight_decay` | [1e-6, 1e-3] | Log-uniform |
| `dropout` | [0.2, 0.5] | Uniform |
| `label_smoothing` | Fixed 0.05 | - |
| `freeze_epochs` | [3, 5] | Integer |
| `aug_level` | light/medium/strong | Categorical |
| `batch_size` | [8, 16, 32] | Categorical |

### Nested Cross-Validation

To prevent tuning leakage:

```
Outer CV (5 folds): For final reporting
  Inner CV (3 folds): For Optuna tuning
  
Optuna uses folds [0, 1, 2] for tuning
Final evaluation uses folds [3, 4] (never seen during tuning)
```

### Running Optuna

```bash
cd grouped_cv_eb0
python main.py
# Select option [2] for Optuna tuning
# Enter number of trials (e.g., 50)
```

Output: `outputs/optuna/best_params.json`

---

## 14. Output Files

### grouped_cv_eb0 Outputs

```
grouped_cv_eb0/outputs/
├── splits/
│   └── grouped_cv_folds.csv        # Fold assignments
├── checkpoints/
│   └── fold_{i}_best.pt            # Model weights per fold
├── eval/
│   ├── metrics_per_fold.csv        # Per-fold metrics
│   ├── metrics_summary.csv         # Mean ± std
│   └── borderline_patients.csv     # Patients near threshold
├── optuna/
│   ├── best_params.json            # Best hyperparameters
│   └── study.pkl                   # Full Optuna study
└── cams/
    └── fold_{i}/                   # CAM visualizations
```

### als_vs_alsftd Outputs

```
als_vs_alsftd/output/
├── lopo/
│   └── lopo_splits.csv
├── logs/
│   └── fold_{i}_seed_{j}_training_log.csv
├── cartography/
│   ├── cartography_per_image.xlsx
│   └── cartography_per_patient_final.xlsx
├── metrics/
│   ├── lopo_clinical_metrics.csv
│   ├── patient_categorization.csv
│   └── hardest_patients.csv
├── plots/
│   ├── fig1_confidence_vs_correctness.png
│   ├── fig2_seed_variance_correctness.png
│   ├── fig3_seed_variance_probability.png
│   ├── fig4_image_disagreement.png
│   └── fig5_category_summary.png
└── checkpoints/
    └── best_model_fold_{i}_seed_{j}.pt
```

---

## 15. How to Run

### Prerequisites

```bash
# Create conda environment
conda create -n neurobiomark-pr1 python=3.10
conda activate neurobiomark-pr1

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0
torchvision
timm
albumentations
pandas
numpy
scipy
scikit-learn
matplotlib
openpyxl
optuna  # for grouped_cv_eb0
```

### Running Pipelines

```bash
# Pipeline 1: Dataset Cartography
cd dataset_cartography
python run_cartography.py

# Pipeline 2: Grouped CV (interactive)
cd grouped_cv_eb0
python main.py

# Pipeline 3: ALS vs ALS-FTD
python -m als_vs_alsftd.main
```

---

## 16. Data Leakage Prevention

### 5 Explicit Safeguards

| # | Leakage Type | Prevention |
|---|--------------|------------|
| 1 | **Patient overlap** | Splits created BEFORE dataset creation |
| 2 | **Tuning leakage** | Nested CV with held-out folds |
| 3 | **Threshold leakage** | Per-fold tuning on validation only |
| 4 | **Stain normalization** | Fit on training fold only |
| 5 | **CAM leakage** | CAMs for diagnostics only, not tuning |

### Code Pattern

```python
# WRONG: Creates leakage
for epoch in epochs:
    train_transform = get_augmented_transform(train_data)  # Uses ALL data
    
# CORRECT: No leakage
train_transform = get_augmented_transform()  # Fixed pipeline
for epoch in epochs:
    # Transform applied independently to each sample
```

---

## 17. Key Configuration Parameters

### Shared Across Pipelines

```python
# Image
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model
MODEL_NAME = "efficientnet_b0"
DROPOUT_RATE = 0.3

# Training
BATCH_SIZE = 8-16
LEARNING_RATE = 5e-5 to 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3-5

# Cartography
WINDOW_HALF_SIZE = 2  # ±2 epochs around best
```

### Pipeline-Specific

**grouped_cv_eb0:**
```python
N_FOLDS = 5
FREEZE_EPOCHS = 3
FINETUNE_EPOCHS = 20
```

**als_vs_alsftd:**
```python
NUM_SEEDS = 5
RANDOM_SEEDS = [1, 2, 3, 4, 5]
POSITIVE_CATEGORY = "Concordant"  # ALS-FTD
NEGATIVE_CATEGORY = "Discordant"  # ALS
```

---

## Quick Reference Card

### Common Tasks

| Task | Command |
|------|---------|
| Train ALS vs Control (5-fold) | `cd grouped_cv_eb0 && python main.py` → [3] |
| Tune hyperparameters | `cd grouped_cv_eb0 && python main.py` → [2] |
| Generate CAM visualizations | `cd grouped_cv_eb0 && python main.py` → [5] |
| Train ALS vs ALS-FTD | `python -m als_vs_alsftd.main` |
| Run original cartography | `cd dataset_cartography && python run_cartography.py` |

### Key Files to Modify

| Change | File |
|--------|------|
| Data paths | `*/config.py` → `DATA_DIR`, `IMAGE_DIR` |
| Training params | `*/config.py` → `LEARNING_RATE`, `BATCH_SIZE`, etc. |
| Augmentation | `grouped_cv_eb0/data/transforms.py` |
| Model architecture | `grouped_cv_eb0/models/efficientnet.py` |
| Metrics | `grouped_cv_eb0/evaluation/metrics.py` |

---

## Contact & Resources

- **Original Paper**: Swayamdipta et al. "Dataset Cartography" (EMNLP 2020)
- **EfficientNet Paper**: Tan & Le "EfficientNet" (ICML 2019)
- **Wilson CI**: Wilson, E.B. (1927) JASA

