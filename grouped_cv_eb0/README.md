# Grouped CV EfficientNetB0 Pipeline

## ALS (1) vs Control (0) Classification from Post-Mortem Brain Tissue IHC Images

A clean, modular ML pipeline implementing 5-fold patient-grouped cross-validation with EfficientNetB0 for binary classification of ALS vs Control brain tissue images.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Leakage Prevention](#data-leakage-prevention)
3. [Pipeline Overview](#pipeline-overview)
4. [Patient-Grouped Cross-Validation](#patient-grouped-cross-validation)
5. [Patient-Balanced Sampling](#patient-balanced-sampling)
6. [Threshold Tuning](#threshold-tuning)
7. [Clinical Metrics](#clinical-metrics)
8. [CAM Interpretability](#cam-interpretability)
9. [Running the Pipeline](#running-the-pipeline)
10. [Output Files](#output-files)

---

## Quick Start

```bash
# Navigate to pipeline directory
cd grouped_cv_eb0

# Run the console menu
python main.py

# Or run individual steps:
# 1. Validate data and create splits
# 2. Run Optuna tuning (optional)
# 3. Train models
# 4. Evaluate
# 5. Generate CAMs
```

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (recommended for GPU acceleration)

### Required packages

```bash
pip install torch torchvision albumentations optuna scikit-learn scipy pandas matplotlib opencv-python tqdm
```

### Data Setup

Create a `metadata.csv` file with columns:
- `image_path`: Path to image file
- `patient_id`: Unique patient identifier
- `label` or `y_true`: Binary label (0=Control, 1=ALS)

Update paths in `config.py`:
```python
DATA_DIR = Path("path/to/your/data")
IMAGE_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"
```

---

## Data Leakage Prevention

**CRITICAL**: This pipeline implements strict safeguards against 5 types of data leakage commonly seen in medical imaging ML.

### Leakage Risk #1 — Patient Overlap Between Train/Val

**Problem**: Same patient's images appearing in both training and validation sets leads to overly optimistic performance estimates.

**Fix**: Patient-grouped cross-validation using `StratifiedGroupKFold`:
- Splits are created BEFORE dataset/dataloader creation
- Each fold's dataset only contains its designated patients
- Verified programmatically: no patient appears in both train and val

```python
# From data/splits.py
overlap = set(train_patients) & set(val_patients)
if overlap:
    raise RuntimeError(f"LEAKAGE DETECTED: Fold {fold_id} has patient overlap!")
```

### Leakage Risk #2 — Hyperparameter Tuning on Final Folds

**Problem**: Tuning hyperparameters using the same folds reported as "final results" biases the performance estimate upward.

**Fix**: Nested cross-validation:
- **Outer CV (folds 3-4)**: Reserved for final reporting, never seen during tuning
- **Inner CV (folds 0-2)**: Used by Optuna for hyperparameter search

```python
# From tuning/optuna_tuner.py
self.tuning_folds = list(range(n_inner_folds))  # [0, 1, 2]
self.held_out_folds = list(range(n_inner_folds, len(splits)))  # [3, 4]
```

### Leakage Risk #3 — Threshold Tuning Peeks Beyond Validation

**Problem**: Setting a global threshold using all validation data allows information to leak between folds.

**Fix**: Per-fold threshold tuning:
- Each fold's optimal threshold is determined using ONLY that fold's validation patients
- The tuned threshold is then used to compute that fold's metrics
- No cross-fold information is used

```python
# From evaluation/threshold.py
for fold_id, patient_df in fold_patient_dfs.items():
    result = find_optimal_threshold(patient_df, min_sensitivity)  # Only uses this fold
    fold_thresholds[fold_id] = result
```

### Leakage Risk #4 — Stain Normalization Learns from Full Dataset

**Problem**: Fitting stain normalization on the entire dataset (including validation) leaks information.

**Fix**: Train-only normalization fitting:
- Stain normalizer (if used) is fit ONLY on training fold images
- The fitted normalizer is then applied to validation images

```python
# From data/transforms.py
def get_stain_normalizer(train_images, method='macenko'):
    """Fit stain normalizer on training images only."""
    normalizer.fit(reference_image)  # From training data only
```

### Leakage Risk #5 — Human-in-the-Loop CAM Tuning

**Problem**: Modifying model/pipeline based on validation CAM appearance introduces subjective bias.

**Fix**: CAMs are for reporting/diagnostics ONLY:
- CAM method and settings are decided once, before seeing any outputs
- CAMs are generated AFTER training is complete
- Model is never modified based on CAM appearance

---

## Pipeline Overview

### Two-Phase Training Schedule

Optimized for small patient N (typical in medical imaging):

**Phase A: Head-Only Training (3-5 epochs)**
- Backbone (EfficientNetB0 features) is frozen
- Only classifier head is trained
- Faster convergence, prevents catastrophic forgetting

**Phase B: Fine-Tuning (15-20 epochs)**
- Last backbone stage is unfrozen
- Differential learning rates:
  - Head: `head_lr` (e.g., 1e-3)
  - Backbone: `head_lr × backbone_lr_multiplier` (e.g., 1e-4)
- Early stopping on validation loss

### Model Architecture

```
EfficientNetB0 (ImageNet pretrained)
├── features [0-8]: Backbone (4.0M params)
│   └── [8]: Last stage (unfrozen in Phase B)
└── classifier: Custom head
    ├── Dropout(p=0.3)
    └── Linear(1280 → 1)  # Single logit for BCEWithLogitsLoss
```

---

## Patient-Grouped Cross-Validation

### Why Patient-Grouped CV?

**Goal**: Estimate how the model will perform on **new patients**, not just new images.

**Problem with random splits**: Images from the same patient are correlated (same tissue, same staining). If one patient's images appear in both train and val, the model can "memorize" patient-specific features rather than learning generalizable disease patterns.

**Solution**: `StratifiedGroupKFold` with `groups=patient_id`:
- Each patient appears in exactly one fold as validation
- Stratification ensures class balance (ALS/Control) in each fold
- 5 folds → each patient used for validation exactly once

```
Fold 0: Train [P2,P3,P4,P5...] | Val [P1, P6, P11]
Fold 1: Train [P1,P3,P4,P5...] | Val [P2, P7, P12]
Fold 2: Train [P1,P2,P4,P5...] | Val [P3, P8, P13]
...
```

### Saved Split Table

`outputs/splits/grouped_cv_folds.csv`:
| patient_id | fold_id | split | y_true |
|------------|---------|-------|--------|
| P001       | 0       | val   | 1      |
| P002       | 0       | train | 0      |
| ...        | ...     | ...   | ...    |

---

## Patient-Balanced Sampling

### Why Patient-Level (Not Image-Level) Balancing?

**Problem**: Different patients have different numbers of images. Image-level balancing would:
- Oversample patients with fewer images
- Give disproportionate weight to patients with many images

**Solution**: Patient-level balanced sampler

### How It Works

Weight for each image sample:
```
weight = (1 / n_images_in_patient) × (1 / n_patients_in_class)
```

This ensures:
1. **Equal patient contribution**: Each patient has the same expected contribution per epoch (regardless of image count)
2. **Equal class contribution**: ALS and Control classes are balanced

### Why NOT Use pos_weight?

**pos_weight** in BCEWithLogitsLoss scales the loss for positive samples. However:
- It doesn't address patient-level imbalance
- It doesn't ensure each patient contributes equally
- It can lead to gradient scale issues

**Our approach**: Use `WeightedRandomSampler` with patient-level weights instead. This:
- Handles both class and patient imbalance
- Works at the data loading level (more stable)
- Is clinically meaningful: each patient (not each image) matters equally

---

## Threshold Tuning

### Clinical Constraint Approach

Standard threshold = 0.5 may not be optimal for clinical applications.

**Our constraint**:
1. **Sensitivity ≥ 0.70** (prioritize detecting ALS cases)
2. Among thresholds meeting this, **maximize Specificity**
3. Tie-breaker: **maximize Balanced Accuracy**

### Why Sensitivity Constraint?

In ALS screening:
- **False Negative** (missed ALS) = Delayed diagnosis, worse outcome
- **False Positive** (healthy flagged as ALS) = Additional testing, but less harmful

Setting `min_sensitivity=0.70` ensures we catch at least 70% of ALS cases.

### Per-Fold Tuning (Leakage Prevention)

Each fold's threshold is tuned using ONLY that fold's validation patients:

```python
# For each fold independently:
threshold = find_optimal_threshold(fold_val_patients, min_sensitivity=0.70)
metrics = compute_metrics(fold_val_patients, threshold)
```

---

## Clinical Metrics

### Patient-Level (Primary) Metrics

All metrics are computed at **patient level** (not image level) because clinical decisions are made per patient.

#### Sensitivity (True Positive Rate)
```
Sensitivity = TP / (TP + FN) = Correctly identified ALS / All ALS patients
```
**Clinical meaning**: Of all ALS patients, what fraction did we correctly identify?

#### Specificity (True Negative Rate)
```
Specificity = TN / (TN + FP) = Correctly identified Controls / All Control patients
```
**Clinical meaning**: Of all Control patients, what fraction did we correctly identify?

#### Balanced Accuracy
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
```
**Clinical meaning**: Average of class-specific accuracies. Robust to class imbalance.

#### ROC AUC
Area Under the Receiver Operating Characteristic curve.
**Clinical meaning**: Probability that a randomly chosen ALS patient ranks higher than a randomly chosen Control patient.

### 95% Wilson Confidence Intervals

For small sample sizes (n=15 patients), we use **Wilson score intervals** rather than normal approximation:

```python
# Wilson interval formula
center = (p + z²/(2n)) / (1 + z²/n)
margin = z × sqrt((p(1-p) + z²/(4n)) / n) / (1 + z²/n)
CI = [center - margin, center + margin]
```

**Why Wilson?** Normal approximation fails when n is small or p is near 0 or 1.

### Calibration Metrics

#### Brier Score
```
Brier = mean((predicted_prob - true_label)²)
```
- Range: [0, 1], lower is better
- 0.25 = random guessing
- Measures both discrimination and calibration

#### Expected Calibration Error (ECE)
```
ECE = Σ |fraction_positive_in_bin - mean_predicted_prob_in_bin| × prop_in_bin
```
**Clinical meaning**: How well do predicted probabilities match observed frequencies?

### Reliability Checks

#### Borderline Rate
```
Borderline = |predicted_prob - threshold| ≤ 0.1
```
**Clinical meaning**: Fraction of patients near the decision boundary. These cases warrant additional clinical review.

#### Prediction Instability (Multi-Seed)
If training with multiple seeds:
- `std_prob`: Standard deviation of predicted probability across seeds
- `flip_rate`: Fraction of seeds where prediction differs from majority

**High instability**: Model is uncertain about this patient → clinical review recommended.

---

## CAM Interpretability

### Methods Implemented

#### 1. LayerCAM
Fine-grained localization using element-wise multiplication of positive gradients with activations.

```
CAM = ReLU(Σ ReLU(gradients) ⊙ activations)
```

**Advantage**: More precise localization than Grad-CAM.

#### 2. Guided Grad-CAM
Fusion of Guided Backpropagation and Grad-CAM:

```
Guided Grad-CAM = Guided Backprop × Grad-CAM
```

**Advantage**: High-resolution, class-discriminative visualizations.

### CAM Outputs (Per Method)

| Output | Description |
|--------|-------------|
| `per_image/` | Individual image overlays |
| `per_patient/` | Patient-averaged CAM maps |
| `cam_patient_consistency.csv` | Within-patient IoU metrics |
| `cam_outlier_patients.csv` | Flagged outlier patients |
| `cam_class_comparison.png` | ALS mean vs Control mean |
| `cam_difference_map.png` | ALS - Control difference |

### CAM Consistency Metrics

#### Within-Patient Consistency
IoU (Intersection over Union) of top-10% activated pixels between a patient's images.

**High IoU**: Model focuses on consistent regions across patient's images → reliable.
**Low IoU**: Model focus varies → patient may have heterogeneous pathology.

#### Patient-to-Class Similarity
Pearson correlation between patient's average CAM and their class mean CAM.

**High correlation**: Patient's pattern matches their class → typical case.
**Low correlation**: Patient is atypical → potential labeling issue or rare variant.

#### Between-Class Separability
Correlation between ALS mean CAM and Control mean CAM.

**Low correlation**: Classes attend to different regions → good separability.
**High correlation**: Classes look similar → model may struggle to distinguish.

### Interpreting Outlier Patients

Patients flagged as outliers may indicate:
1. **Labeling errors**: Wrong diagnosis in ground truth
2. **Atypical presentation**: Unusual disease pattern
3. **Technical issues**: Poor staining, artifacts
4. **Interesting cases**: Warrant clinical review

---

## Running the Pipeline

### Console Menu

```bash
python main.py
```

Menu options:
1. **Validate data and create CV splits** - Load metadata, validate, create patient-grouped folds
2. **Run Optuna hyperparameter tuning** - Nested CV optimization
3. **Run full cross-validation training** - Train all 5 folds
4. **Evaluate trained models** - Compute metrics, thresholds, calibration
5. **Generate CAM visualizations** - LayerCAM and Guided Grad-CAM
6. **Run complete pipeline** - Execute 1→3→4→5

### Command Line Usage

```python
# Import modules directly
from grouped_cv_eb0.data import load_and_validate_metadata, create_patient_grouped_splits
from grouped_cv_eb0.models import create_efficientnet_b0, Trainer
from grouped_cv_eb0.evaluation import compute_fold_metrics, tune_threshold_per_fold
from grouped_cv_eb0.interpretability import generate_layercam_outputs
```

### Configuration

Edit `config.py` to customize:
- Data paths
- Training parameters
- Augmentation settings
- Optuna search space
- Threshold constraints

---

## Output Files

```
outputs/
├── splits/
│   └── grouped_cv_folds.csv        # Patient → fold mapping
│
├── checkpoints/
│   ├── fold_0_best.pt              # Best model per fold
│   ├── fold_1_best.pt
│   └── ...
│
├── thresholds/
│   └── fold_thresholds.csv         # Per-fold optimal thresholds
│
├── eval/
│   ├── metrics_summary.csv         # Cross-fold summary (mean±std)
│   ├── metrics_per_fold.csv        # Per-fold detailed metrics
│   ├── borderline_patients.csv     # Patients near threshold
│   ├── patient_instability.csv     # Multi-seed instability
│   ├── calibration_brier.csv       # Brier scores per fold
│   └── reliability_diagram.png     # Calibration curve
│
├── optuna/
│   ├── best_params.json            # Optimal hyperparameters
│   ├── optuna_trials.csv           # All trial results
│   └── optuna_study.pkl            # Pickled study object
│
└── cams/
    ├── layercam/
    │   ├── per_image/              # Individual overlays
    │   ├── per_patient/            # Patient averages
    │   ├── cam_patient_consistency.csv
    │   ├── cam_outlier_patients.csv
    │   ├── cam_class_comparison.png
    │   └── cam_difference_map.png
    │
    └── guided_gradcam/
        ├── per_image/
        ├── per_patient/
        ├── cam_patient_consistency.csv
        ├── cam_outlier_patients.csv
        ├── cam_class_comparison.png
        └── cam_difference_map.png
```

---

## Augmentation Policies

### Morphology-Preserving Augmentation

Three levels selectable via config:

| Level | Transforms |
|-------|------------|
| **Light** | Horizontal/vertical flip, rotation (±10-30°) |
| **Medium** | Light + shift/scale (10-15%), random crop+pad |
| **Strong** | Medium + rare blur/noise, optional elastic |

**Design principles**:
- Histopathology-appropriate magnitudes
- Elastic deformation is rare and mild (preserve cellular structure)
- No color jittering (affects stain interpretation)

### Stain-Aware Augmentation (Optional)

If `USE_STAIN_AUG=True` and RandStainNA is installed:
- Stain variation augmentation
- Reference statistics fit on **training fold only** (leakage prevention)

---

## Module Structure

```
grouped_cv_eb0/
├── main.py                 # Console menu interface
├── config.py               # Central configuration
├── README.md               # This file
│
├── data/
│   ├── dataset.py          # ALSDataset class
│   ├── dataloader.py       # Patient-balanced sampling
│   ├── splits.py           # Patient-grouped CV
│   └── transforms.py       # Augmentation policies
│
├── models/
│   ├── efficientnet.py     # EfficientNetB0 + freeze/unfreeze
│   └── trainer.py          # Two-phase training
│
├── evaluation/
│   ├── metrics.py          # Clinical metrics + Wilson CI
│   ├── threshold.py        # Constrained threshold tuning
│   ├── calibration.py      # Brier, ECE, reliability diagram
│   └── reliability.py      # Borderline, instability analysis
│
├── interpretability/
│   ├── layercam.py         # LayerCAM implementation
│   ├── guided_gradcam.py   # Guided Grad-CAM
│   └── cam_analysis.py     # Consistency, outliers
│
├── tuning/
│   └── optuna_tuner.py     # Nested CV hyperparameter search
│
└── utils/
    └── __init__.py         # Seed setting, device info
```

---

## Summary: 5 Leakage Risks + Fixes

| Risk | Description | Fix |
|------|-------------|-----|
| **#1** | Patient overlap train/val | Patient-grouped splits, verified no overlap |
| **#2** | Tuning leakage | Nested CV: inner folds for tuning, outer for reporting |
| **#3** | Threshold leakage | Per-fold threshold tuning using only that fold |
| **#4** | Normalization leakage | Fit stain normalizer on training fold only |
| **#5** | CAM tuning | CAMs for diagnostics only, no model changes |

---

## Citation

If using this pipeline, please cite the relevant methods:
- EfficientNet: Tan & Le, 2019
- Grad-CAM: Selvaraju et al., 2017
- LayerCAM: Jiang et al., 2021
- Wilson CI: Wilson, 1927

---

*Generated for Project Neurobiomark - ALS Classification Pipeline*
