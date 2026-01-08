# Dataset Cartography for Medical Imaging Analysis

## 🧠 Overview

This project implements **Dataset Cartography** to analyze the difficulty of learning each sample in a medical imaging dataset. By training a model multiple times and tracking how confidently each sample is predicted, we can identify which images are:

- **Easy**: Consistently predicted correctly with high confidence
- **Hard**: Consistently mispredicted - may indicate labeling errors or outliers  
- **Ambiguous**: Inconsistent predictions - borderline cases that vary across training runs
- **Medium**: Moderately difficult samples

### Key Features

- **Windowed Epoch Strategy**: Computes metrics using epochs centered on best validation loss
- **Patient-Level Analysis**: Aggregates image-level signals to patient-level insights
- **Multi-Seed Training**: 5 independent runs for robust metric estimation
- **Outlier Detection**: Identifies isolated/problematic patients
- **Comprehensive Reports**: HTML report + Excel outputs + analysis visualizations

## 📁 Project Structure

```
dataset_cartography/
├── config.py              # Configuration settings (EDIT THIS FIRST!)
├── data_loader.py         # Data loading and patient-level splitting
├── model.py               # EfficientNetB0 model definition
├── trainer.py             # Training loop with metric tracking
├── cartography.py         # Legacy cartography metrics computation
├── windowed_cartography.py # Windowed epoch cartography (NEW)
├── patient_analysis.py    # Patient-level analysis module (NEW)
├── visualize.py           # Visualization generation
├── report.py              # HTML report generation
├── run_cartography.py     # Main execution script
├── run_lopo_evaluation.py # LOPO evaluation script (NEW)
├── requirements.txt       # Python dependencies
├── __init__.py            # Package initialization
└── lopo/                  # LOPO evaluation module (NEW)
    ├── __init__.py
    ├── lopo_runner.py     # Main LOPO orchestration
    ├── lopo_metrics.py    # Clinical metrics computation
    └── lopo_visualize.py  # LOPO-specific visualizations

outputs/                   # Created when you run the analysis
├── cartography_metrics.csv           # Legacy per-image metrics
├── cartography_per_image.xlsx        # Windowed per-image metrics (NEW)
├── cartography_per_patient.xlsx      # Patient-level metrics (NEW)
├── best_models_summary.txt           # Best model info per run
├── logs/
│   └── training_logs.csv             # Comprehensive training metrics
├── models/
│   └── model_run1.pth, etc.          # Saved model weights
├── visualizations/
│   ├── cartography_scatter_plot.png
│   └── category_distribution.png
├── analysis/                         # Patient analysis outputs (NEW)
│   ├── analysis1_confidence_vs_correctness.png
│   ├── analysis3_variance_across_seeds.png
│   ├── analysis4_image_disagreement.png
│   ├── isolated_patients.csv
│   ├── patient_variance_seeds.csv
│   ├── image_disagreement.csv
│   ├── patient_instability_summary.csv
│   └── README.md                     # Analysis findings summary
├── lopo/                             # LOPO outputs (NEW)
│   ├── lopo_per_image.xlsx           # Per-image windowed metrics
│   ├── lopo_per_patient_seed.xlsx    # Per-patient-seed metrics
│   ├── lopo_per_patient_final.xlsx   # Final patient metrics
│   ├── lopo_clinical_metrics.csv     # Clinical evaluation metrics
│   ├── lopo_training_logs.csv        # Training logs for all folds
│   └── analysis/                     # LOPO visualizations
│       ├── lopo_patient_confidence_correctness.png
│       ├── lopo_seed_stability.png
│       ├── lopo_image_instability.png
│       ├── lopo_metrics_across_folds.png
│       ├── lopo_patient_predictions.png
│       └── LOPO_README.md
└── CARTOGRAPHY_REPORT.html           # Main HTML report
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd dataset_cartography
pip install -r requirements.txt
```

### Step 2: Configure Your Paths

Open `dataset_cartography/config.py` and update these two lines:

```python
# UPDATE THESE PATHS TO YOUR ACTUAL DATA LOCATIONS
IMAGE_FOLDER = r"C:\path\to\your\images"      # Folder with 1.tiff, 2.tiff, etc.
EXCEL_FILE = r"C:\path\to\your\metadata.xlsx"  # Excel file with metadata
```

### Step 3: Run the Analysis

```bash
cd dataset_cartography
python run_cartography.py
```

The analysis will:
1. Load your images and metadata
2. Split data at **patient level** (prevents data leakage)
3. Train EfficientNetB0 5 times with different random seeds
4. Track predictions for each image across all epochs
5. Compute **windowed cartography metrics** centered on best validation loss
6. Aggregate metrics: Epoch → Image → Patient → Seed
7. Categorize samples as Easy/Medium/Ambiguous/Hard
8. Run patient-level analysis to identify outliers
9. Generate visualizations and comprehensive reports

### Step 4: View Results

- **Main Report**: Open `outputs/CARTOGRAPHY_REPORT.html` in a web browser
- **Patient Analysis**: Check `outputs/analysis/README.md` for findings
- **Excel Data**: Open `outputs/cartography_per_patient.xlsx` for patient metrics

## 📊 Understanding the Metrics

### Windowed Cartography (New Approach)

Metrics are computed using a **windowed epoch strategy**:

1. **Find t\***: Epoch with best validation loss
2. **Window**: Use epochs [t\*-2, t\*+2] (5 epochs total)
3. **Aggregation Order**: Epoch → Image → Patient → Seed

| Metric | Level | Description |
|--------|-------|-------------|
| **Confidence** | Image | Probability assigned to true label, averaged over window epochs |
| **Correctness** | Image | Fraction of correct predictions in window epochs |
| **Patient Confidence** | Patient | Mean of image confidences within patient |
| **Patient Correctness** | Patient | Mean of image correctness within patient |

### Legacy Metrics

| Metric | Description |
|--------|-------------|
| **Confidence** | Average probability for true label across last 5 epochs |
| **Variability** | Std deviation of confidence (lower = more consistent) |
| **Correctness** | Fraction of correct predictions (0-1) |

## 🏷️ Category Definitions

| Category | Criteria | Interpretation |
|----------|----------|----------------|
| **Easy** | Correctness ≥ 0.8, Variability < 0.1 | Model learns these reliably |
| **Hard** | Correctness < 0.5, Variability ≥ 0.2 | May be mislabeled or outliers |
| **Ambiguous** | Correctness ≥ 0.5, Variability ≥ 0.2 | High uncertainty cases |
| **Medium** | Variability < 0.2, Correctness 0.5-0.8 | Moderate difficulty |

## 🔍 Patient-Level Analysis

The patient analysis module (`patient_analysis.py`) provides:

### Analysis 1: Confidence vs Correctness Plot
- Scatter plot with one point per patient
- Point size proportional to number of images
- Reference lines at x=0.5, y=0.5, and y=x
- Outliers highlighted in red

### Analysis 2: Isolated Patient Detection
Identifies problematic patients using:
- IQR-based outlier detection
- Z-score > 2 threshold
- Low correctness (< 0.5)
- Overconfident errors (high confidence, low correctness)

### Analysis 3: Variance Across Seeds
- Bar chart of patient correctness with uncertainty
- Shows stability of predictions across training runs

### Analysis 4: Image-Level Disagreement
- Boxplot of instability per patient
- Identifies patients with high internal variance

**Run standalone** (if cartography files exist):
```bash
python -m dataset_cartography.patient_analysis
```

## 🏥 LOPO (Leave-One-Patient-Out) Evaluation

For **clinical validation**, we provide a LOPO cross-validation module that:

1. Creates **15 folds** (one patient held out per fold)
2. Trains **5 seeds per fold** for ensemble stability
3. Computes **windowed metrics** around best validation epoch
4. Aggregates predictions across seeds for final patient prediction

### Running LOPO Evaluation

```bash
cd dataset_cartography
python run_lopo_evaluation.py
```

⚠️ **Note**: LOPO is computationally intensive (15 folds × 5 seeds = 75 training runs). Expect several hours of runtime with GPU.

### LOPO Clinical Metrics

| Metric | Description |
|--------|-------------|
| **Sensitivity** | True Positive Rate (correctly identifying disease) |
| **Specificity** | True Negative Rate (correctly identifying healthy) |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 |
| **AUC** | Area Under ROC Curve |
| **F1 Score** | Harmonic mean of precision and recall |

### LOPO Outputs

| File | Description |
|------|-------------|
| `lopo_per_image.xlsx` | Per-image windowed metrics for each fold/seed |
| `lopo_per_patient_seed.xlsx` | Per-patient metrics per seed |
| `lopo_per_patient_final.xlsx` | Final patient predictions (aggregated across seeds) |
| `lopo_clinical_metrics.csv` | Clinical evaluation metrics |
| `analysis/LOPO_README.md` | Detailed results summary |

### LOPO vs Standard Cartography

| Aspect | Standard Cartography | LOPO Evaluation |
|--------|---------------------|-----------------|
| **Purpose** | Understand sample difficulty | Clinical performance validation |
| **Split** | 80/20 patient-level | Leave-one-patient-out |
| **Folds** | 1 (with multiple seeds) | 15 (one per patient) |
| **Metrics** | Confidence, Correctness | Sensitivity, Specificity, AUC |
| **Use Case** | Data exploration | Publication-ready results |

## 📋 Data Requirements

Your **Excel file** should have these columns:
- **Image No**: Number matching the .tiff filename (1, 2, 3, ...)
- **Case ID**: Full identifier (e.g., "SD028-12 Concord BA46")
- **Condition**: Classification label ("Case" or "Control")

Your **images** should be:
- Named as numbers: `1.tiff`, `2.tiff`, ..., `190.tiff`
- Located in a single folder

## ⚙️ Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Model | EfficientNetB0 (ImageNet pretrained) | Transfer learning |
| Training Runs | 5 (seeds: 1, 2, 3, 4, 5) | Robust estimates |
| Epochs | 25 (with early stopping) | Prevent overfitting |
| Early Stopping | patience=3 | Stop if no improvement |
| Batch Size | 8 | Regularization effect |
| Learning Rate | 5e-5 | Stability |
| Optimizer | AdamW (weight_decay=1e-4) | L2 regularization |
| Loss | BCEWithLogitsLoss | Binary classification |
| Label Smoothing | 0.05 | Reduces overconfidence |
| LR Scheduler | ReduceLROnPlateau | Adaptive learning |
| Image Size | 224×224 | EfficientNet input |
| Dropout | 0.3 | Regularization |
| Window Size | ±2 epochs around t* | Windowed cartography |

### Patient-Level Data Split

⚠️ **Important**: Data is split at the **patient level** to prevent data leakage.

- 80% of patients for training, 20% for validation
- All images from same patient stay in same split
- Uses fixed seed (42) for reproducible splits across runs

### Anti-Overfitting Strategies

- Slower learning rate (5e-5)
- Smaller batch size (8)
- Label smoothing (0.05)
- LR scheduler with plateau detection
- Early stopping (patience=3)
- Best checkpoint restoration
- Patient-level splitting (prevents memorization)

## 📈 Output Files

### Main Outputs

| File | Description |
|------|-------------|
| `cartography_per_image.xlsx` | Per-image windowed metrics |
| `cartography_per_patient.xlsx` | Per-patient aggregated metrics |
| `cartography_metrics.csv` | Legacy per-sample metrics |
| `CARTOGRAPHY_REPORT.html` | Comprehensive HTML report |

### Analysis Outputs (in `outputs/analysis/`)

| File | Description |
|------|-------------|
| `isolated_patients.csv` | Outlier patients with detection reasons |
| `image_disagreement.csv` | Per-image instability metrics |
| `patient_instability_summary.csv` | Per-patient instability stats |
| `patient_variance_seeds.csv` | Cross-seed variance data |
| `README.md` | Analysis findings summary |

### Training Logs

| Column | Description |
|--------|-------------|
| `run`, `epoch` | Run and epoch identifiers |
| `train_loss`, `val_loss` | Loss values |
| `train_acc`, `val_acc` | Accuracy |
| `train_auc`, `val_auc` | AUC-ROC |
| `train_f1`, `val_f1` | F1-score |
| `train_recall`, `val_recall` | Recall/Sensitivity |
| `train_precision`, `val_precision` | Precision |

## 🔬 How to Use Results

1. **Review Hard Samples**: Manually inspect images categorized as "Hard" - they may have labeling errors

2. **Check Isolated Patients**: Review `outputs/analysis/isolated_patients.csv` for problematic cases

3. **Examine Ambiguous Samples**: These borderline cases may need expert review

4. **Patient-Level Insights**: Use `cartography_per_patient.xlsx` to identify challenging patients

5. **Data Cleaning**: Consider removing or relabeling problematic samples

6. **Curriculum Learning**: Train final models starting with easy samples, then harder ones

## 📚 Reference

This implementation is based on:

> Swayamdipta et al. (2020). "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics." *EMNLP 2020*

## 🛠️ Troubleshooting

**Error: "Image folder not found"**
- Check that `IMAGE_FOLDER` in config.py points to the correct location

**Error: "Excel file not found"**  
- Check that `EXCEL_FILE` in config.py points to your metadata file

**Out of memory error**
- Reduce `BATCH_SIZE` in config.py (try 4 instead of 8)

**Training is very slow**
- Make sure you have a GPU available (CUDA)
- The analysis takes ~2-4 hours on GPU, longer on CPU

**Patient analysis fails**
- Ensure cartography files exist in `outputs/` before running standalone analysis
- Run full pipeline first: `python run_cartography.py`