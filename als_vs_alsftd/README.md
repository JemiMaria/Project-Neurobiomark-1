# ALS vs ALS-FTD Classification Pipeline

## 1. Executive Summary

This pipeline implements **Leave-One-Patient-Out (LOPO) cross-validation** with **windowed Dataset Cartography** for binary classification of **ALS vs ALS-FTD** brain tissue images.

**Key Characteristics:**
- **Model**: EfficientNetB0 (ImageNet pretrained)
- **Validation Strategy**: LOPO (each patient held out once)
- **Cartography Method**: Windowed (t* ± 2 epochs around best val_loss)
- **Seeds**: 5 random seeds for stability analysis
- **Confidence Intervals**: 95% Wilson (appropriate for small N ~10)

---

## 2. Label Mapping

| Category | Label | Description |
|----------|-------|-------------|
| Concordant | y = 1 (positive) | ALS-FTD phenotype |
| Discordant | y = 0 (negative) | ALS phenotype |
| Control | Excluded | Not included in analysis |

**Rationale**: This pipeline focuses on distinguishing between ALS and ALS-FTD phenotypes. Controls are excluded as the clinical question is phenotype differentiation within ALS spectrum disorders.

**Expected Distribution**: ~10 patients, ~120 images, approximately balanced classes.

---

## 3. LOPO Rationale

### Why Leave-One-Patient-Out?

With only ~10 patients:
- **K-Fold CV fails**: Random splits cannot guarantee patient separation
- **Patient leakage**: Images from the same patient in train+val causes massive overfitting
- **Clinical validity**: Real-world deployment tests on unseen patients

### Implementation

```
For each patient p in {1, ..., N}:
    Test set  = all images from patient p
    Train set = all images from patients != p
    
    Train model → predict on patient p → store predictions
```

### Interpretation

- LOPO accuracy represents expected performance on a new patient
- Wide confidence intervals expected due to small N
- Each patient contributes one independent test case

---

## 4. Windowed Cartography Method

### Standard Cartography (Swayamdipta et al., 2020)

Average metrics across ALL training epochs. **Problem**: Early epochs (random) and late epochs (memorized) add noise.

### Windowed Cartography (This Pipeline)

1. Find **t*** = epoch with best validation loss
2. Define window: **[t* - 2, t* + 2]** (5 epochs)
3. Average metrics only within this window

**Advantage**: Captures model behavior during peak generalization, filtering out:
- Early random predictions
- Late overfitting/memorization

### Metrics Computed

For each image i in the validation set:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Confidence** | mean(prob if y=1 else 1-prob) | How certain the model is |
| **Correctness** | mean(prob > 0.5 == y) | How often correct |
| **Variability** | std(prob) | Prediction stability |
| **Flip Rate** | mean(sign changes) | Unstable across epochs |

---

## 5. Metrics Definitions

### Image-Level Metrics

- **mean_prob**: Average predicted probability (P(ALS-FTD))
- **mean_confidence**: Average model certainty
- **mean_correctness**: Fraction of epochs with correct prediction
- **std_prob**: Standard deviation of probability across epochs
- **variability**: Same as std_prob (legacy naming)
- **flip_rate**: Fraction of epochs where prediction flipped sign

### Patient-Level Metrics (Aggregated)

- **final_mean_prob**: Mean of image-level mean_prob
- **final_mean_confidence**: Mean of image-level confidence
- **final_mean_correctness**: Mean of image-level correctness
- **final_std_prob**: Std of image-level mean_prob (within-patient variation)
- **final_std_correctness**: Std of image-level correctness across seeds
- **flip_rate**: Max flip rate across images

### Cross-Seed Aggregation

For each patient:
1. Compute metrics per seed
2. Average across seeds for final values
3. Track std across seeds for stability assessment

---

## 6. Patient Categorization

Patients are categorized based on cartography metrics:

| Category | Criteria | Clinical Interpretation |
|----------|----------|------------------------|
| **Easy** | correctness ≥ 0.8 AND confidence ≥ 0.8 | Reliably classified |
| **Medium** | 0.6 ≤ correctness < 0.8 AND std < 0.1 | Moderately confident |
| **Ambiguous** | 0.4 ≤ correctness < 0.6 | Near chance level |
| **Hard** | correctness < 0.4 | Consistently misclassified |

### Additional Flags

| Flag | Criteria | Meaning |
|------|----------|---------|
| **Borderline** | |mean_prob - 0.5| ≤ 0.1 | Near decision threshold |
| **Unstable (seeds)** | flip_rate > 0.2 OR std_correctness > 0.2 | Varies across seeds |
| **Unstable (images)** | std_prob > 0.15 | High within-patient variance |
| **Outlier** | Distance from median > 2σ | Statistical anomaly |

---

## 7. Reliability Findings (Template)

*[This section will be populated after running the pipeline]*

### Clinical Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Sensitivity | - | [-, -] |
| Specificity | - | [-, -] |
| Balanced Accuracy | - | - |
| F1 Score | - | - |

### Probability Margin

- Min ALS-FTD probability: -
- Max ALS probability: -
- **Margin**: - (positive = separable)

### Seed Stability

- Patients unstable across seeds: - / -
- Max seed variance: -

---

## 8. Problem Patients (Template)

*[This section will be populated after running the pipeline]*

### High-Risk Classifications

| Patient | True Label | Category | Flags | Notes |
|---------|------------|----------|-------|-------|
| - | - | - | - | - |

### Recommended Actions

1. **Hard patients**: Review histopathology for annotation errors
2. **Borderline patients**: Consider for follow-up or additional imaging
3. **Unstable patients**: May indicate ambiguous phenotype

---

## 9. Visual Interpretation Guide

### Figure 1: Confidence vs Correctness

**How to read:**
- X-axis: Mean correctness (0 = always wrong, 1 = always right)
- Y-axis: Mean confidence (0 = uncertain, 1 = certain)
- Colors: Blue = ALS, Red = ALS-FTD
- Size: Larger = more stable predictions

**Regions:**
- Top-right (green zone): Easy patients, reliable
- Bottom-left (red zone): Hard patients, problematic
- Top-left: Wrong but confident (label issues?)
- Bottom-right: Correct but uncertain (need more data?)

### Figure 2: Seed Variance (Correctness)

**How to read:**
- Bar height = standard deviation of correctness across seeds
- High bars = unstable predictions across seeds
- Orange line = instability threshold (0.2)

### Figure 3: Seed Variance (Probability)

**How to read:**
- Bar height = standard deviation of probability across seeds
- Complements Figure 2 with probability perspective

### Figure 4: Image Disagreement

**How to read:**
- Stacked bars showing correct vs incorrect images per patient
- Red portions = misclassified images
- Identifies patients with heterogeneous image behavior

### Figure 5: Category Summary

**How to read:**
- Left pie: Distribution of Easy/Medium/Ambiguous/Hard
- Right bars: Count of each problem flag

---

## 10. Output Files

### Directory Structure

```
outputs/
├── lopo/
│   └── lopo_splits.csv           # Patient-fold assignments
├── training/
│   ├── fold_X_seed_Y/            # Per-fold training logs
│   │   ├── training_log.csv      # Loss/metrics per epoch
│   │   ├── epoch_predictions.csv # Per-epoch predictions
│   │   └── best_model.pth        # Best checkpoint
├── cartography/
│   ├── cartography_per_image.xlsx        # Image-level metrics
│   ├── cartography_per_patient_final.xlsx # Patient-level final
│   └── cartography_per_image_all_seeds.csv
├── metrics/
│   ├── lopo_clinical_metrics.csv   # Sens/Spec/etc.
│   ├── lopo_per_patient_predictions.csv
│   ├── probability_margin.csv
│   ├── patient_instability.csv
│   ├── hardest_patients.csv
│   └── patient_categorization.csv
└── figures/
    ├── fig1_confidence_vs_correctness.png
    ├── fig2_seed_variance_correctness.png
    ├── fig3_seed_variance_probability.png
    ├── fig4_image_disagreement.png
    └── fig5_category_summary.png
```

### Key Files Explained

| File | Purpose | Columns |
|------|---------|---------|
| `lopo_splits.csv` | Which patient is held out per fold | patient_id, fold, role |
| `cartography_per_patient_final.xlsx` | Final patient metrics | All metrics + categories |
| `lopo_clinical_metrics.csv` | Overall model performance | Sens, Spec, Acc, CIs |
| `patient_categorization.csv` | Patient categories + flags | Easy/Medium/Hard + flags |

---

## Usage

### Full Pipeline

```bash
cd als_vs_alsftd
python main.py
```

### Individual Stages

```bash
python main.py --stage splits      # Only create LOPO splits
python main.py --stage train       # Only train (requires splits)
python main.py --stage reliability # Re-run reliability checks
python main.py --stage visualize   # Re-generate plots
```

---

## Dependencies

- Python 3.8+
- PyTorch 2.x
- torchvision
- timm (for EfficientNetB0)
- albumentations
- pandas, numpy, scipy
- matplotlib
- openpyxl (for Excel export)

---

## References

- Swayamdipta, S., et al. (2020). "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics." EMNLP.
- Wilson, E.B. (1927). "Probable Inference, the Law of Succession, and Statistical Inference." JASA.
