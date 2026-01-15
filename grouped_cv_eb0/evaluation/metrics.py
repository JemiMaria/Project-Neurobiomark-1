"""
Clinical metrics computation for ALS classification.

All metrics are computed at PATIENT level (primary) with image-level as secondary.
Uses Wilson confidence intervals for small sample sizes.
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVICE


def compute_wilson_ci(successes, total, confidence=0.95):
    """
    Compute Wilson score confidence interval for a proportion.
    
    Recommended for small sample sizes (e.g., n=15 patients).
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        tuple: (proportion, ci_lower, ci_upper)
    """
    if total == 0:
        return (np.nan, np.nan, np.nan)
    
    p = successes / total
    n = total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    
    return (p, ci_lower, ci_upper)


@torch.no_grad()
def compute_image_level_predictions(model, dataloader, device=None):
    """
    Get predictions for all images in dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader to evaluate
        device: torch device
        
    Returns:
        pd.DataFrame: Predictions with columns [image_path, patient_id, y_true, prob, logit]
    """
    if device is None:
        device = DEVICE
    
    model.eval()
    model.to(device)
    
    records = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        patient_ids = batch['patient_id']
        image_paths = batch['image_path']
        
        logits = model(images).squeeze(-1).cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        for i in range(len(labels)):
            records.append({
                'image_path': image_paths[i],
                'patient_id': patient_ids[i],
                'y_true': int(labels[i]),
                'prob': float(probs[i]),
                'logit': float(logits[i])
            })
    
    return pd.DataFrame(records)


def aggregate_to_patient_level(image_preds_df, agg_method='mean'):
    """
    Aggregate image-level predictions to patient level.
    
    Args:
        image_preds_df: DataFrame with image predictions
        agg_method: Aggregation method ('mean', 'median', 'max')
        
    Returns:
        pd.DataFrame: Patient-level predictions
    """
    agg_funcs = {
        'mean': 'mean',
        'median': 'median',
        'max': 'max'
    }
    
    if agg_method not in agg_funcs:
        raise ValueError(f"Unknown agg_method: {agg_method}")
    
    patient_df = image_preds_df.groupby('patient_id').agg({
        'y_true': 'first',  # Same label per patient
        'prob': agg_funcs[agg_method],
        'logit': agg_funcs[agg_method]
    }).reset_index()
    
    patient_df['n_images'] = image_preds_df.groupby('patient_id').size().values
    
    return patient_df


def compute_patient_level_metrics(patient_df, threshold=0.5):
    """
    Compute clinical metrics at patient level.
    
    Metrics:
    - Sensitivity (TPR): TP / (TP + FN)
    - Specificity (TNR): TN / (TN + FP)
    - Balanced Accuracy: (Sens + Spec) / 2
    - ROC AUC
    
    Args:
        patient_df: DataFrame with [patient_id, y_true, prob]
        threshold: Classification threshold
        
    Returns:
        dict: Metrics with confidence intervals
    """
    y_true = patient_df['y_true'].values
    y_prob = patient_df['prob'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    balanced_acc = (sensitivity + specificity) / 2 if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan
    accuracy = (tp + tn) / len(y_true)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan  # Only one class present
    
    # Wilson CIs
    sens_val, sens_ci_lo, sens_ci_hi = compute_wilson_ci(tp, tp + fn)
    spec_val, spec_ci_lo, spec_ci_hi = compute_wilson_ci(tn, tn + fp)
    acc_val, acc_ci_lo, acc_ci_hi = compute_wilson_ci(tp + tn, len(y_true))
    
    return {
        'n_patients': len(y_true),
        'n_cases': int(tp + fn),
        'n_controls': int(tn + fp),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'threshold': threshold,
        'sensitivity': sensitivity,
        'sensitivity_ci_lower': sens_ci_lo,
        'sensitivity_ci_upper': sens_ci_hi,
        'specificity': specificity,
        'specificity_ci_lower': spec_ci_lo,
        'specificity_ci_upper': spec_ci_hi,
        'balanced_accuracy': balanced_acc,
        'accuracy': accuracy,
        'accuracy_ci_lower': acc_ci_lo,
        'accuracy_ci_upper': acc_ci_hi,
        'roc_auc': roc_auc
    }


def compute_fold_metrics(model, val_loader, threshold=0.5, device=None):
    """
    Compute all metrics for a validation fold.
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        threshold: Classification threshold
        device: torch device
        
    Returns:
        tuple: (metrics_dict, patient_df, image_df)
    """
    # Get image-level predictions
    image_df = compute_image_level_predictions(model, val_loader, device)
    
    # Aggregate to patient level
    patient_df = aggregate_to_patient_level(image_df)
    
    # Compute metrics
    metrics = compute_patient_level_metrics(patient_df, threshold)
    
    return metrics, patient_df, image_df


def compute_cv_summary(fold_metrics_list, pooled_confusion=None):
    """
    Compute cross-validation summary statistics.
    
    Computes mean ± std across folds for each metric.
    Optionally computes pooled Wilson CIs from combined confusion matrix.
    
    Args:
        fold_metrics_list: List of metrics dicts from each fold
        pooled_confusion: Dict with pooled {tp, tn, fp, fn} (optional)
        
    Returns:
        dict: Summary with mean, std, and optionally pooled CIs
    """
    # Stack metrics into arrays
    metrics_df = pd.DataFrame(fold_metrics_list)
    
    # Compute mean and std
    summary = {}
    for col in ['sensitivity', 'specificity', 'balanced_accuracy', 'accuracy', 'roc_auc']:
        values = metrics_df[col].dropna()
        summary[f'{col}_mean'] = values.mean()
        summary[f'{col}_std'] = values.std()
    
    summary['n_folds'] = len(fold_metrics_list)
    
    # Pooled Wilson CIs (if provided)
    if pooled_confusion is not None:
        tp = pooled_confusion['tp']
        tn = pooled_confusion['tn']
        fp = pooled_confusion['fp']
        fn = pooled_confusion['fn']
        
        sens_val, sens_ci_lo, sens_ci_hi = compute_wilson_ci(tp, tp + fn)
        spec_val, spec_ci_lo, spec_ci_hi = compute_wilson_ci(tn, tn + fp)
        
        summary['pooled_sensitivity'] = sens_val
        summary['pooled_sensitivity_ci_lower'] = sens_ci_lo
        summary['pooled_sensitivity_ci_upper'] = sens_ci_hi
        summary['pooled_specificity'] = spec_val
        summary['pooled_specificity_ci_lower'] = spec_ci_lo
        summary['pooled_specificity_ci_upper'] = spec_ci_hi
    
    return summary
