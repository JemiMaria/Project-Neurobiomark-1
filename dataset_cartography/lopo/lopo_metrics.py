"""
LOPO Clinical Metrics - Compute clinical evaluation metrics.

This module computes:
- Image-level metrics per fold (sensitivity, specificity, etc.)
- Patient-level metrics across all folds
- Confusion matrices and derived statistics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)


def compute_image_level_metrics_per_fold(lopo_per_image):
    """
    Compute image-level clinical metrics for each LOPO fold.
    
    Uses windowed averaged probabilities for prediction.
    
    Args:
        lopo_per_image: DataFrame with per-image windowed metrics
        
    Returns:
        DataFrame with metrics per fold
    """
    results = []
    
    # Get unique folds
    folds = lopo_per_image['fold_patient_id'].unique()
    
    for fold_patient in folds:
        # Get all images for this fold (across all seeds)
        fold_data = lopo_per_image[lopo_per_image['fold_patient_id'] == fold_patient]
        
        # Aggregate across seeds: mean probability per image
        image_agg = fold_data.groupby('image_id').agg({
            'y_true': 'first',
            'mean_prob_window': 'mean',
            'mean_confidence_window': 'mean',
            'mean_correctness_window': 'mean'
        }).reset_index()
        
        y_true = image_agg['y_true'].values
        y_prob = image_agg['mean_prob_window'].values
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Compute metrics
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        metrics['fold_patient_id'] = fold_patient
        metrics['n_images'] = len(image_agg)
        metrics['n_positive'] = int(y_true.sum())
        metrics['n_negative'] = int((1 - y_true).sum())
        metrics['level'] = 'image'
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def compute_patient_level_metrics(lopo_per_patient_final):
    """
    Compute patient-level clinical metrics across all LOPO folds.
    
    Each fold provides one patient prediction (the held-out patient).
    
    Args:
        lopo_per_patient_final: DataFrame with per-patient aggregated metrics
        
    Returns:
        dict: Patient-level metrics
    """
    # Use mean probability across seeds for patient-level prediction
    y_true = lopo_per_patient_final['y_true'].values
    y_prob = lopo_per_patient_final['mean_prob_across_seeds'].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics['n_patients'] = len(lopo_per_patient_final)
    metrics['n_positive'] = int(y_true.sum())
    metrics['n_negative'] = int((1 - y_true).sum())
    metrics['level'] = 'patient'
    metrics['fold_patient_id'] = 'ALL_FOLDS'
    
    return metrics


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)
        
    Returns:
        dict: Metrics dictionary
    """
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        # Only one class present
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': np.nan,
            'specificity': np.nan,
            'balanced_accuracy': np.nan,
            'precision': np.nan,
            'f1': np.nan,
            'auc': np.nan,
            'note': 'single_class'
        }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': (sensitivity + specificity) / 2,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    # AUC (requires probabilities)
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    return metrics


def compute_clinical_metrics(lopo_per_image, lopo_per_patient_final):
    """
    Compute all clinical metrics for LOPO evaluation.
    
    Args:
        lopo_per_image: DataFrame with per-image windowed metrics
        lopo_per_patient_final: DataFrame with per-patient aggregated metrics
        
    Returns:
        DataFrame with all clinical metrics
    """
    print("\nComputing clinical metrics...")
    
    # Image-level metrics per fold
    image_metrics = compute_image_level_metrics_per_fold(lopo_per_image)
    
    # Patient-level metrics (overall)
    patient_metrics = compute_patient_level_metrics(lopo_per_patient_final)
    patient_metrics_df = pd.DataFrame([patient_metrics])
    
    # Combine
    all_metrics = pd.concat([image_metrics, patient_metrics_df], ignore_index=True)
    
    # Reorder columns
    col_order = ['level', 'fold_patient_id', 'n_images', 'n_patients', 
                 'n_positive', 'n_negative',
                 'accuracy', 'sensitivity', 'specificity', 'balanced_accuracy',
                 'precision', 'f1', 'auc', 'tp', 'tn', 'fp', 'fn', 'note']
    col_order = [c for c in col_order if c in all_metrics.columns]
    all_metrics = all_metrics[col_order]
    
    # Print summary
    print("\n--- Image-Level Metrics (Per Fold) ---")
    img_summary = image_metrics[['fold_patient_id', 'n_images', 'accuracy', 
                                  'sensitivity', 'specificity', 'balanced_accuracy']].round(4)
    print(img_summary.to_string(index=False))
    
    print("\n--- Patient-Level Metrics (Overall) ---")
    print(f"  Accuracy:          {patient_metrics['accuracy']:.4f}")
    print(f"  Sensitivity:       {patient_metrics['sensitivity']:.4f}")
    print(f"  Specificity:       {patient_metrics['specificity']:.4f}")
    print(f"  Balanced Accuracy: {patient_metrics['balanced_accuracy']:.4f}")
    print(f"  AUC:               {patient_metrics['auc']:.4f}")
    print(f"  F1:                {patient_metrics['f1']:.4f}")
    
    return all_metrics


def compute_confidence_intervals(metrics_df, confidence=0.95):
    """
    Compute confidence intervals for metrics across folds.
    
    Args:
        metrics_df: DataFrame with per-fold metrics
        confidence: Confidence level (default 0.95)
        
    Returns:
        DataFrame with mean, std, CI for each metric
    """
    from scipy import stats
    
    # Only image-level metrics (per-fold)
    image_metrics = metrics_df[metrics_df['level'] == 'image']
    
    metric_cols = ['accuracy', 'sensitivity', 'specificity', 
                   'balanced_accuracy', 'precision', 'f1', 'auc']
    
    results = []
    for col in metric_cols:
        values = image_metrics[col].dropna().values
        if len(values) > 1:
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            
            # t-distribution CI
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
            ci_lower = mean - t_val * std / np.sqrt(n)
            ci_upper = mean + t_val * std / np.sqrt(n)
            
            results.append({
                'metric': col,
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_folds': n
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("LOPO Metrics Module")
    print("Run via lopo_runner.py")
