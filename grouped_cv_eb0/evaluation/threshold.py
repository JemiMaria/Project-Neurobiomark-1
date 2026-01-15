"""
Threshold tuning with clinical constraints.

LEAKAGE PREVENTION:
- Threshold is tuned PER FOLD using validation fold only
- No information from other folds or test set is used

Clinical constraint:
- Sensitivity must be >= MIN_SENSITIVITY (e.g., 0.70)
- Among thresholds meeting this constraint, pick best specificity
- Tie-breaker: best balanced accuracy
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIN_SENSITIVITY, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS, THRESHOLDS_DIR

from .metrics import compute_patient_level_metrics


def find_optimal_threshold(patient_df, min_sensitivity=None, threshold_range=None):
    """
    Find optimal threshold with sensitivity constraint.
    
    Algorithm:
    1. Evaluate all thresholds in range
    2. Filter to those meeting min_sensitivity constraint
    3. Among those, pick threshold with best specificity
    4. Tie-breaker: best balanced accuracy
    
    Args:
        patient_df: DataFrame with [patient_id, y_true, prob]
        min_sensitivity: Minimum required sensitivity (default from config)
        threshold_range: Tuple (min, max, steps) or None for config defaults
        
    Returns:
        dict: {threshold, sensitivity, specificity, balanced_accuracy, meets_constraint}
    """
    if min_sensitivity is None:
        min_sensitivity = MIN_SENSITIVITY
    
    if threshold_range is None:
        thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    else:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
    
    # Evaluate all thresholds
    results = []
    for thresh in thresholds:
        metrics = compute_patient_level_metrics(patient_df, threshold=thresh)
        results.append({
            'threshold': thresh,
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'tp': metrics['tp'],
            'fn': metrics['fn'],
        })
    
    results_df = pd.DataFrame(results)
    
    # Filter by sensitivity constraint
    valid = results_df[results_df['sensitivity'] >= min_sensitivity]
    
    if len(valid) == 0:
        # No threshold meets constraint - pick threshold with best sensitivity
        print(f"  ⚠ No threshold achieves sensitivity >= {min_sensitivity}")
        best_idx = results_df['sensitivity'].idxmax()
        best = results_df.loc[best_idx]
        return {
            'threshold': best['threshold'],
            'sensitivity': best['sensitivity'],
            'specificity': best['specificity'],
            'balanced_accuracy': best['balanced_accuracy'],
            'meets_constraint': False
        }
    
    # Among valid thresholds, pick best specificity (then balanced accuracy)
    valid = valid.sort_values(
        ['specificity', 'balanced_accuracy'],
        ascending=[False, False]
    )
    
    best = valid.iloc[0]
    
    return {
        'threshold': best['threshold'],
        'sensitivity': best['sensitivity'],
        'specificity': best['specificity'],
        'balanced_accuracy': best['balanced_accuracy'],
        'meets_constraint': True
    }


def tune_threshold_per_fold(fold_patient_dfs, min_sensitivity=None, save=True):
    """
    Tune threshold for each fold using only that fold's validation data.
    
    LEAKAGE PREVENTION:
    - Each fold's threshold is determined using ONLY that fold's validation patients
    - No cross-fold information is used
    
    Args:
        fold_patient_dfs: Dict {fold_id: patient_df}
        min_sensitivity: Minimum required sensitivity
        save: Whether to save thresholds to CSV
        
    Returns:
        dict: {fold_id: threshold_info}
    """
    if min_sensitivity is None:
        min_sensitivity = MIN_SENSITIVITY
    
    print(f"\n{'='*60}")
    print("THRESHOLD TUNING (Clinical Constraint)")
    print(f"{'='*60}")
    print(f"  Constraint: Sensitivity >= {min_sensitivity}")
    print(f"  Objective: Maximize Specificity, then Balanced Accuracy")
    
    fold_thresholds = {}
    records = []
    
    for fold_id, patient_df in fold_patient_dfs.items():
        result = find_optimal_threshold(patient_df, min_sensitivity)
        fold_thresholds[fold_id] = result
        
        status = "✓" if result['meets_constraint'] else "⚠"
        print(f"\n  Fold {fold_id}: threshold={result['threshold']:.3f} | "
              f"Sens={result['sensitivity']:.3f} | Spec={result['specificity']:.3f} | {status}")
        
        records.append({
            'fold_id': fold_id,
            'threshold': result['threshold'],
            'sensitivity': result['sensitivity'],
            'specificity': result['specificity'],
            'balanced_accuracy': result['balanced_accuracy'],
            'meets_constraint': result['meets_constraint']
        })
    
    # Save thresholds
    if save:
        thresholds_df = pd.DataFrame(records)
        save_path = THRESHOLDS_DIR / "fold_thresholds.csv"
        thresholds_df.to_csv(save_path, index=False)
        print(f"\n  ✓ Saved: {save_path}")
    
    return fold_thresholds


def get_fold_threshold(fold_id, thresholds_path=None):
    """
    Load threshold for a specific fold.
    
    Args:
        fold_id: Fold ID
        thresholds_path: Path to thresholds CSV
        
    Returns:
        float: Threshold for this fold
    """
    if thresholds_path is None:
        thresholds_path = THRESHOLDS_DIR / "fold_thresholds.csv"
    
    thresholds_df = pd.read_csv(thresholds_path)
    row = thresholds_df[thresholds_df['fold_id'] == fold_id]
    
    if len(row) == 0:
        raise ValueError(f"No threshold found for fold {fold_id}")
    
    return row['threshold'].values[0]
