"""
Reliability analysis for model predictions.

Computes:
- Borderline patients (probability near threshold)
- Prediction instability across seeds (if multiple runs)
- Comprehensive evaluation report
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_DIR


def compute_borderline_patients(patient_df, threshold=0.5, margin=0.1):
    """
    Identify borderline patients (probability near decision threshold).
    
    Borderline patients have |prob - threshold| <= margin
    These cases are clinically uncertain and may require additional review.
    
    Args:
        patient_df: DataFrame with [patient_id, y_true, prob]
        threshold: Classification threshold
        margin: Margin around threshold for borderline definition
        
    Returns:
        tuple: (borderline_df, borderline_rate)
    """
    patient_df = patient_df.copy()
    patient_df['distance_to_threshold'] = np.abs(patient_df['prob'] - threshold)
    patient_df['is_borderline'] = patient_df['distance_to_threshold'] <= margin
    
    borderline_df = patient_df[patient_df['is_borderline']].copy()
    borderline_rate = len(borderline_df) / len(patient_df)
    
    return borderline_df, borderline_rate


def compute_patient_instability(multi_run_patient_dfs):
    """
    Compute prediction instability across multiple runs (seeds).
    
    For each patient, computes:
    - std(prob): Standard deviation of probabilities across runs
    - flip_rate: Fraction of runs where prediction differs from majority
    
    Args:
        multi_run_patient_dfs: List of patient DataFrames from different seeds
        
    Returns:
        pd.DataFrame: Instability metrics per patient
    """
    if len(multi_run_patient_dfs) <= 1:
        print("  ⚠ Only one run available, cannot compute instability")
        return pd.DataFrame()
    
    # Collect probabilities for each patient across runs
    patient_probs = {}
    patient_labels = {}
    
    for run_df in multi_run_patient_dfs:
        for _, row in run_df.iterrows():
            pid = row['patient_id']
            if pid not in patient_probs:
                patient_probs[pid] = []
                patient_labels[pid] = row['y_true']
            patient_probs[pid].append(row['prob'])
    
    # Compute instability metrics
    records = []
    for pid, probs in patient_probs.items():
        probs = np.array(probs)
        preds = (probs >= 0.5).astype(int)
        
        # Majority prediction
        majority = int(preds.mean() >= 0.5)
        
        # Flip rate
        flip_rate = np.mean(preds != majority)
        
        records.append({
            'patient_id': pid,
            'y_true': patient_labels[pid],
            'mean_prob': np.mean(probs),
            'std_prob': np.std(probs),
            'min_prob': np.min(probs),
            'max_prob': np.max(probs),
            'flip_rate': flip_rate,
            'n_runs': len(probs)
        })
    
    instability_df = pd.DataFrame(records)
    instability_df = instability_df.sort_values('std_prob', ascending=False)
    
    return instability_df


def generate_evaluation_report(
    fold_metrics_list,
    fold_thresholds,
    all_patient_dfs,
    calibration_results,
    output_dir=None
):
    """
    Generate comprehensive evaluation report.
    
    Saves:
    - metrics_summary.csv: Cross-fold summary
    - metrics_per_fold.csv: Per-fold metrics
    - borderline_patients.csv: Patients near threshold
    
    Args:
        fold_metrics_list: List of metrics dicts per fold
        fold_thresholds: Dict of thresholds per fold
        all_patient_dfs: List of patient DataFrames per fold
        calibration_results: List of calibration dicts per fold
        output_dir: Output directory
        
    Returns:
        dict: Summary statistics
    """
    if output_dir is None:
        output_dir = EVAL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING EVALUATION REPORT")
    print(f"{'='*60}")
    
    # ===== Per-fold metrics =====
    fold_records = []
    for fold_id, metrics in enumerate(fold_metrics_list):
        record = {'fold_id': fold_id}
        record.update(metrics)
        record['threshold'] = fold_thresholds.get(fold_id, {}).get('threshold', 0.5)
        fold_records.append(record)
    
    metrics_per_fold = pd.DataFrame(fold_records)
    metrics_per_fold.to_csv(output_dir / "metrics_per_fold.csv", index=False)
    print(f"  ✓ Saved: metrics_per_fold.csv")
    
    # ===== Summary metrics =====
    # Compute pooled confusion matrix
    pooled_tp = sum(m['tp'] for m in fold_metrics_list)
    pooled_tn = sum(m['tn'] for m in fold_metrics_list)
    pooled_fp = sum(m['fp'] for m in fold_metrics_list)
    pooled_fn = sum(m['fn'] for m in fold_metrics_list)
    
    from .metrics import compute_wilson_ci
    
    pooled_sens, pooled_sens_lo, pooled_sens_hi = compute_wilson_ci(pooled_tp, pooled_tp + pooled_fn)
    pooled_spec, pooled_spec_lo, pooled_spec_hi = compute_wilson_ci(pooled_tn, pooled_tn + pooled_fp)
    
    # Mean ± std across folds
    summary = {
        'n_folds': len(fold_metrics_list),
        'sensitivity_mean': np.mean([m['sensitivity'] for m in fold_metrics_list]),
        'sensitivity_std': np.std([m['sensitivity'] for m in fold_metrics_list]),
        'specificity_mean': np.mean([m['specificity'] for m in fold_metrics_list]),
        'specificity_std': np.std([m['specificity'] for m in fold_metrics_list]),
        'balanced_accuracy_mean': np.mean([m['balanced_accuracy'] for m in fold_metrics_list]),
        'balanced_accuracy_std': np.std([m['balanced_accuracy'] for m in fold_metrics_list]),
        'roc_auc_mean': np.mean([m['roc_auc'] for m in fold_metrics_list if not np.isnan(m['roc_auc'])]),
        'roc_auc_std': np.std([m['roc_auc'] for m in fold_metrics_list if not np.isnan(m['roc_auc'])]),
        'pooled_sensitivity': pooled_sens,
        'pooled_sensitivity_ci_lower': pooled_sens_lo,
        'pooled_sensitivity_ci_upper': pooled_sens_hi,
        'pooled_specificity': pooled_spec,
        'pooled_specificity_ci_lower': pooled_spec_lo,
        'pooled_specificity_ci_upper': pooled_spec_hi,
        'brier_mean': np.mean([c['brier_score'] for c in calibration_results]),
        'brier_std': np.std([c['brier_score'] for c in calibration_results]),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    print(f"  ✓ Saved: metrics_summary.csv")
    
    # ===== Borderline patients =====
    all_borderline = []
    for fold_id, patient_df in enumerate(all_patient_dfs):
        threshold = fold_thresholds.get(fold_id, {}).get('threshold', 0.5)
        borderline_df, rate = compute_borderline_patients(patient_df, threshold)
        borderline_df['fold_id'] = fold_id
        all_borderline.append(borderline_df)
    
    if all_borderline:
        combined_borderline = pd.concat(all_borderline, ignore_index=True)
        combined_borderline.to_csv(output_dir / "borderline_patients.csv", index=False)
        print(f"  ✓ Saved: borderline_patients.csv ({len(combined_borderline)} borderline cases)")
        
        # Compute overall borderline rate
        total_patients = sum(len(df) for df in all_patient_dfs)
        borderline_rate = len(combined_borderline) / total_patients if total_patients > 0 else 0
        summary['borderline_rate'] = borderline_rate
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Sensitivity: {summary['sensitivity_mean']:.3f} ± {summary['sensitivity_std']:.3f}")
    print(f"    Specificity: {summary['specificity_mean']:.3f} ± {summary['specificity_std']:.3f}")
    print(f"    Balanced Acc: {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_std']:.3f}")
    print(f"    ROC AUC: {summary['roc_auc_mean']:.3f} ± {summary['roc_auc_std']:.3f}")
    print(f"\n  Pooled 95% Wilson CIs:")
    print(f"    Sensitivity: [{pooled_sens_lo:.3f}, {pooled_sens_hi:.3f}]")
    print(f"    Specificity: [{pooled_spec_lo:.3f}, {pooled_spec_hi:.3f}]")
    
    return summary


def save_patient_instability(instability_df, output_path=None):
    """
    Save patient instability analysis results.
    
    Args:
        instability_df: DataFrame from compute_patient_instability
        output_path: Path to save CSV
    """
    if output_path is None:
        output_path = EVAL_DIR / "patient_instability.csv"
    
    if len(instability_df) > 0:
        instability_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved: {output_path}")
    else:
        print(f"  ⚠ No instability data to save (single run)")
