"""
Reliability Checks for ALS vs ALS-FTD LOPO Evaluation.

Implements:
- LOPO clinical metrics (Sensitivity, Specificity, Balanced Accuracy, etc.)
- Patient categorization (Easy, Medium, Ambiguous, Hard, Outlier, Borderline, Unstable)
- Probability margin analysis
- Patient instability across seeds
- Hardest patients ranking
- Wilson confidence intervals for small N (~10 patients)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from config import (
    METRICS_DIR,
    EASY_CORRECTNESS_THRESHOLD, EASY_CONFIDENCE_THRESHOLD,
    MEDIUM_CORRECTNESS_MIN, MEDIUM_CORRECTNESS_MAX, MEDIUM_STD_THRESHOLD,
    AMBIGUOUS_CORRECTNESS_MIN, AMBIGUOUS_CORRECTNESS_MAX,
    HARD_CORRECTNESS_THRESHOLD, OUTLIER_SIGMA_THRESHOLD,
    BORDERLINE_MARGIN, UNSTABLE_FLIP_RATE_THRESHOLD,
    UNSTABLE_STD_CORRECTNESS_THRESHOLD, UNSTABLE_STD_PROB_THRESHOLD
)
from utils import compute_wilson_confidence_interval


def compute_lopo_clinical_metrics(patient_final_df, output_dir=None):
    """
    Compute LOPO clinical metrics from patient-level predictions.
    
    Per-patient (binary prediction from mean_prob >= 0.5):
    - TP, TN, FP, FN
    
    Global (aggregate across all patients):
    - Sensitivity, Specificity, Balanced Accuracy, Accuracy, F1
    - 95% Wilson confidence intervals
    
    Args:
        patient_final_df: DataFrame with [patient_id, y_true, final_mean_prob]
        output_dir: Output directory
        
    Returns:
        dict: Computed metrics with CIs
    """
    if output_dir is None:
        output_dir = METRICS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("COMPUTING LOPO CLINICAL METRICS")
    print(f"{'='*60}")
    
    # Binary predictions
    patient_final_df = patient_final_df.copy()
    patient_final_df['y_pred'] = (patient_final_df['final_mean_prob'] >= 0.5).astype(int)
    
    y_true = patient_final_df['y_true'].values
    y_pred = patient_final_df['y_pred'].values
    
    # Confusion matrix
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    n_positive = TP + FN  # Total ALS-FTD
    n_negative = TN + FP  # Total ALS
    n_total = n_positive + n_negative
    
    # Compute metrics
    sensitivity = TP / n_positive if n_positive > 0 else 0
    specificity = TN / n_negative if n_negative > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    accuracy = (TP + TN) / n_total if n_total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    
    # Wilson CIs (appropriate for small N ~10)
    sens_ci = compute_wilson_confidence_interval(TP, n_positive)
    spec_ci = compute_wilson_confidence_interval(TN, n_negative)
    acc_ci = compute_wilson_confidence_interval(TP + TN, n_total)
    
    metrics = {
        'n_patients': n_total,
        'n_positive_alsftd': n_positive,
        'n_negative_als': n_negative,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'sensitivity': sensitivity,
        'sensitivity_95ci_lower': sens_ci[0],
        'sensitivity_95ci_upper': sens_ci[1],
        'specificity': specificity,
        'specificity_95ci_lower': spec_ci[0],
        'specificity_95ci_upper': spec_ci[1],
        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
        'accuracy_95ci_lower': acc_ci[0],
        'accuracy_95ci_upper': acc_ci[1],
        'precision': precision,
        'f1': f1
    }
    
    # Print summary
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted ALS  Predicted ALS-FTD")
    print(f"    Actual ALS         {TN:3d}             {FP:3d}")
    print(f"    Actual ALS-FTD     {FN:3d}             {TP:3d}")
    
    print(f"\n  Clinical Metrics:")
    print(f"    Sensitivity (TPR):    {sensitivity:.3f} [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]")
    print(f"    Specificity (TNR):    {specificity:.3f} [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}]")
    print(f"    Balanced Accuracy:    {balanced_accuracy:.3f}")
    print(f"    Accuracy:             {accuracy:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
    print(f"    F1 Score:             {f1:.3f}")
    
    # Per-patient metrics
    per_patient_records = []
    for _, row in patient_final_df.iterrows():
        per_patient_records.append({
            'patient_id': row['patient_id'],
            'y_true': row['y_true'],
            'y_pred': row['y_pred'],
            'prob': row['final_mean_prob'],
            'correct': 1 if row['y_true'] == row['y_pred'] else 0
        })
    
    per_patient_df = pd.DataFrame(per_patient_records)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "lopo_clinical_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  ✓ Saved clinical metrics: {metrics_path}")
    
    per_patient_path = output_dir / "lopo_per_patient_predictions.csv"
    per_patient_df.to_csv(per_patient_path, index=False)
    print(f"  ✓ Saved per-patient predictions: {per_patient_path}")
    
    return metrics


def compute_probability_margin(patient_final_df, output_dir=None):
    """
    Compute probability margin between classes.
    
    margin = min(prob where y=1) - max(prob where y=0)
    
    Positive margin = good separation
    Negative margin = classes overlap
    
    Args:
        patient_final_df: DataFrame with [patient_id, y_true, final_mean_prob]
        output_dir: Output directory
        
    Returns:
        dict: margin metrics
    """
    if output_dir is None:
        output_dir = METRICS_DIR
    output_dir = Path(output_dir)
    
    print(f"\n  Probability Margin Analysis:")
    
    positive_probs = patient_final_df[patient_final_df['y_true'] == 1]['final_mean_prob']
    negative_probs = patient_final_df[patient_final_df['y_true'] == 0]['final_mean_prob']
    
    if len(positive_probs) == 0 or len(negative_probs) == 0:
        print("    Warning: One class has no samples")
        return None
    
    min_positive_prob = positive_probs.min()
    max_negative_prob = negative_probs.max()
    margin = min_positive_prob - max_negative_prob
    
    margin_metrics = {
        'min_positive_prob': min_positive_prob,
        'max_negative_prob': max_negative_prob,
        'margin': margin,
        'mean_positive_prob': positive_probs.mean(),
        'mean_negative_prob': negative_probs.mean(),
        'std_positive_prob': positive_probs.std(),
        'std_negative_prob': negative_probs.std()
    }
    
    print(f"    Min ALS-FTD prob:  {min_positive_prob:.3f}")
    print(f"    Max ALS prob:      {max_negative_prob:.3f}")
    print(f"    Margin:            {margin:.3f}")
    
    if margin > 0:
        print(f"    → Good: Classes are separable (no overlap)")
    elif margin > -0.1:
        print(f"    → Marginal: Some overlap near threshold")
    else:
        print(f"    → Poor: Significant class overlap")
    
    # Save
    margin_df = pd.DataFrame([margin_metrics])
    margin_path = output_dir / "probability_margin.csv"
    margin_df.to_csv(margin_path, index=False)
    print(f"\n  ✓ Saved probability margin: {margin_path}")
    
    return margin_metrics


def compute_patient_instability(patient_final_df, output_dir=None):
    """
    Compute patient instability metrics from cross-seed analysis.
    
    Already computed in patient_final_df:
    - final_std_correctness: std of correctness across seeds
    - flip_rate: fraction of seeds with different prediction
    
    Args:
        patient_final_df: DataFrame with instability columns
        output_dir: Output directory
        
    Returns:
        pd.DataFrame: Instability metrics per patient
    """
    if output_dir is None:
        output_dir = METRICS_DIR
    output_dir = Path(output_dir)
    
    print(f"\n  Patient Instability Analysis:")
    
    instability_df = patient_final_df[[
        'patient_id', 'y_true', 'final_mean_prob',
        'final_std_prob', 'final_std_correctness', 'flip_rate'
    ]].copy()
    
    # Flag unstable patients
    instability_df['unstable_across_seeds'] = (
        (instability_df['flip_rate'] > UNSTABLE_FLIP_RATE_THRESHOLD) |
        (instability_df['final_std_correctness'] > UNSTABLE_STD_CORRECTNESS_THRESHOLD)
    )
    
    n_unstable = instability_df['unstable_across_seeds'].sum()
    print(f"    Unstable patients (across seeds): {n_unstable} / {len(instability_df)}")
    
    if n_unstable > 0:
        unstable_patients = instability_df[instability_df['unstable_across_seeds']]
        for _, row in unstable_patients.iterrows():
            print(f"      - {row['patient_id']}: flip_rate={row['flip_rate']:.2f}, "
                  f"std_correctness={row['final_std_correctness']:.3f}")
    
    # Save
    instability_path = output_dir / "patient_instability.csv"
    instability_df.to_csv(instability_path, index=False)
    print(f"\n  ✓ Saved patient instability: {instability_path}")
    
    return instability_df


def compute_hardest_patients(patient_final_df, output_dir=None, top_k=5):
    """
    Rank patients by difficulty (lowest correctness).
    
    Primary: lowest mean_correctness
    Secondary (tie-breaker): highest std_prob
    
    Args:
        patient_final_df: DataFrame with patient metrics
        output_dir: Output directory
        top_k: Number of hardest patients to report
        
    Returns:
        pd.DataFrame: Hardest patients ranking
    """
    if output_dir is None:
        output_dir = METRICS_DIR
    output_dir = Path(output_dir)
    
    print(f"\n  Hardest Patients Ranking:")
    
    # Sort by correctness (ascending), then by std_prob (descending)
    hardest_df = patient_final_df.sort_values(
        by=['final_mean_correctness', 'final_std_prob'],
        ascending=[True, False]
    ).copy()
    
    hardest_df['rank'] = range(1, len(hardest_df) + 1)
    
    # Report top K
    print(f"\n    Top {min(top_k, len(hardest_df))} hardest patients:")
    for i, (_, row) in enumerate(hardest_df.head(top_k).iterrows()):
        label = "ALS-FTD" if row['y_true'] == 1 else "ALS"
        print(f"      {i+1}. {row['patient_id']} ({label}): "
              f"correctness={row['final_mean_correctness']:.3f}, "
              f"std_prob={row['final_std_prob']:.3f}")
    
    # Save
    hardest_path = output_dir / "hardest_patients.csv"
    hardest_df.to_csv(hardest_path, index=False)
    print(f"\n  ✓ Saved hardest patients: {hardest_path}")
    
    return hardest_df


def categorize_patients(patient_final_df, output_dir=None):
    """
    Categorize patients based on cartography metrics.
    
    Categories:
    - Easy: high correctness (≥0.8) + high confidence (≥0.8)
    - Medium: 0.6–0.8 correctness + stable (std < 0.1)
    - Ambiguous: 0.4–0.6 correctness
    - Hard: low correctness (<0.4)
    - Outlier: distance from median > 2σ
    - Borderline: |mean_prob - 0.5| ≤ 0.1
    - Unstable (seeds): high flip_rate OR high std_correctness
    - Unstable (images): high std_prob_window
    
    Args:
        patient_final_df: DataFrame with patient metrics
        output_dir: Output directory
        
    Returns:
        pd.DataFrame: Patient categorization
    """
    if output_dir is None:
        output_dir = METRICS_DIR
    output_dir = Path(output_dir)
    
    print(f"\n{'='*60}")
    print("PATIENT CATEGORIZATION")
    print(f"{'='*60}")
    
    df = patient_final_df.copy()
    
    # Initialize categories
    df['category'] = 'Unclassified'
    df['is_borderline'] = False
    df['is_unstable_seeds'] = False
    df['is_unstable_images'] = False
    df['is_outlier'] = False
    
    # Easy: high correctness + high confidence
    easy_mask = (
        (df['final_mean_correctness'] >= EASY_CORRECTNESS_THRESHOLD) &
        (df['final_mean_confidence'] >= EASY_CONFIDENCE_THRESHOLD)
    )
    df.loc[easy_mask, 'category'] = 'Easy'
    
    # Hard: low correctness
    hard_mask = df['final_mean_correctness'] < HARD_CORRECTNESS_THRESHOLD
    df.loc[hard_mask, 'category'] = 'Hard'
    
    # Ambiguous: mid correctness
    ambiguous_mask = (
        (df['final_mean_correctness'] >= AMBIGUOUS_CORRECTNESS_MIN) &
        (df['final_mean_correctness'] < AMBIGUOUS_CORRECTNESS_MAX) &
        (df['category'] == 'Unclassified')
    )
    df.loc[ambiguous_mask, 'category'] = 'Ambiguous'
    
    # Medium: moderate correctness + stable
    medium_mask = (
        (df['final_mean_correctness'] >= MEDIUM_CORRECTNESS_MIN) &
        (df['final_mean_correctness'] < MEDIUM_CORRECTNESS_MAX) &
        (df['final_std_prob'] < MEDIUM_STD_THRESHOLD) &
        (df['category'] == 'Unclassified')
    )
    df.loc[medium_mask, 'category'] = 'Medium'
    
    # Remaining unclassified → Medium (default)
    df.loc[df['category'] == 'Unclassified', 'category'] = 'Medium'
    
    # Borderline flag: |mean_prob - 0.5| ≤ 0.1
    df['is_borderline'] = np.abs(df['final_mean_prob'] - 0.5) <= BORDERLINE_MARGIN
    
    # Unstable (seeds) flag
    df['is_unstable_seeds'] = (
        (df['flip_rate'] > UNSTABLE_FLIP_RATE_THRESHOLD) |
        (df['final_std_correctness'] > UNSTABLE_STD_CORRECTNESS_THRESHOLD)
    )
    
    # Unstable (images) flag
    df['is_unstable_images'] = df['final_std_prob'] > UNSTABLE_STD_PROB_THRESHOLD
    
    # Outlier detection (distance from median in (confidence, correctness) space)
    median_conf = df['final_mean_confidence'].median()
    median_corr = df['final_mean_correctness'].median()
    std_conf = df['final_mean_confidence'].std()
    std_corr = df['final_mean_correctness'].std()
    
    if std_conf > 0 and std_corr > 0:
        df['distance_from_median'] = np.sqrt(
            ((df['final_mean_confidence'] - median_conf) / std_conf)**2 +
            ((df['final_mean_correctness'] - median_corr) / std_corr)**2
        )
        df['is_outlier'] = df['distance_from_median'] > OUTLIER_SIGMA_THRESHOLD
    else:
        df['distance_from_median'] = 0
        df['is_outlier'] = False
    
    # Print summary
    print(f"\n  Category Distribution:")
    for cat in ['Easy', 'Medium', 'Ambiguous', 'Hard']:
        count = (df['category'] == cat).sum()
        print(f"    {cat:12s}: {count}")
    
    print(f"\n  Flags:")
    print(f"    Borderline:       {df['is_borderline'].sum()}")
    print(f"    Unstable (seeds): {df['is_unstable_seeds'].sum()}")
    print(f"    Unstable (images):{df['is_unstable_images'].sum()}")
    print(f"    Outlier:          {df['is_outlier'].sum()}")
    
    # List problem patients
    problem_patients = df[
        df['is_borderline'] | df['is_unstable_seeds'] | 
        df['is_unstable_images'] | df['is_outlier'] |
        (df['category'] == 'Hard')
    ]
    
    if len(problem_patients) > 0:
        print(f"\n  Problem Patients:")
        for _, row in problem_patients.iterrows():
            flags = []
            if row['is_borderline']:
                flags.append('borderline')
            if row['is_unstable_seeds']:
                flags.append('unstable-seeds')
            if row['is_unstable_images']:
                flags.append('unstable-images')
            if row['is_outlier']:
                flags.append('outlier')
            if row['category'] == 'Hard':
                flags.append('hard')
            
            label = "ALS-FTD" if row['y_true'] == 1 else "ALS"
            print(f"    - {row['patient_id']} ({label}): {', '.join(flags)}")
    
    # Save
    cat_path = output_dir / "patient_categorization.csv"
    df.to_csv(cat_path, index=False)
    print(f"\n  ✓ Saved patient categorization: {cat_path}")
    
    return df


def run_reliability_checks(patient_final_df):
    """
    Run all reliability checks.
    
    Args:
        patient_final_df: DataFrame with final patient metrics
        
    Returns:
        dict: All reliability check results
    """
    print(f"\n{'='*60}")
    print("RUNNING RELIABILITY CHECKS")
    print(f"{'='*60}")
    
    results = {}
    
    # Clinical metrics
    results['clinical_metrics'] = compute_lopo_clinical_metrics(patient_final_df)
    
    # Probability margin
    results['probability_margin'] = compute_probability_margin(patient_final_df)
    
    # Patient instability
    results['instability'] = compute_patient_instability(patient_final_df)
    
    # Hardest patients
    results['hardest_patients'] = compute_hardest_patients(patient_final_df)
    
    # Patient categorization
    results['categorization'] = categorize_patients(patient_final_df)
    
    return results
