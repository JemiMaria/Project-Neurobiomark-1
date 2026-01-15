"""
LOPO Reliability Checks at Patient Level

This module performs reliability analysis on LOPO results:
A) Global clinical metrics with 95% Wilson confidence intervals
B) Probability margin analysis (separation between classes)
C) Seed variance analysis (prediction instability across runs)
D) Hardest patient identification

Input files (from lopo_runner.py):
- lopo_per_patient_final.xlsx
- lopo_per_patient_seed.xlsx

Output files:
- clinical_metrics_with_ci.csv
- probability_margin_summary.csv
- patient_probability_details.csv
- patient_seed_instability.csv
- hardest_patient_report.csv
- plot_seed_variance_prob.png
- plot_seed_variance_correctness.png
- README.md
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import LOPO_DIR
except ModuleNotFoundError:
    from dataset_cartography.config import LOPO_DIR


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def dataframe_to_markdown_simple(df):
    """
    Convert DataFrame to markdown table without requiring tabulate.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        str: Markdown formatted table
    """
    if len(df) == 0:
        return "No data"
    
    # Get column names
    cols = df.columns.tolist()
    
    # Header row
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "|" + "|".join(["---" for _ in cols]) + "|"
    
    # Data rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)
    
    return "\n".join([header, separator] + rows)


def compute_wilson_confidence_interval(successes, total, confidence=0.95):
    """
    Compute Wilson score confidence interval for a proportion.
    
    This is recommended for small sample sizes (like n=15 patients).
    
    Args:
        successes: Number of successes (e.g., correct predictions)
        total: Total number of trials (e.g., total patients)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (proportion, ci_lower, ci_upper)
    """
    from scipy import stats
    
    # Handle edge cases
    if total == 0:
        return (np.nan, np.nan, np.nan)
    
    # Proportion
    p = successes / total
    n = total
    
    # Z-score for confidence level
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Wilson score interval formula
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    
    return (p, ci_lower, ci_upper)


# =============================================================================
# A) GLOBAL CLINICAL METRICS WITH 95% CONFIDENCE INTERVALS
# =============================================================================

def compute_clinical_metrics_with_ci(patient_final_df):
    """
    Compute clinical metrics with 95% Wilson confidence intervals.
    
    Metrics computed:
    - Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate)
    - Balanced Accuracy
    - Overall Accuracy
    
    Args:
        patient_final_df: DataFrame with columns [patient_id, y_true, mean_prob_across_seeds]
    
    Returns:
        DataFrame with columns [metric, value, ci_lower, ci_upper]
    """
    print("\n" + "="*60)
    print("A) GLOBAL CLINICAL METRICS WITH 95% CONFIDENCE INTERVALS")
    print("="*60)
    
    # Step 1: Extract true labels and predictions
    y_true = patient_final_df['y_true'].values
    y_prob = patient_final_df['mean_prob_across_seeds'].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    print(f"\nTotal patients: {len(y_true)}")
    print(f"  - Cases (y_true=1): {y_true.sum()}")
    print(f"  - Controls (y_true=0): {(1 - y_true).sum()}")
    
    # Step 2: Build confusion matrix components
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {true_positives}")
    print(f"  True Negatives:  {true_negatives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    
    # Step 3: Compute metrics with Wilson CIs
    total_cases = true_positives + false_negatives
    total_controls = true_negatives + false_positives
    total_patients = len(y_true)
    total_correct = true_positives + true_negatives
    
    # Sensitivity = TP / (TP + FN)
    sens_value, sens_ci_lower, sens_ci_upper = compute_wilson_confidence_interval(
        true_positives, total_cases
    )
    
    # Specificity = TN / (TN + FP)
    spec_value, spec_ci_lower, spec_ci_upper = compute_wilson_confidence_interval(
        true_negatives, total_controls
    )
    
    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    # CI for balanced accuracy: use bootstrap or approximate
    balanced_acc = (sens_value + spec_value) / 2
    balanced_acc_ci_lower = (sens_ci_lower + spec_ci_lower) / 2
    balanced_acc_ci_upper = (sens_ci_upper + spec_ci_upper) / 2
    
    # Overall Accuracy
    acc_value, acc_ci_lower, acc_ci_upper = compute_wilson_confidence_interval(
        total_correct, total_patients
    )
    
    # Step 4: Create results DataFrame
    metrics_data = [
        {'metric': 'Sensitivity', 'value': sens_value, 
         'ci_lower': sens_ci_lower, 'ci_upper': sens_ci_upper},
        {'metric': 'Specificity', 'value': spec_value, 
         'ci_lower': spec_ci_lower, 'ci_upper': spec_ci_upper},
        {'metric': 'Balanced_Accuracy', 'value': balanced_acc, 
         'ci_lower': balanced_acc_ci_lower, 'ci_upper': balanced_acc_ci_upper},
        {'metric': 'Overall_Accuracy', 'value': acc_value, 
         'ci_lower': acc_ci_lower, 'ci_upper': acc_ci_upper},
    ]
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Step 5: Print summary
    print(f"\nClinical Metrics with 95% Wilson CI:")
    print("-" * 50)
    for _, row in metrics_df.iterrows():
        print(f"  {row['metric']:20s}: {row['value']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    
    return metrics_df


# =============================================================================
# B) PROBABILITY MARGIN ANALYSIS
# =============================================================================

def compute_probability_margin(patient_final_df):
    """
    Analyze class separation using probability margins.
    
    Computes:
    - min_case_prob: Minimum probability among Case patients
    - max_control_prob: Maximum probability among Control patients
    - margin: min_case_prob - max_control_prob (positive = good separation)
    
    Args:
        patient_final_df: DataFrame with columns [patient_id, y_true, mean_prob_across_seeds]
    
    Returns:
        tuple: (margin_summary_df, patient_details_df)
    """
    print("\n" + "="*60)
    print("B) PROBABILITY MARGIN ANALYSIS")
    print("="*60)
    
    # Step 1: Separate cases and controls
    cases = patient_final_df[patient_final_df['y_true'] == 1]
    controls = patient_final_df[patient_final_df['y_true'] == 0]
    
    print(f"\nCases (n={len(cases)}): probability range = "
          f"[{cases['mean_prob_across_seeds'].min():.4f}, {cases['mean_prob_across_seeds'].max():.4f}]")
    print(f"Controls (n={len(controls)}): probability range = "
          f"[{controls['mean_prob_across_seeds'].min():.4f}, {controls['mean_prob_across_seeds'].max():.4f}]")
    
    # Step 2: Compute margin metrics
    min_case_prob = cases['mean_prob_across_seeds'].min()
    max_control_prob = controls['mean_prob_across_seeds'].max()
    margin = min_case_prob - max_control_prob
    
    print(f"\nMargin Analysis:")
    print(f"  - Minimum Case probability:    {min_case_prob:.4f}")
    print(f"  - Maximum Control probability: {max_control_prob:.4f}")
    print(f"  - Margin (min_case - max_ctrl): {margin:.4f}")
    
    if margin > 0:
        print(f"  → GOOD: Classes are separable (no overlap)")
    else:
        print(f"  → WARNING: Classes overlap (margin < 0)")
    
    # Step 3: Create margin summary
    margin_summary = pd.DataFrame([
        {'metric': 'min_case_prob', 'value': min_case_prob},
        {'metric': 'max_control_prob', 'value': max_control_prob},
        {'metric': 'margin', 'value': margin}
    ])
    
    # Step 4: Create patient details table
    # Use fold_patient_id as the patient identifier column
    patient_col = 'fold_patient_id' if 'fold_patient_id' in patient_final_df.columns else 'patient_id'
    patient_details = patient_final_df[[patient_col, 'y_true', 'mean_prob_across_seeds']].copy()
    patient_details = patient_details.rename(columns={patient_col: 'patient_id'})
    patient_details['patient_pred'] = (patient_details['mean_prob_across_seeds'] >= 0.5).astype(int)
    patient_details['prob_margin_to_threshold'] = np.abs(patient_details['mean_prob_across_seeds'] - 0.5)
    patient_details = patient_details.sort_values('mean_prob_across_seeds', ascending=False)
    
    print(f"\nPatient Probability Details:")
    print(patient_details.to_string(index=False))
    
    return margin_summary, patient_details


# =============================================================================
# C) SEED VARIANCE (INSTABILITY ACROSS RUNS)
# =============================================================================

def compute_seed_instability(patient_seed_df):
    """
    Analyze prediction instability across training seeds.
    
    For each patient, computes:
    - mean_prob, std_prob across seeds
    - mean_correctness, std_correctness across seeds
    - flip_rate: fraction of seeds where prediction differs from majority
    
    Args:
        patient_seed_df: DataFrame with per-patient-seed results
    
    Returns:
        DataFrame with instability metrics per patient
    """
    print("\n" + "="*60)
    print("C) SEED VARIANCE (INSTABILITY ACROSS RUNS)")
    print("="*60)
    
    # Step 1: Group by patient and compute statistics
    instability_records = []
    
    # Determine column names (handle different naming conventions)
    patient_col = 'fold_patient_id' if 'fold_patient_id' in patient_seed_df.columns else 'patient_id'
    prob_col = 'patient_mean_prob' if 'patient_mean_prob' in patient_seed_df.columns else 'mean_prob'
    correctness_col = 'patient_mean_correctness' if 'patient_mean_correctness' in patient_seed_df.columns else 'mean_correctness'
    
    for patient_id, group in patient_seed_df.groupby(patient_col):
        # Get probability values across seeds
        probs = group[prob_col].values
        
        # Get correctness values across seeds (if available)
        if correctness_col in group.columns:
            correctness_values = group[correctness_col].values
        else:
            # Compute correctness from probability
            y_true = group['y_true'].iloc[0]
            correctness_values = np.where(y_true == 1, probs, 1 - probs)
        
        # Compute predictions per seed
        predictions_per_seed = (probs >= 0.5).astype(int)
        
        # Majority prediction (mode)
        majority_pred = int(predictions_per_seed.mean() >= 0.5)
        
        # Flip rate: fraction of seeds that disagree with majority
        flip_rate = np.mean(predictions_per_seed != majority_pred)
        
        instability_records.append({
            'patient_id': patient_id,
            'y_true': group['y_true'].iloc[0],
            'mean_prob': np.mean(probs),
            'std_prob': np.std(probs),
            'mean_correctness': np.mean(correctness_values),
            'std_correctness': np.std(correctness_values),
            'flip_rate': flip_rate,
            'n_seeds': len(probs)
        })
    
    instability_df = pd.DataFrame(instability_records)
    instability_df = instability_df.sort_values('std_prob', ascending=False)
    
    # Step 2: Print summary
    print(f"\nPatients analyzed: {len(instability_df)}")
    print(f"\nInstability Summary (sorted by std_prob descending):")
    print("-" * 80)
    
    display_cols = ['patient_id', 'y_true', 'mean_prob', 'std_prob', 
                    'mean_correctness', 'std_correctness', 'flip_rate']
    print(instability_df[display_cols].round(4).to_string(index=False))
    
    # Step 3: Identify high-instability patients
    high_instability = instability_df[instability_df['std_prob'] > 0.1]
    if len(high_instability) > 0:
        print(f"\n⚠ High instability patients (std_prob > 0.1): {len(high_instability)}")
        for _, row in high_instability.iterrows():
            print(f"   - {row['patient_id']}: std_prob={row['std_prob']:.4f}, flip_rate={row['flip_rate']:.2f}")
    else:
        print(f"\n✓ No high-instability patients (all std_prob <= 0.1)")
    
    return instability_df


def plot_seed_variance(instability_df, output_dir):
    """
    Create error bar plots for seed variance analysis.
    
    Creates two plots:
    1. Probability variance across seeds
    2. Correctness variance across seeds
    """
    print("\nGenerating seed variance plots...")
    
    # Sort by mean_prob for consistent visualization
    df = instability_df.sort_values('mean_prob')
    
    # --- Plot 1: Probability Variance ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df))
    colors = ['tab:orange' if y == 1 else 'tab:blue' for y in df['y_true']]
    
    ax.bar(x, df['mean_prob'], yerr=df['std_prob'], capsize=4, 
           color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['patient_id'], rotation=45, ha='right')
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel('Mean Probability (± 1 Std across Seeds)', fontsize=12)
    ax.set_title('Seed Variance: Probability Predictions per Patient', fontsize=14)
    ax.set_ylim(0, 1)
    
    # Legend for class labels
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor='tab:orange', label='Case (y_true=1)'),
        Patch(facecolor='tab:blue', label='Control (y_true=0)'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Threshold (0.5)')
    ]
    ax.legend(handles=legend_patches, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_seed_variance_prob.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: plot_seed_variance_prob.png")
    
    # --- Plot 2: Correctness Variance ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_sorted = df.sort_values('mean_correctness')
    x = np.arange(len(df_sorted))
    colors = ['tab:orange' if y == 1 else 'tab:blue' for y in df_sorted['y_true']]
    
    ax.bar(x, df_sorted['mean_correctness'], yerr=df_sorted['std_correctness'], 
           capsize=4, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random Guess (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['patient_id'], rotation=45, ha='right')
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel('Mean Correctness (± 1 Std across Seeds)', fontsize=12)
    ax.set_title('Seed Variance: Correctness per Patient', fontsize=14)
    ax.set_ylim(0, 1.05)
    
    ax.legend(handles=legend_patches, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_seed_variance_correctness.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved: plot_seed_variance_correctness.png")


# =============================================================================
# D) IDENTIFY HARDEST PATIENTS
# =============================================================================

def identify_hardest_patients(instability_df, patient_final_df):
    """
    Identify the hardest patients to classify.
    
    Hardest patient criteria:
    - Primary: lowest mean_correctness
    - Tie-breaker: highest std_prob
    
    Also flags borderline patients (probability within 0.1 of 0.5).
    
    Args:
        instability_df: DataFrame with seed instability metrics
        patient_final_df: DataFrame with final patient predictions
    
    Returns:
        DataFrame with hardest patient report
    """
    print("\n" + "="*60)
    print("D) IDENTIFY HARDEST PATIENTS")
    print("="*60)
    
    # Step 1: Merge data for comprehensive view
    merged = instability_df.copy()
    
    # Step 2: Sort by hardness (lowest correctness first, then highest std_prob)
    merged = merged.sort_values(
        ['mean_correctness', 'std_prob'], 
        ascending=[True, False]
    )
    
    # Step 3: Add rank
    merged['rank'] = range(1, len(merged) + 1)
    
    # Step 4: Get top 5 hardest
    top5_hardest = merged.head(5)[['rank', 'patient_id', 'y_true', 
                                    'mean_correctness', 'std_prob', 'flip_rate']].copy()
    
    print(f"\nTop 5 Hardest Patients:")
    print("-" * 70)
    print(top5_hardest.round(4).to_string(index=False))
    
    # Step 5: Identify borderline patients (probability within 0.1 of threshold)
    borderline_threshold = 0.1
    merged['prob_distance_to_threshold'] = np.abs(merged['mean_prob'] - 0.5)
    borderline_patients = merged[merged['prob_distance_to_threshold'] <= borderline_threshold]
    
    print(f"\nBorderline Patients (|prob - 0.5| <= {borderline_threshold}):")
    print("-" * 70)
    if len(borderline_patients) > 0:
        borderline_display = borderline_patients[['patient_id', 'y_true', 'mean_prob', 
                                                   'prob_distance_to_threshold']].round(4)
        print(borderline_display.to_string(index=False))
    else:
        print("  No borderline patients found.")
    
    # Step 6: Create report DataFrame
    report_df = merged[['rank', 'patient_id', 'y_true', 'mean_prob',
                        'mean_correctness', 'std_prob', 'flip_rate', 
                        'prob_distance_to_threshold']].copy()
    report_df = report_df.rename(columns={'prob_distance_to_threshold': 'distance_to_threshold'})
    
    return report_df, borderline_patients


# =============================================================================
# E) EXCLUDED PATIENTS SENSITIVITY ANALYSIS
# =============================================================================

def identify_exclusion_groups(patient_final_df, borderline_threshold=0.1):
    """
    Identify patients for exclusion in sensitivity analysis.
    
    Groups:
    - Borderline: |mean_prob - 0.5| <= borderline_threshold
    - Hardest: Single patient with lowest mean_correctness (tie-breaker: closest to threshold)
    - Both: Union of borderline and hardest
    
    Args:
        patient_final_df: DataFrame with patient-level results
        borderline_threshold: Threshold for borderline classification (default 0.1)
    
    Returns:
        dict with keys 'borderline_ids', 'hardest_id', 'both_ids', 'exclusion_details_df'
    """
    # Determine column names
    patient_col = 'fold_patient_id' if 'fold_patient_id' in patient_final_df.columns else 'patient_id'
    prob_col = 'mean_prob_across_seeds'
    correctness_col = 'mean_correctness_across_seeds'
    
    # Calculate distance to threshold for all patients
    df = patient_final_df.copy()
    df['prob_margin_to_threshold'] = np.abs(df[prob_col] - 0.5)
    
    # Group 1: Borderline patients
    borderline_mask = df['prob_margin_to_threshold'] <= borderline_threshold
    borderline_ids = set(df[borderline_mask][patient_col].tolist())
    
    # Group 2: Hardest patient (lowest correctness, tie-breaker: closest to threshold)
    df_sorted = df.sort_values(
        [correctness_col, 'prob_margin_to_threshold'],
        ascending=[True, True]
    )
    hardest_id = df_sorted.iloc[0][patient_col]
    
    # Group 3: Union of borderline and hardest
    both_ids = borderline_ids | {hardest_id}
    
    # Create exclusion details dataframe
    exclusion_records = []
    
    for patient_id in both_ids:
        patient_row = df[df[patient_col] == patient_id].iloc[0]
        reasons = []
        if patient_id in borderline_ids:
            reasons.append('borderline')
        if patient_id == hardest_id:
            reasons.append('hardest')
        
        exclusion_records.append({
            'patient_id': patient_id,
            'reason': ' + '.join(reasons) if len(reasons) > 1 else reasons[0],
            'y_true': patient_row['y_true'],
            'mean_prob_across_seeds': patient_row[prob_col],
            'mean_correctness_across_seeds': patient_row[correctness_col],
            'prob_margin_to_threshold': patient_row['prob_margin_to_threshold']
        })
    
    exclusion_details_df = pd.DataFrame(exclusion_records)
    if len(exclusion_details_df) > 0:
        exclusion_details_df = exclusion_details_df.sort_values('reason')
    
    return {
        'borderline_ids': borderline_ids,
        'hardest_id': hardest_id,
        'both_ids': both_ids,
        'exclusion_details_df': exclusion_details_df
    }


def compute_metrics_for_subset(patient_df, scenario_name):
    """
    Compute clinical metrics for a subset of patients.
    
    Reuses the same Wilson CI computation as baseline.
    
    Args:
        patient_df: Filtered DataFrame with patient subset
        scenario_name: Name for this scenario (e.g., 'baseline', 'no_borderline')
    
    Returns:
        dict with all metric values and CIs
    """
    n_patients = len(patient_df)
    
    # Handle empty dataframe
    if n_patients == 0:
        return {
            'scenario': scenario_name,
            'n_patients': 0,
            'n_cases': 0,
            'n_controls': 0,
            'sensitivity': np.nan,
            'sensitivity_ci_lower': np.nan,
            'sensitivity_ci_upper': np.nan,
            'specificity': np.nan,
            'specificity_ci_lower': np.nan,
            'specificity_ci_upper': np.nan,
            'balanced_accuracy': np.nan,
            'balanced_accuracy_ci_lower': np.nan,
            'balanced_accuracy_ci_upper': np.nan,
            'overall_accuracy': np.nan,
            'overall_accuracy_ci_lower': np.nan,
            'overall_accuracy_ci_upper': np.nan
        }
    
    # Extract true labels and predictions
    y_true = patient_df['y_true'].values
    y_prob = patient_df['mean_prob_across_seeds'].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Confusion matrix components
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    total_cases = true_positives + false_negatives
    total_controls = true_negatives + false_positives
    total_correct = true_positives + true_negatives
    
    # Compute metrics with Wilson CIs
    sens_value, sens_ci_lower, sens_ci_upper = compute_wilson_confidence_interval(
        true_positives, total_cases
    )
    spec_value, spec_ci_lower, spec_ci_upper = compute_wilson_confidence_interval(
        true_negatives, total_controls
    )
    
    # Balanced Accuracy
    balanced_acc = (sens_value + spec_value) / 2 if not (np.isnan(sens_value) or np.isnan(spec_value)) else np.nan
    balanced_acc_ci_lower = (sens_ci_lower + spec_ci_lower) / 2 if not (np.isnan(sens_ci_lower) or np.isnan(spec_ci_lower)) else np.nan
    balanced_acc_ci_upper = (sens_ci_upper + spec_ci_upper) / 2 if not (np.isnan(sens_ci_upper) or np.isnan(spec_ci_upper)) else np.nan
    
    # Overall Accuracy
    acc_value, acc_ci_lower, acc_ci_upper = compute_wilson_confidence_interval(
        total_correct, n_patients
    )
    
    return {
        'scenario': scenario_name,
        'n_patients': n_patients,
        'n_cases': int(total_cases),
        'n_controls': int(total_controls),
        'sensitivity': sens_value,
        'sensitivity_ci_lower': sens_ci_lower,
        'sensitivity_ci_upper': sens_ci_upper,
        'specificity': spec_value,
        'specificity_ci_lower': spec_ci_lower,
        'specificity_ci_upper': spec_ci_upper,
        'balanced_accuracy': balanced_acc,
        'balanced_accuracy_ci_lower': balanced_acc_ci_lower,
        'balanced_accuracy_ci_upper': balanced_acc_ci_upper,
        'overall_accuracy': acc_value,
        'overall_accuracy_ci_lower': acc_ci_lower,
        'overall_accuracy_ci_upper': acc_ci_upper
    }


def run_sensitivity_analysis(patient_final_df, output_dir, borderline_threshold=0.1):
    """
    Run excluded patients sensitivity analysis.
    
    Computes clinical metrics after excluding:
    - Borderline patients (probability near threshold)
    - Hardest patient (lowest correctness)
    - Both combined
    
    Args:
        patient_final_df: DataFrame with patient-level results
        output_dir: Directory to save outputs
        borderline_threshold: Threshold for borderline classification (default 0.1)
    
    Returns:
        dict with sensitivity analysis results
    """
    print("\n" + "="*60)
    print("E) EXCLUDED PATIENTS SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Determine column names
    patient_col = 'fold_patient_id' if 'fold_patient_id' in patient_final_df.columns else 'patient_id'
    
    # Step 1: Identify exclusion groups
    exclusion_groups = identify_exclusion_groups(patient_final_df, borderline_threshold)
    borderline_ids = exclusion_groups['borderline_ids']
    hardest_id = exclusion_groups['hardest_id']
    both_ids = exclusion_groups['both_ids']
    exclusion_details_df = exclusion_groups['exclusion_details_df']
    
    print(f"\nBorderline patients (|prob - 0.5| <= {borderline_threshold}):")
    if len(borderline_ids) > 0:
        for pid in sorted(borderline_ids):
            print(f"  - {pid}")
    else:
        print("  None")
    
    print(f"\nHardest patient (lowest correctness):")
    print(f"  - {hardest_id}")
    
    print(f"\nCombined (both):")
    for pid in sorted(both_ids):
        print(f"  - {pid}")
    
    # Step 2: Create filtered dataframes
    df_baseline = patient_final_df.copy()
    df_no_borderline = patient_final_df[~patient_final_df[patient_col].isin(borderline_ids)].copy()
    df_no_hardest = patient_final_df[patient_final_df[patient_col] != hardest_id].copy()
    df_no_both = patient_final_df[~patient_final_df[patient_col].isin(both_ids)].copy()
    
    # Step 3: Compute metrics for each scenario
    metrics_baseline = compute_metrics_for_subset(df_baseline, 'baseline')
    metrics_no_borderline = compute_metrics_for_subset(df_no_borderline, 'no_borderline')
    metrics_no_hardest = compute_metrics_for_subset(df_no_hardest, 'no_hardest')
    metrics_no_both = compute_metrics_for_subset(df_no_both, 'no_both')
    
    # Check for edge cases
    for scenario_name, df in [('no_borderline', df_no_borderline), 
                               ('no_hardest', df_no_hardest), 
                               ('no_both', df_no_both)]:
        n_cases = df['y_true'].sum()
        n_controls = len(df) - n_cases
        if n_cases == 0:
            print(f"\n⚠ WARNING: Scenario '{scenario_name}' removed all positive cases!")
        if n_controls == 0:
            print(f"\n⚠ WARNING: Scenario '{scenario_name}' removed all negative cases!")
    
    # Step 4: Create comparison DataFrame
    comparison_df = pd.DataFrame([
        metrics_baseline,
        metrics_no_borderline,
        metrics_no_hardest,
        metrics_no_both
    ])
    
    # Step 5: Print comparison summary
    print(f"\n" + "-"*60)
    print("Sensitivity Analysis Results:")
    print("-"*60)
    
    def format_change(baseline_val, new_val):
        if np.isnan(baseline_val) or np.isnan(new_val):
            return "N/A"
        delta = new_val - baseline_val
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.4f}"
    
    baseline_sens = metrics_baseline['sensitivity']
    baseline_spec = metrics_baseline['specificity']
    baseline_bal_acc = metrics_baseline['balanced_accuracy']
    
    for scenario, metrics in [('no_borderline', metrics_no_borderline),
                               ('no_hardest', metrics_no_hardest),
                               ('no_both', metrics_no_both)]:
        print(f"\n{scenario.upper()} (n={metrics['n_patients']} patients):")
        
        sens_change = format_change(baseline_sens, metrics['sensitivity'])
        spec_change = format_change(baseline_spec, metrics['specificity'])
        bal_acc_change = format_change(baseline_bal_acc, metrics['balanced_accuracy'])
        
        sens_str = f"{metrics['sensitivity']:.4f}" if not np.isnan(metrics['sensitivity']) else "N/A"
        spec_str = f"{metrics['specificity']:.4f}" if not np.isnan(metrics['specificity']) else "N/A"
        bal_acc_str = f"{metrics['balanced_accuracy']:.4f}" if not np.isnan(metrics['balanced_accuracy']) else "N/A"
        
        print(f"  Sensitivity:       {baseline_sens:.4f} → {sens_str} (Δ {sens_change})")
        print(f"  Specificity:       {baseline_spec:.4f} → {spec_str} (Δ {spec_change})")
        print(f"  Balanced Accuracy: {baseline_bal_acc:.4f} → {bal_acc_str} (Δ {bal_acc_change})")
    
    # Step 6: Save outputs
    comparison_df.to_csv(os.path.join(output_dir, 'clinical_metrics_excluding_flagged.csv'), index=False)
    print(f"\n  ✓ Saved: clinical_metrics_excluding_flagged.csv")
    
    exclusion_details_df.to_csv(os.path.join(output_dir, 'excluded_patients_used.csv'), index=False)
    print(f"  ✓ Saved: excluded_patients_used.csv")
    
    print(f"\nSee clinical_metrics_excluding_flagged.csv for full results.")
    
    return {
        'comparison_df': comparison_df,
        'exclusion_details_df': exclusion_details_df,
        'borderline_ids': borderline_ids,
        'hardest_id': hardest_id,
        'both_ids': both_ids,
        'metrics_baseline': metrics_baseline,
        'metrics_no_borderline': metrics_no_borderline,
        'metrics_no_hardest': metrics_no_hardest,
        'metrics_no_both': metrics_no_both
    }


# =============================================================================
# README GENERATION
# =============================================================================

def generate_reliability_readme(clinical_metrics_df, margin_summary, borderline_patients, 
                                hardest_report, instability_df, output_dir,
                                sensitivity_results=None):
    """
    Generate README summarizing reliability check findings.
    """
    print("\nGenerating README.md...")
    
    # Extract key values
    sens = clinical_metrics_df[clinical_metrics_df['metric'] == 'Sensitivity'].iloc[0]
    spec = clinical_metrics_df[clinical_metrics_df['metric'] == 'Specificity'].iloc[0]
    bal_acc = clinical_metrics_df[clinical_metrics_df['metric'] == 'Balanced_Accuracy'].iloc[0]
    
    margin_val = margin_summary[margin_summary['metric'] == 'margin']['value'].values[0]
    
    # Top 5 hardest
    top5 = hardest_report.head(5)
    
    # High instability patients
    high_instab = instability_df[instability_df['std_prob'] > 0.1]
    
    readme_content = f"""# LOPO Reliability Checks - Summary Report

## Overview
This report summarizes the reliability analysis of LOPO (Leave-One-Patient-Out) evaluation results.

---

## A) Global Clinical Metrics with 95% Confidence Intervals

| Metric | Value | 95% CI Lower | 95% CI Upper |
|--------|-------|--------------|--------------|
| Sensitivity | {sens['value']:.4f} | {sens['ci_lower']:.4f} | {sens['ci_upper']:.4f} |
| Specificity | {spec['value']:.4f} | {spec['ci_lower']:.4f} | {spec['ci_upper']:.4f} |
| Balanced Accuracy | {bal_acc['value']:.4f} | {bal_acc['ci_lower']:.4f} | {bal_acc['ci_upper']:.4f} |

**Interpretation:**
- Sensitivity measures how well the model identifies Cases (true positive rate)
- Specificity measures how well the model identifies Controls (true negative rate)
- Confidence intervals account for small sample size (n=15 patients)

---

## B) Probability Margin Analysis

| Metric | Value |
|--------|-------|
| Minimum Case Probability | {margin_summary[margin_summary['metric']=='min_case_prob']['value'].values[0]:.4f} |
| Maximum Control Probability | {margin_summary[margin_summary['metric']=='max_control_prob']['value'].values[0]:.4f} |
| **Margin** | **{margin_val:.4f}** |

**Interpretation:**
- Margin > 0: Classes are separable (no probability overlap)
- Margin < 0: Classes overlap (some misclassification expected)
- {"✓ Good class separation" if margin_val > 0 else "⚠ Class overlap detected"}

---

## C) Seed Variance Analysis

**High Instability Patients (std_prob > 0.1):** {len(high_instab)}

{dataframe_to_markdown_simple(high_instab[['patient_id', 'y_true', 'std_prob', 'flip_rate']]) if len(high_instab) > 0 else "None - all patients have stable predictions across seeds."}

**Interpretation:**
- High std_prob indicates inconsistent predictions across training runs
- flip_rate shows how often the predicted class changes between seeds
- Stable predictions (low variance) indicate reliable model behavior

---

## D) Hardest Patients to Classify

**Top 5 Hardest Patients (by lowest correctness):**

{dataframe_to_markdown_simple(top5[['rank', 'patient_id', 'y_true', 'mean_correctness', 'std_prob', 'flip_rate']].round(4))}

**Borderline Patients (probability within 0.1 of threshold):** {len(borderline_patients)}

{dataframe_to_markdown_simple(borderline_patients[['patient_id', 'y_true', 'mean_prob']].round(4)) if len(borderline_patients) > 0 else "None - all patients have confident predictions."}

**Interpretation:**
- Hardest patients may have ambiguous imaging features or labeling issues
- Borderline patients are near the decision threshold and warrant clinical review

---

## Output Files

| File | Description |
|------|-------------|
| `clinical_metrics_with_ci.csv` | Global metrics with 95% Wilson CIs |
| `probability_margin_summary.csv` | Class separation metrics |
| `patient_probability_details.csv` | Per-patient probability details |
| `patient_seed_instability.csv` | Instability metrics per patient |
| `hardest_patient_report.csv` | Complete ranking of patients by difficulty |
| `plot_seed_variance_prob.png` | Probability variance visualization |
| `plot_seed_variance_correctness.png` | Correctness variance visualization |
| `clinical_metrics_excluding_flagged.csv` | Sensitivity analysis metrics |
| `excluded_patients_used.csv` | Details of excluded patients |

---

*Generated by LOPO Reliability Checks Module*
"""
    
    # Add sensitivity analysis section if results provided
    if sensitivity_results is not None:
        sensitivity_section = generate_sensitivity_analysis_readme_section(sensitivity_results)
        # Insert before Output Files section
        readme_content = readme_content.replace(
            "---\n\n## Output Files",
            sensitivity_section + "\n---\n\n## Output Files"
        )
    
    readme_path = os.path.join(output_dir, 'RELIABILITY_README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  ✓ Saved: RELIABILITY_README.md")


def generate_sensitivity_analysis_readme_section(sensitivity_results):
    """
    Generate the sensitivity analysis section for the README.
    
    Args:
        sensitivity_results: dict from run_sensitivity_analysis()
    
    Returns:
        str: Markdown content for sensitivity analysis section
    """
    borderline_ids = sensitivity_results['borderline_ids']
    hardest_id = sensitivity_results['hardest_id']
    both_ids = sensitivity_results['both_ids']
    exclusion_details_df = sensitivity_results['exclusion_details_df']
    
    metrics_baseline = sensitivity_results['metrics_baseline']
    metrics_no_borderline = sensitivity_results['metrics_no_borderline']
    metrics_no_hardest = sensitivity_results['metrics_no_hardest']
    metrics_no_both = sensitivity_results['metrics_no_both']
    
    def format_metric(val):
        return f"{val:.4f}" if not np.isnan(val) else "N/A"
    
    def format_change(baseline_val, new_val):
        if np.isnan(baseline_val) or np.isnan(new_val):
            return "N/A"
        delta = new_val - baseline_val
        sign = "+" if delta >= 0 else ""
        pct = (delta / baseline_val * 100) if baseline_val != 0 else 0
        return f"{sign}{delta:.4f} ({sign}{pct:.1f}%)"
    
    # Build excluded patients list
    borderline_list = ", ".join(sorted([str(p) for p in borderline_ids])) if borderline_ids else "None"
    
    # Build comparison table
    comparison_rows = []
    for scenario, metrics in [('baseline', metrics_baseline),
                               ('no_borderline', metrics_no_borderline),
                               ('no_hardest', metrics_no_hardest),
                               ('no_both', metrics_no_both)]:
        comparison_rows.append(
            f"| {scenario} | {metrics['n_patients']} | {metrics['n_cases']} | {metrics['n_controls']} | "
            f"{format_metric(metrics['sensitivity'])} | {format_metric(metrics['specificity'])} | "
            f"{format_metric(metrics['balanced_accuracy'])} |"
        )
    
    comparison_table = "\n".join(comparison_rows)
    
    # Build change summary
    sens_change_borderline = format_change(metrics_baseline['sensitivity'], metrics_no_borderline['sensitivity'])
    spec_change_borderline = format_change(metrics_baseline['specificity'], metrics_no_borderline['specificity'])
    
    sens_change_hardest = format_change(metrics_baseline['sensitivity'], metrics_no_hardest['sensitivity'])
    spec_change_hardest = format_change(metrics_baseline['specificity'], metrics_no_hardest['specificity'])
    
    sens_change_both = format_change(metrics_baseline['sensitivity'], metrics_no_both['sensitivity'])
    spec_change_both = format_change(metrics_baseline['specificity'], metrics_no_both['specificity'])
    
    # Clinical interpretation
    interpretations = []
    
    # Check sensitivity changes
    if not np.isnan(metrics_no_borderline['sensitivity']) and not np.isnan(metrics_baseline['sensitivity']):
        sens_delta = metrics_no_borderline['sensitivity'] - metrics_baseline['sensitivity']
        if sens_delta > 0.05:
            interpretations.append(f"- Excluding borderline patients improved sensitivity by {sens_delta:.1%}")
        elif sens_delta < -0.05:
            interpretations.append(f"- Excluding borderline patients decreased sensitivity by {abs(sens_delta):.1%}")
    
    if not np.isnan(metrics_no_borderline['specificity']) and not np.isnan(metrics_baseline['specificity']):
        spec_delta = metrics_no_borderline['specificity'] - metrics_baseline['specificity']
        if spec_delta > 0.05:
            interpretations.append(f"- Excluding borderline patients improved specificity by {spec_delta:.1%}")
        elif spec_delta < -0.05:
            interpretations.append(f"- Excluding borderline patients decreased specificity by {abs(spec_delta):.1%}")
    
    if not np.isnan(metrics_no_both['balanced_accuracy']) and not np.isnan(metrics_baseline['balanced_accuracy']):
        bal_delta = metrics_no_both['balanced_accuracy'] - metrics_baseline['balanced_accuracy']
        if bal_delta > 0.05:
            interpretations.append(f"- Excluding all flagged patients improved balanced accuracy by {bal_delta:.1%}")
    
    if not interpretations:
        interpretations.append("- Excluding flagged patients had minimal impact on overall metrics")
    
    interpretation_text = "\n".join(interpretations)
    
    section = f"""
---

## E) Sensitivity Analysis (Excluding Flagged Patients)

This analysis examines how clinical metrics change when problematic patients are excluded.

### Excluded Patients

**Borderline Patients** (|prob - 0.5| ≤ 0.1): {borderline_list}

**Hardest Patient** (lowest correctness): {hardest_id}

**Combined**: {", ".join(sorted([str(p) for p in both_ids]))}

### Metrics Comparison

| Scenario | n_patients | n_cases | n_controls | Sensitivity | Specificity | Balanced Accuracy |
|----------|------------|---------|------------|-------------|-------------|-------------------|
{comparison_table}

### Change Summary

**Excluding Borderline Patients:**
- Sensitivity: {sens_change_borderline}
- Specificity: {spec_change_borderline}

**Excluding Hardest Patient:**
- Sensitivity: {sens_change_hardest}
- Specificity: {spec_change_hardest}

**Excluding Both:**
- Sensitivity: {sens_change_both}
- Specificity: {spec_change_both}

### Clinical Interpretation

{interpretation_text}

**Note:** This sensitivity analysis helps identify whether certain patients disproportionately affect model performance, 
potentially indicating labeling issues, atypical presentations, or cases warranting clinical review.

"""
    return section


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_reliability_checks():
    """
    Run all LOPO reliability checks.
    
    Reads input files from outputs/lopo/ and saves results to outputs/lopo/analysis/
    """
    print("\n" + "="*70)
    print("    LOPO RELIABILITY CHECKS AT PATIENT LEVEL")
    print("="*70)
    
    # Setup paths
    lopo_dir = LOPO_DIR
    analysis_dir = lopo_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load Input Data ---
    print("\nLoading input files...")
    
    patient_final_path = lopo_dir / "lopo_per_patient_final.xlsx"
    patient_seed_path = lopo_dir / "lopo_per_patient_seed.xlsx"
    
    if not patient_final_path.exists():
        print(f"ERROR: File not found: {patient_final_path}")
        print("Please run LOPO evaluation first: python run_lopo_evaluation.py")
        return
    
    if not patient_seed_path.exists():
        print(f"ERROR: File not found: {patient_seed_path}")
        print("Please run LOPO evaluation first: python run_lopo_evaluation.py")
        return
    
    patient_final_df = pd.read_excel(patient_final_path)
    patient_seed_df = pd.read_excel(patient_seed_path)
    
    print(f"  ✓ Loaded lopo_per_patient_final.xlsx ({len(patient_final_df)} patients)")
    print(f"  ✓ Loaded lopo_per_patient_seed.xlsx ({len(patient_seed_df)} records)")
    
    # --- A) Clinical Metrics with CI ---
    clinical_metrics_df = compute_clinical_metrics_with_ci(patient_final_df)
    clinical_metrics_df.to_csv(analysis_dir / "clinical_metrics_with_ci.csv", index=False)
    print(f"\n  ✓ Saved: clinical_metrics_with_ci.csv")
    
    # --- B) Probability Margin Analysis ---
    margin_summary, patient_details = compute_probability_margin(patient_final_df)
    margin_summary.to_csv(analysis_dir / "probability_margin_summary.csv", index=False)
    patient_details.to_csv(analysis_dir / "patient_probability_details.csv", index=False)
    print(f"\n  ✓ Saved: probability_margin_summary.csv")
    print(f"  ✓ Saved: patient_probability_details.csv")
    
    # --- C) Seed Variance Analysis ---
    instability_df = compute_seed_instability(patient_seed_df)
    instability_df.to_csv(analysis_dir / "patient_seed_instability.csv", index=False)
    print(f"\n  ✓ Saved: patient_seed_instability.csv")
    
    # Create variance plots
    plot_seed_variance(instability_df, analysis_dir)
    
    # --- D) Hardest Patient Identification ---
    hardest_report, borderline_patients = identify_hardest_patients(instability_df, patient_final_df)
    hardest_report.to_csv(analysis_dir / "hardest_patient_report.csv", index=False)
    print(f"\n  ✓ Saved: hardest_patient_report.csv")
    
    # --- E) Excluded Patients Sensitivity Analysis ---
    sensitivity_results = run_sensitivity_analysis(patient_final_df, analysis_dir)
    
    # --- Generate README ---
    generate_reliability_readme(
        clinical_metrics_df, margin_summary, borderline_patients,
        hardest_report, instability_df, analysis_dir,
        sensitivity_results=sensitivity_results
    )
    
    # --- Final Summary ---
    print("\n" + "="*70)
    print("    RELIABILITY CHECKS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {analysis_dir}")
    print("\nKey findings:")
    
    bal_acc = clinical_metrics_df[clinical_metrics_df['metric'] == 'Balanced_Accuracy'].iloc[0]
    print(f"  - Balanced Accuracy: {bal_acc['value']:.4f} [{bal_acc['ci_lower']:.4f}, {bal_acc['ci_upper']:.4f}]")
    
    margin_val = margin_summary[margin_summary['metric'] == 'margin']['value'].values[0]
    print(f"  - Class Margin: {margin_val:.4f} ({'separable' if margin_val > 0 else 'overlap'})")
    
    high_instab = instability_df[instability_df['std_prob'] > 0.1]
    print(f"  - High-instability patients: {len(high_instab)}")
    
    print(f"  - Borderline patients: {len(borderline_patients)}")
    print(f"  - Sensitivity analysis: {len(sensitivity_results['both_ids'])} patients flagged")
    
    return {
        'clinical_metrics': clinical_metrics_df,
        'margin_summary': margin_summary,
        'patient_details': patient_details,
        'instability': instability_df,
        'hardest_report': hardest_report,
        'sensitivity_analysis': sensitivity_results
    }


if __name__ == "__main__":
    run_reliability_checks()
