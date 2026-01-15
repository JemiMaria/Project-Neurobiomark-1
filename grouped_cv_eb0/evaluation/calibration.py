"""
Calibration analysis for probability predictions.

Computes:
- Brier score (lower is better)
- Reliability diagram (calibration curve)
- Expected Calibration Error (ECE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_DIR


def compute_brier_score(y_true, y_prob):
    """
    Compute Brier score (mean squared error of probabilities).
    
    Brier score = mean((prob - y_true)^2)
    - Range: [0, 1]
    - Lower is better
    - 0.25 = random guessing
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        float: Brier score
    """
    return np.mean((y_prob - y_true) ** 2)


def compute_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        float: ECE
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def compute_calibration_metrics(patient_df, n_bins=10):
    """
    Compute all calibration metrics.
    
    Args:
        patient_df: DataFrame with [y_true, prob]
        n_bins: Number of bins for calibration curve
        
    Returns:
        dict: Calibration metrics
    """
    y_true = patient_df['y_true'].values
    y_prob = patient_df['prob'].values
    
    brier = compute_brier_score(y_true, y_prob)
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins)
    
    # Calibration curve data
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    except ValueError:
        prob_true, prob_pred = np.array([]), np.array([])
    
    return {
        'brier_score': brier,
        'ece': ece,
        'n_samples': len(y_true),
        'calibration_curve_true': prob_true.tolist(),
        'calibration_curve_pred': prob_pred.tolist()
    }


def plot_reliability_diagram(all_patient_dfs, output_path=None, n_bins=10):
    """
    Plot reliability diagram (calibration curve).
    
    Shows:
    - Perfect calibration line (diagonal)
    - Actual calibration curve
    - Histogram of predictions
    
    Args:
        all_patient_dfs: List of patient DataFrames (one per fold)
        output_path: Path to save plot
        n_bins: Number of bins
    """
    if output_path is None:
        output_path = EVAL_DIR / "reliability_diagram.png"
    
    # Combine all predictions
    combined_df = pd.concat(all_patient_dfs, ignore_index=True)
    y_true = combined_df['y_true'].values
    y_prob = combined_df['prob'].values
    
    # Compute calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    except ValueError:
        print("  ⚠ Could not compute calibration curve")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
    ax1.plot(prob_pred, prob_true, 's-', color='tab:blue', label='Model', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Reliability Diagram (Calibration Curve)', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Compute metrics for display
    brier = compute_brier_score(y_true, y_prob)
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins)
    ax1.text(0.05, 0.95, f'Brier Score: {brier:.4f}\nECE: {ece:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), color='tab:blue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def save_calibration_results(fold_calibration_list, output_path=None):
    """
    Save calibration results to CSV.
    
    Args:
        fold_calibration_list: List of calibration dicts per fold
        output_path: Path to save CSV
    """
    if output_path is None:
        output_path = EVAL_DIR / "calibration_brier.csv"
    
    records = []
    for fold_id, cal in enumerate(fold_calibration_list):
        records.append({
            'fold_id': fold_id,
            'brier_score': cal['brier_score'],
            'ece': cal['ece'],
            'n_samples': cal['n_samples']
        })
    
    # Add summary row
    brier_vals = [c['brier_score'] for c in fold_calibration_list]
    ece_vals = [c['ece'] for c in fold_calibration_list]
    
    records.append({
        'fold_id': 'mean',
        'brier_score': np.mean(brier_vals),
        'ece': np.mean(ece_vals),
        'n_samples': sum(c['n_samples'] for c in fold_calibration_list)
    })
    
    records.append({
        'fold_id': 'std',
        'brier_score': np.std(brier_vals),
        'ece': np.std(ece_vals),
        'n_samples': 0
    })
    
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
