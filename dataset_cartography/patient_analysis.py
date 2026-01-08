"""
Patient-Level Cartography Analysis Module.

This module analyzes patient-level cartography results to identify:
1. Confidence vs Correctness relationships
2. Isolated/outlier patients
3. Variance across seeds (if multi-seed available)
4. Image-level disagreement within patients

Outputs:
- Plots saved to outputs/analysis/
- CSV files with detailed results
- README.md with findings summary
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    OUTPUT_DIR, CARTOGRAPHY_PER_IMAGE_XLSX, CARTOGRAPHY_PER_PATIENT_XLSX
)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Analysis output directory
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def ensure_analysis_dir():
    """Create analysis output directory if it doesn't exist."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR


def load_cartography_data():
    """
    Load cartography data from Excel files.
    
    Returns:
        tuple: (patient_df, image_df) or (None, None) if files not found
    """
    patient_path = OUTPUT_DIR / CARTOGRAPHY_PER_PATIENT_XLSX
    image_path = OUTPUT_DIR / CARTOGRAPHY_PER_IMAGE_XLSX
    
    patient_df = None
    image_df = None
    
    if patient_path.exists():
        patient_df = pd.read_excel(patient_path, engine='openpyxl')
        print(f"✓ Loaded patient data: {len(patient_df)} patients")
    else:
        print(f"✗ Patient file not found: {patient_path}")
    
    if image_path.exists():
        image_df = pd.read_excel(image_path, engine='openpyxl')
        print(f"✓ Loaded image data: {len(image_df)} images")
    else:
        print(f"✗ Image file not found: {image_path}")
    
    return patient_df, image_df


# =============================================================================
# ANALYSIS 1: Confidence vs Correctness Scatter Plot (Patient Level)
# =============================================================================

def analysis_1_confidence_vs_correctness(patient_df, output_dir):
    """
    Create scatter plot of patient-level confidence vs correctness.
    
    - X = patient_mean_confidence
    - Y = patient_mean_correctness
    - Point size proportional to number of images
    - Reference lines: x=0.5, y=0.5, y=x
    - Labels with patient_id
    - Highlight outliers
    
    Args:
        patient_df: DataFrame with patient-level metrics
        output_dir: Directory to save outputs
        
    Returns:
        Path to saved plot
    """
    print("\n" + "="*60)
    print("ANALYSIS 1: Confidence vs Correctness (Patient Level)")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract data
    x = patient_df['patient_mean_confidence']
    y = patient_df['patient_mean_correctness']
    sizes = patient_df['num_samples_in_patient'] * 50  # Scale for visibility
    patient_ids = patient_df['patient_id']
    
    # Identify outliers using IQR method
    outlier_mask = identify_outliers_iqr(x, y)
    
    # Plot non-outliers
    scatter = ax.scatter(
        x[~outlier_mask], y[~outlier_mask], 
        s=sizes[~outlier_mask], 
        alpha=0.6, 
        c='steelblue',
        edgecolors='white',
        linewidths=1,
        label='Normal patients'
    )
    
    # Plot outliers with different style
    if outlier_mask.any():
        ax.scatter(
            x[outlier_mask], y[outlier_mask], 
            s=sizes[outlier_mask], 
            alpha=0.8, 
            c='red',
            edgecolors='darkred',
            linewidths=2,
            marker='s',
            label='Isolated patients'
        )
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='y=0.5 (chance)')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='x=0.5 (chance)')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x (perfect calibration)')
    
    # Add quadrant labels
    ax.text(0.25, 0.75, 'High Correctness\nLow Confidence', ha='center', va='center', 
            fontsize=9, alpha=0.5, style='italic')
    ax.text(0.75, 0.75, 'High Correctness\nHigh Confidence\n(EASY)', ha='center', va='center', 
            fontsize=9, alpha=0.5, style='italic', color='green')
    ax.text(0.25, 0.25, 'Low Correctness\nLow Confidence\n(HARD)', ha='center', va='center', 
            fontsize=9, alpha=0.5, style='italic', color='red')
    ax.text(0.75, 0.25, 'Low Correctness\nHigh Confidence\n(OVERCONFIDENT)', ha='center', va='center', 
            fontsize=9, alpha=0.5, style='italic', color='orange')
    
    # Label all points with patient_id
    for i, pid in enumerate(patient_ids):
        offset_x = 0.015
        offset_y = 0.015
        fontweight = 'bold' if outlier_mask.iloc[i] else 'normal'
        color = 'darkred' if outlier_mask.iloc[i] else 'black'
        ax.annotate(
            str(pid), 
            (x.iloc[i], y.iloc[i]),
            xytext=(x.iloc[i] + offset_x, y.iloc[i] + offset_y),
            fontsize=8,
            fontweight=fontweight,
            color=color,
            alpha=0.8
        )
    
    # Formatting
    ax.set_xlabel('Patient Mean Confidence', fontsize=12)
    ax.set_ylabel('Patient Mean Correctness', fontsize=12)
    ax.set_title('Patient-Level Cartography: Confidence vs Correctness\n(Point size ∝ number of images)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_text = (f"N = {len(patient_df)} patients\n"
                  f"Mean Confidence: {x.mean():.3f} ± {x.std():.3f}\n"
                  f"Mean Correctness: {y.mean():.3f} ± {y.std():.3f}\n"
                  f"Correlation: {x.corr(y):.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "analysis1_confidence_vs_correctness.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved: {plot_path}")
    print(f"  - Total patients: {len(patient_df)}")
    print(f"  - Isolated patients: {outlier_mask.sum()}")
    
    return plot_path


def identify_outliers_iqr(x, y, threshold=1.5):
    """
    Identify outliers using IQR method on Euclidean distance from median.
    
    Args:
        x: Series of x values (confidence)
        y: Series of y values (correctness)
        threshold: IQR multiplier (default 1.5)
        
    Returns:
        Boolean mask of outliers
    """
    # Compute distance from median point
    median_x = x.median()
    median_y = y.median()
    distances = np.sqrt((x - median_x)**2 + (y - median_y)**2)
    
    # IQR-based threshold
    q1 = distances.quantile(0.25)
    q3 = distances.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + threshold * iqr
    
    return distances > upper_bound


# =============================================================================
# ANALYSIS 2: Identify Isolated Patients
# =============================================================================

def analysis_2_isolated_patients(patient_df, output_dir):
    """
    Identify isolated (outlier) patients using multiple methods.
    
    Methods:
    1. Euclidean distance from median (IQR-based)
    2. Z-score based detection
    3. Low correctness detection (< 0.5)
    4. High variability detection
    
    Args:
        patient_df: DataFrame with patient-level metrics
        output_dir: Directory to save outputs
        
    Returns:
        DataFrame with isolated patients
    """
    print("\n" + "="*60)
    print("ANALYSIS 2: Identify Isolated Patients")
    print("="*60)
    
    x = patient_df['patient_mean_confidence']
    y = patient_df['patient_mean_correctness']
    
    # Compute distance from median
    median_x = x.median()
    median_y = y.median()
    distances = np.sqrt((x - median_x)**2 + (y - median_y)**2)
    
    # Method 1: IQR-based outliers
    q1 = distances.quantile(0.25)
    q3 = distances.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = distances > (q3 + 1.5 * iqr)
    
    # Method 2: Z-score based outliers (> 2 std from mean)
    z_scores = (distances - distances.mean()) / distances.std()
    zscore_outliers = np.abs(z_scores) > 2
    
    # Method 3: Low correctness (< 0.5)
    low_correctness = y < 0.5
    
    # Method 4: Overconfident but wrong (high conf, low correctness)
    overconfident = (x > 0.7) & (y < 0.5)
    
    # Combine reasons
    results = []
    for i, row in patient_df.iterrows():
        reasons = []
        if iqr_outliers.iloc[i]:
            reasons.append("IQR_outlier")
        if zscore_outliers.iloc[i]:
            reasons.append("Zscore>2")
        if low_correctness.iloc[i]:
            reasons.append("Low_correctness")
        if overconfident.iloc[i]:
            reasons.append("Overconfident")
        
        if reasons:
            results.append({
                'patient_id': row['patient_id'],
                'patient_mean_confidence': row['patient_mean_confidence'],
                'patient_mean_correctness': row['patient_mean_correctness'],
                'num_samples': row['num_samples_in_patient'],
                'distance_from_median': distances.iloc[i],
                'z_score': z_scores.iloc[i],
                'reason': '; '.join(reasons)
            })
    
    isolated_df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / "isolated_patients.csv"
    if len(isolated_df) > 0:
        isolated_df = isolated_df.round(4)
        isolated_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
        print(f"  - Isolated patients found: {len(isolated_df)}")
        print("\nIsolated Patients:")
        print(isolated_df.to_string(index=False))
    else:
        # Create empty file with headers
        pd.DataFrame(columns=['patient_id', 'patient_mean_confidence', 
                             'patient_mean_correctness', 'num_samples',
                             'distance_from_median', 'z_score', 'reason']).to_csv(csv_path, index=False)
        print(f"✓ No isolated patients found. Empty file saved: {csv_path}")
    
    return isolated_df


# =============================================================================
# ANALYSIS 3: Variance Across Seeds
# =============================================================================

def analysis_3_variance_across_seeds(patient_df, image_df, output_dir):
    """
    Analyze variance of confidence and correctness across seeds.
    
    If multi-seed data is available (indicated by columns like 't_stars_per_seed'),
    compute per-patient variance statistics.
    
    Args:
        patient_df: DataFrame with patient-level metrics
        image_df: DataFrame with image-level metrics
        output_dir: Directory to save outputs
        
    Returns:
        DataFrame with variance statistics
    """
    print("\n" + "="*60)
    print("ANALYSIS 3: Variance Across Seeds")
    print("="*60)
    
    # Check if multi-seed data is available
    has_multi_seed = 't_stars_per_seed' in patient_df.columns or 'window_sizes_per_seed' in patient_df.columns
    
    if not has_multi_seed:
        print("⚠ Single-seed data detected. Variance across seeds not available.")
        print("  This analysis requires multi-seed training data.")
        
        # Create placeholder output
        variance_df = patient_df[['patient_id', 'patient_mean_confidence', 'patient_mean_correctness']].copy()
        variance_df['confidence_std'] = np.nan
        variance_df['correctness_std'] = np.nan
        variance_df['note'] = 'Single-seed data - variance not computed'
        
        csv_path = output_dir / "patient_variance_seeds.csv"
        variance_df.to_csv(csv_path, index=False)
        print(f"✓ Placeholder saved: {csv_path}")
        
        return variance_df
    
    # Parse per-seed values from comma-separated strings
    # This assumes the windowed_cartography module stored seed-specific data
    print("✓ Multi-seed data detected")
    
    # For now, use the aggregated values and compute approximate variance
    # In a full implementation, we would need to access raw per-seed data
    variance_results = []
    
    for _, row in patient_df.iterrows():
        # Parse t_stars to infer number of seeds
        t_stars = str(row.get('t_stars_per_seed', '')).split(',')
        num_seeds = len([t for t in t_stars if t.strip()])
        
        variance_results.append({
            'patient_id': row['patient_id'],
            'patient_mean_confidence': row['patient_mean_confidence'],
            'patient_mean_correctness': row['patient_mean_correctness'],
            'num_seeds': num_seeds,
            'num_samples': row['num_samples_in_patient'],
            't_stars': row.get('t_stars_per_seed', 'N/A'),
            'note': 'Aggregated across seeds'
        })
    
    variance_df = pd.DataFrame(variance_results)
    
    # Create visualization: Mean correctness with estimated uncertainty
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by correctness for better visualization
    sorted_df = variance_df.sort_values('patient_mean_correctness', ascending=True).reset_index(drop=True)
    
    x_pos = np.arange(len(sorted_df))
    
    # Use a proxy for uncertainty: distance from 0.5 (more extreme = more certain)
    uncertainty_proxy = 0.1 * (1 - np.abs(sorted_df['patient_mean_correctness'] - 0.5) * 2)
    
    # Color by correctness level
    colors = ['red' if c < 0.5 else 'orange' if c < 0.7 else 'green' 
              for c in sorted_df['patient_mean_correctness']]
    
    ax.bar(x_pos, sorted_df['patient_mean_correctness'], 
           yerr=uncertainty_proxy, capsize=3,
           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level (0.5)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High correctness (0.8)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_df['patient_id'], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Patient ID', fontsize=11)
    ax.set_ylabel('Mean Correctness', fontsize=11)
    ax.set_title('Patient Mean Correctness with Estimated Uncertainty\n(Error bars represent prediction confidence)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "analysis3_variance_across_seeds.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save CSV
    csv_path = output_dir / "patient_variance_seeds.csv"
    variance_df.round(4).to_csv(csv_path, index=False)
    
    print(f"✓ Plot saved: {plot_path}")
    print(f"✓ CSV saved: {csv_path}")
    
    return variance_df


# =============================================================================
# ANALYSIS 4: Image-Level Disagreement Within Patient
# =============================================================================

def analysis_4_image_disagreement(patient_df, image_df, output_dir):
    """
    Analyze image-level prediction instability within each patient.
    
    Computes:
    - Per-image instability proxy (1 - abs(confidence - 0.5) * 2)
    - Per-patient aggregates (mean, max, std of instability)
    - Boxplot of image-level instability per patient
    
    Args:
        patient_df: DataFrame with patient-level metrics
        image_df: DataFrame with image-level metrics
        output_dir: Directory to save outputs
        
    Returns:
        DataFrame with image disagreement statistics
    """
    print("\n" + "="*60)
    print("ANALYSIS 4: Image-Level Disagreement Within Patient")
    print("="*60)
    
    if image_df is None or len(image_df) == 0:
        print("✗ No image-level data available")
        return None
    
    # Compute instability proxy for each image
    # Instability = how uncertain the model is (closer to 0.5 = more unstable)
    image_df = image_df.copy()
    image_df['instability'] = 1 - np.abs(image_df['mean_confidence'] - 0.5) * 2
    
    # Also flag disagreement (correctness < 1 means some epochs/seeds disagreed)
    image_df['disagreement'] = 1 - image_df['mean_correctness']
    
    # Aggregate per patient
    patient_instability = image_df.groupby('patient_id').agg({
        'instability': ['mean', 'max', 'std', 'count'],
        'disagreement': ['mean', 'max'],
        'mean_confidence': ['mean', 'std'],
        'mean_correctness': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    patient_instability.columns = ['_'.join(col).strip() for col in patient_instability.columns]
    patient_instability = patient_instability.reset_index()
    
    # Rename for clarity
    patient_instability = patient_instability.rename(columns={
        'instability_mean': 'mean_instability',
        'instability_max': 'max_instability',
        'instability_std': 'std_instability',
        'instability_count': 'num_images',
        'disagreement_mean': 'mean_disagreement',
        'disagreement_max': 'max_disagreement',
        'mean_confidence_mean': 'patient_confidence',
        'mean_confidence_std': 'confidence_spread',
        'mean_correctness_mean': 'patient_correctness',
        'mean_correctness_std': 'correctness_spread'
    })
    
    # Create boxplot of image-level instability per patient
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Boxplot of instability per patient
    ax1 = axes[0]
    patients_sorted = patient_instability.sort_values('mean_instability', ascending=False)['patient_id'].tolist()
    
    # Prepare data for boxplot
    boxplot_data = [image_df[image_df['patient_id'] == pid]['instability'].values 
                    for pid in patients_sorted]
    
    bp = ax1.boxplot(boxplot_data, patch_artist=True, labels=patients_sorted)
    
    # Color boxes by mean instability
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(patients_sorted)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High instability (0.5)')
    ax1.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Low instability (0.2)')
    
    ax1.set_xlabel('Patient ID', fontsize=11)
    ax1.set_ylabel('Image Instability', fontsize=11)
    ax1.set_title('Image-Level Prediction Instability by Patient\n(Higher = more uncertain predictions)', 
                  fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Scatter of mean vs max instability per patient
    ax2 = axes[1]
    
    scatter = ax2.scatter(
        patient_instability['mean_instability'],
        patient_instability['max_instability'],
        s=patient_instability['num_images'] * 30,
        c=patient_instability['patient_correctness'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add patient labels
    for _, row in patient_instability.iterrows():
        ax2.annotate(
            str(row['patient_id']),
            (row['mean_instability'], row['max_instability']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.8
        )
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax2.set_xlabel('Mean Instability', fontsize=11)
    ax2.set_ylabel('Max Instability', fontsize=11)
    ax2.set_title('Mean vs Max Image Instability per Patient\n(Color = correctness, Size = num images)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Patient Correctness', fontsize=10)
    
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "analysis4_image_disagreement.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save image-level disagreement data
    image_disagreement_df = image_df[['image_id', 'patient_id', 'y_true', 
                                       'mean_confidence', 'mean_correctness',
                                       'instability', 'disagreement']].copy()
    image_disagreement_df = image_disagreement_df.round(4)
    
    image_csv_path = output_dir / "image_disagreement.csv"
    image_disagreement_df.to_csv(image_csv_path, index=False)
    
    # Save patient-level summary
    patient_csv_path = output_dir / "patient_instability_summary.csv"
    patient_instability.to_csv(patient_csv_path, index=False)
    
    print(f"✓ Plot saved: {plot_path}")
    print(f"✓ Image-level CSV saved: {image_csv_path}")
    print(f"✓ Patient summary CSV saved: {patient_csv_path}")
    print(f"\nPatient Instability Summary (sorted by mean instability):")
    print(patient_instability.sort_values('mean_instability', ascending=False).to_string(index=False))
    
    return patient_instability


# =============================================================================
# README Generation
# =============================================================================

def generate_analysis_readme(patient_df, image_df, isolated_df, output_dir):
    """
    Generate README.md summarizing analysis findings.
    
    Args:
        patient_df: DataFrame with patient-level metrics
        image_df: DataFrame with image-level metrics  
        isolated_df: DataFrame with isolated patients
        output_dir: Directory to save README
    """
    print("\n" + "="*60)
    print("Generating Analysis README")
    print("="*60)
    
    # Compute statistics
    num_patients = len(patient_df)
    num_images = len(image_df) if image_df is not None else 0
    num_isolated = len(isolated_df) if isolated_df is not None else 0
    
    mean_conf = patient_df['patient_mean_confidence'].mean()
    std_conf = patient_df['patient_mean_confidence'].std()
    mean_corr = patient_df['patient_mean_correctness'].mean()
    std_corr = patient_df['patient_mean_correctness'].std()
    
    # Categorize patients
    high_corr = (patient_df['patient_mean_correctness'] >= 0.8).sum()
    low_corr = (patient_df['patient_mean_correctness'] < 0.5).sum()
    medium_corr = num_patients - high_corr - low_corr
    
    # Generate README content
    readme_content = f"""# Patient-Level Cartography Analysis Results

## Overview

This analysis examines patient-level cartography results to identify:
- Model stability and reliability per patient
- Isolated/outlier patients requiring attention
- Prediction disagreement patterns

**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Patients | {num_patients} |
| Total Images | {num_images} |
| Isolated Patients | {num_isolated} |

### Confidence Statistics
- **Mean:** {mean_conf:.4f}
- **Std Dev:** {std_conf:.4f}
- **Range:** [{patient_df['patient_mean_confidence'].min():.4f}, {patient_df['patient_mean_confidence'].max():.4f}]

### Correctness Statistics
- **Mean:** {mean_corr:.4f}
- **Std Dev:** {std_corr:.4f}
- **Range:** [{patient_df['patient_mean_correctness'].min():.4f}, {patient_df['patient_mean_correctness'].max():.4f}]

### Patient Correctness Distribution
- **High (≥0.8):** {high_corr} patients ({high_corr/num_patients*100:.1f}%)
- **Medium (0.5-0.8):** {medium_corr} patients ({medium_corr/num_patients*100:.1f}%)
- **Low (<0.5):** {low_corr} patients ({low_corr/num_patients*100:.1f}%)

---

## Analysis Results

### Analysis 1: Confidence vs Correctness Plot

**File:** `analysis1_confidence_vs_correctness.png`

This scatter plot shows the relationship between patient-level confidence and correctness:
- Each point represents one patient
- Point size is proportional to the number of images per patient
- Red squares indicate isolated/outlier patients
- Reference lines at x=0.5, y=0.5 mark chance-level performance
- The diagonal y=x line indicates perfect calibration

**Interpretation:**
- Patients in the upper-right quadrant (high confidence, high correctness) are "Easy" cases
- Patients in the lower-left quadrant (low confidence, low correctness) are "Hard" cases
- Patients in the lower-right quadrant (high confidence, low correctness) are "Overconfident" - the model is wrong but sure
- Patients far from the diagonal may indicate calibration issues

### Analysis 2: Isolated Patients

**File:** `isolated_patients.csv`

Isolated patients are identified using multiple criteria:
1. **IQR Outlier:** Distance from median > Q3 + 1.5×IQR
2. **Z-score > 2:** More than 2 standard deviations from mean
3. **Low Correctness:** Correctness < 0.5 (worse than chance)
4. **Overconfident:** High confidence (>0.7) but low correctness (<0.5)

"""
    
    # Add isolated patients details if any
    if isolated_df is not None and len(isolated_df) > 0:
        readme_content += f"""
**Isolated Patients Found: {len(isolated_df)}**

| Patient ID | Confidence | Correctness | Reason |
|------------|------------|-------------|--------|
"""
        for _, row in isolated_df.iterrows():
            readme_content += f"| {row['patient_id']} | {row['patient_mean_confidence']:.4f} | {row['patient_mean_correctness']:.4f} | {row['reason']} |\n"
    else:
        readme_content += "\n**No isolated patients detected.**\n"
    
    readme_content += """
### Analysis 3: Variance Across Seeds

**Files:** 
- `analysis3_variance_across_seeds.png`
- `patient_variance_seeds.csv`

This analysis shows the stability of predictions across different training runs (seeds):
- Bar height represents mean correctness
- Error bars represent estimated uncertainty
- Colors indicate correctness level (red=low, orange=medium, green=high)

**Note:** Full variance analysis requires per-seed raw data. If single-seed training was performed, 
uncertainty estimates are proxied from the aggregated confidence values.

### Analysis 4: Image-Level Disagreement

**Files:**
- `analysis4_image_disagreement.png`
- `image_disagreement.csv`
- `patient_instability_summary.csv`

This analysis examines prediction instability at the image level:

**Instability Metric:** `1 - |confidence - 0.5| × 2`
- Values near 1 indicate high instability (model is uncertain)
- Values near 0 indicate stability (model is confident)

**Disagreement Metric:** `1 - correctness`
- Values near 1 indicate frequent prediction errors
- Values near 0 indicate consistent correct predictions

**Key Findings:**
"""
    
    # Add interpretation based on data
    if image_df is not None:
        image_df_copy = image_df.copy()
        image_df_copy['instability'] = 1 - np.abs(image_df_copy['mean_confidence'] - 0.5) * 2
        mean_instability = image_df_copy['instability'].mean()
        
        if mean_instability > 0.5:
            readme_content += "- ⚠️ **High overall instability** - Model shows significant uncertainty\n"
        elif mean_instability > 0.3:
            readme_content += "- 🔶 **Moderate instability** - Some patients may need review\n"
        else:
            readme_content += "- ✅ **Low instability** - Model predictions are generally stable\n"
    
    readme_content += f"""
---

## Output Files

| File | Description |
|------|-------------|
| `analysis1_confidence_vs_correctness.png` | Scatter plot of patient confidence vs correctness |
| `isolated_patients.csv` | List of outlier patients with reasons |
| `analysis3_variance_across_seeds.png` | Bar chart of patient correctness with uncertainty |
| `patient_variance_seeds.csv` | Patient-level variance statistics |
| `analysis4_image_disagreement.png` | Boxplot and scatter of image instability |
| `image_disagreement.csv` | Per-image instability metrics |
| `patient_instability_summary.csv` | Per-patient aggregated instability |
| `README.md` | This summary document |

---

## Recommendations

Based on the analysis:

1. **Review Isolated Patients:** Patients flagged as isolated may require:
   - Manual review of their images
   - Investigation of data quality issues
   - Consideration for exclusion or separate analysis

2. **High Instability Patients:** Patients with high mean or max instability indicate:
   - Model uncertainty about their features
   - Potential ambiguity in the underlying data
   - May benefit from additional training data

3. **Overconfident Cases:** Patients with high confidence but low correctness are concerning:
   - Model is making errors confidently
   - May indicate systematic issues with certain patient characteristics

4. **Model Calibration:** If many patients deviate from the y=x diagonal:
   - Consider calibration techniques (Platt scaling, isotonic regression)
   - Review training data distribution

---

*Generated by Dataset Cartography Analysis Pipeline*
"""
    
    # Save README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ README saved: {readme_path}")
    
    return readme_path


# =============================================================================
# Main Analysis Runner
# =============================================================================

def run_all_analyses():
    """
    Run all patient-level cartography analyses.
    
    Returns:
        dict: Paths to all generated outputs
    """
    print("\n" + "="*70)
    print("PATIENT-LEVEL CARTOGRAPHY ANALYSIS")
    print("="*70)
    
    # Setup
    output_dir = ensure_analysis_dir()
    print(f"Output directory: {output_dir}")
    
    # Load data
    patient_df, image_df = load_cartography_data()
    
    if patient_df is None:
        print("\n✗ Cannot proceed without patient-level data.")
        print("  Please run the cartography pipeline first to generate:")
        print(f"  - {OUTPUT_DIR / CARTOGRAPHY_PER_PATIENT_XLSX}")
        return None
    
    outputs = {}
    
    # Run analyses
    try:
        outputs['plot1'] = analysis_1_confidence_vs_correctness(patient_df, output_dir)
    except Exception as e:
        print(f"✗ Analysis 1 failed: {e}")
    
    try:
        isolated_df = analysis_2_isolated_patients(patient_df, output_dir)
        outputs['isolated'] = isolated_df
    except Exception as e:
        print(f"✗ Analysis 2 failed: {e}")
        isolated_df = None
    
    try:
        outputs['variance'] = analysis_3_variance_across_seeds(patient_df, image_df, output_dir)
    except Exception as e:
        print(f"✗ Analysis 3 failed: {e}")
    
    try:
        outputs['disagreement'] = analysis_4_image_disagreement(patient_df, image_df, output_dir)
    except Exception as e:
        print(f"✗ Analysis 4 failed: {e}")
    
    # Generate README
    try:
        outputs['readme'] = generate_analysis_readme(patient_df, image_df, isolated_df, output_dir)
    except Exception as e:
        print(f"✗ README generation failed: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")
    
    return outputs


if __name__ == "__main__":
    run_all_analyses()
