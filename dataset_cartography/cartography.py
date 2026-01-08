"""
Cartography metrics computation and sample categorization.

This module handles:
- Computing final confidence, variability, and correctness metrics
- Categorizing samples into Easy, Hard, Ambiguous, and Medium
- Saving metrics to CSV
"""

import os
import numpy as np
import pandas as pd

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    METRIC_START_EPOCH, METRIC_END_EPOCH,
    EASY_CORRECTNESS_THRESHOLD, EASY_VARIABILITY_THRESHOLD,
    HARD_CORRECTNESS_THRESHOLD, HARD_VARIABILITY_THRESHOLD,
    AMBIGUOUS_CORRECTNESS_THRESHOLD, AMBIGUOUS_VARIABILITY_THRESHOLD,
    MEDIUM_CORRECTNESS_MIN, MEDIUM_CORRECTNESS_MAX, MEDIUM_VARIABILITY_THRESHOLD,
    OUTPUT_DIR, METRICS_CSV
)


def compute_cartography_metrics(confidence_matrix, correctness_matrix, metadata):
    """
    Compute final cartography metrics for all samples.
    
    For each image, compute:
    - Confidence: average confidence across last 5 epochs and all runs
    - Variability: standard deviation of confidence across last 5 epochs and all runs
    - Correctness: fraction of times predicted correctly across last 5 epochs and all runs
    
    Args:
        confidence_matrix: dict {image_no: [[conf per epoch] per run]}
        correctness_matrix: dict {image_no: [[corr per epoch] per run]}
        metadata: DataFrame with image metadata
        
    Returns:
        DataFrame with columns: image_no, confidence, variability, correctness, category, ...
    """
    print("\nComputing cartography metrics...")
    print(f"Using epochs {METRIC_START_EPOCH+1} to {METRIC_END_EPOCH} (last 5 epochs)")
    
    results = []
    
    for _, row in metadata.iterrows():
        img_no = int(row['image_no'])
        
        # Get confidence values for this image
        conf_runs = confidence_matrix.get(img_no, [])
        corr_runs = correctness_matrix.get(img_no, [])
        
        # Collect values from last 5 epochs across all runs
        conf_values = []
        corr_values = []
        
        for run_idx in range(len(conf_runs)):
            run_conf = conf_runs[run_idx]
            run_corr = corr_runs[run_idx]
            
            # Get values from last 5 epochs (or available epochs if less)
            start_idx = min(METRIC_START_EPOCH, len(run_conf) - 5) if len(run_conf) >= 5 else 0
            end_idx = len(run_conf)
            
            conf_values.extend(run_conf[start_idx:end_idx])
            corr_values.extend(run_corr[start_idx:end_idx])
        
        # Compute metrics
        if len(conf_values) > 0:
            confidence = np.mean(conf_values)
            variability = np.std(conf_values)
            correctness = np.mean(corr_values)
        else:
            confidence = 0.5
            variability = 0.0
            correctness = 0.0
        
        # Store result
        result = {
            'image_no': img_no,
            'patient_id': row['patient_id'],
            'case_id': row['case_id'],
            'condition': row['condition'],
            'label': row['label'],
            'confidence': confidence,
            'variability': variability,
            'correctness': correctness
        }
        results.append(result)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Categorize samples
    metrics_df['category'] = metrics_df.apply(categorize_sample, axis=1)
    
    print(f"Computed metrics for {len(metrics_df)} images")
    
    return metrics_df


def categorize_sample(row):
    """
    Categorize a sample based on its metrics.
    
    Categories:
    - Easy: correctness >= 0.8 AND variability < 0.1
    - Ambiguous: correctness >= 0.5 AND variability >= 0.2
    - Hard: correctness < 0.5 AND variability >= 0.2
    - Medium: variability < 0.2 AND correctness between 0.5 and 0.8
    
    Args:
        row: DataFrame row with confidence, variability, correctness
        
    Returns:
        str: Category name
    """
    confidence = row['confidence']
    variability = row['variability']
    correctness = row['correctness']
    
    # Easy samples: high correctness, low variability
    if correctness >= EASY_CORRECTNESS_THRESHOLD and variability < EASY_VARIABILITY_THRESHOLD:
        return 'Easy'
    
    # Hard samples: low correctness, high variability
    if correctness < HARD_CORRECTNESS_THRESHOLD and variability >= HARD_VARIABILITY_THRESHOLD:
        return 'Hard'
    
    # Ambiguous samples: medium-high correctness, high variability
    if correctness >= AMBIGUOUS_CORRECTNESS_THRESHOLD and variability >= AMBIGUOUS_VARIABILITY_THRESHOLD:
        return 'Ambiguous'
    
    # Medium samples: low variability, medium correctness
    if variability < MEDIUM_VARIABILITY_THRESHOLD and MEDIUM_CORRECTNESS_MIN <= correctness < MEDIUM_CORRECTNESS_MAX:
        return 'Medium'
    
    # Default fallback (should rarely occur)
    return 'Medium'


def get_category_distribution(metrics_df):
    """
    Get the distribution of samples across categories.
    
    Args:
        metrics_df: DataFrame with category column
        
    Returns:
        dict: {category: count}
    """
    distribution = metrics_df['category'].value_counts().to_dict()
    
    # Ensure all categories are present
    for cat in ['Easy', 'Hard', 'Ambiguous', 'Medium']:
        if cat not in distribution:
            distribution[cat] = 0
    
    return distribution


def get_per_patient_breakdown(metrics_df):
    """
    Get category breakdown per patient.
    
    Args:
        metrics_df: DataFrame with patient_id and category columns
        
    Returns:
        DataFrame with columns: patient_id, Easy, Hard, Ambiguous, Medium, Total
    """
    # Group by patient and category, count samples
    patient_category = metrics_df.groupby(['patient_id', 'category']).size().unstack(fill_value=0)
    
    # Ensure all categories are present
    for cat in ['Easy', 'Hard', 'Ambiguous', 'Medium']:
        if cat not in patient_category.columns:
            patient_category[cat] = 0
    
    # Add total column
    patient_category['Total'] = patient_category.sum(axis=1)
    
    # Calculate proportion of hard/ambiguous samples
    patient_category['Hard_Ambiguous_Ratio'] = (
        (patient_category['Hard'] + patient_category['Ambiguous']) / patient_category['Total']
    )
    
    # Sort by Hard+Ambiguous ratio (highest first)
    patient_category = patient_category.sort_values('Hard_Ambiguous_Ratio', ascending=False)
    
    # Reset index
    patient_category = patient_category.reset_index()
    
    return patient_category


def save_metrics(metrics_df, output_path=None):
    """
    Save cartography metrics to CSV file.
    
    Args:
        metrics_df: DataFrame with metrics
        output_path: Path to save CSV (default from config)
    """
    if output_path is None:
        output_path = OUTPUT_DIR / METRICS_CSV
    
    metrics_df.to_csv(output_path, index=False)
    print(f"Cartography metrics saved to: {output_path}")


def print_summary(metrics_df):
    """
    Print a summary of the cartography analysis.
    
    Args:
        metrics_df: DataFrame with metrics
    """
    print("\n" + "="*60)
    print("CARTOGRAPHY ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall category distribution
    distribution = get_category_distribution(metrics_df)
    total = sum(distribution.values())
    
    print("\nCategory Distribution:")
    print("-" * 40)
    for cat in ['Easy', 'Medium', 'Ambiguous', 'Hard']:
        count = distribution[cat]
        pct = 100 * count / total if total > 0 else 0
        print(f"  {cat:12s}: {count:4d} ({pct:5.1f}%)")
    print(f"  {'Total':12s}: {total:4d}")
    
    # Metrics summary
    print("\nMetrics Summary:")
    print("-" * 40)
    print(f"  Confidence:  mean={metrics_df['confidence'].mean():.3f}, "
          f"std={metrics_df['confidence'].std():.3f}")
    print(f"  Variability: mean={metrics_df['variability'].mean():.3f}, "
          f"std={metrics_df['variability'].std():.3f}")
    print(f"  Correctness: mean={metrics_df['correctness'].mean():.3f}, "
          f"std={metrics_df['correctness'].std():.3f}")
    
    # Per-condition breakdown
    print("\nCategory Distribution by Condition:")
    print("-" * 40)
    for condition in metrics_df['condition'].unique():
        cond_df = metrics_df[metrics_df['condition'] == condition]
        cond_dist = cond_df['category'].value_counts()
        print(f"\n  {condition}:")
        for cat in ['Easy', 'Medium', 'Ambiguous', 'Hard']:
            count = cond_dist.get(cat, 0)
            pct = 100 * count / len(cond_df)
            print(f"    {cat:12s}: {count:3d} ({pct:5.1f}%)")
    
    # Patients with most hard/ambiguous samples
    patient_breakdown = get_per_patient_breakdown(metrics_df)
    
    print("\nPatients with Highest Hard/Ambiguous Ratio:")
    print("-" * 40)
    top_patients = patient_breakdown.head(5)
    for _, row in top_patients.iterrows():
        print(f"  {row['patient_id']:15s}: "
              f"Hard={int(row['Hard']):2d}, Ambiguous={int(row['Ambiguous']):2d}, "
              f"Ratio={row['Hard_Ambiguous_Ratio']:.2f}")


if __name__ == "__main__":
    print("Cartography metrics module loaded successfully.")
    print("Use compute_cartography_metrics() after training to analyze results.")
