"""
LOPO Visualizations - Create clinical evaluation plots.

This module creates:
1. Confidence vs Correctness scatter (per patient)
2. Seed Stability error bar plot
3. Image Instability boxplot
4. ROC curve across folds
5. Patient prediction summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def create_lopo_visualizations(lopo_per_patient_final, lopo_per_patient_seed, 
                               lopo_per_image, clinical_metrics, output_dir):
    """
    Create all LOPO visualizations.
    
    Args:
        lopo_per_patient_final: DataFrame with per-patient aggregated metrics
        lopo_per_patient_seed: DataFrame with per-patient-seed metrics
        lopo_per_image: DataFrame with per-image windowed metrics
        clinical_metrics: DataFrame with clinical metrics
        output_dir: Directory to save plots
    """
    # output_dir is already the analysis directory, don't create a subdirectory
    analysis_dir = output_dir
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"\nCreating LOPO visualizations in: {analysis_dir}")
    
    # 1. Patient Confidence vs Correctness
    plot_patient_confidence_correctness(lopo_per_patient_final, analysis_dir)
    
    # 2. Seed Stability per Patient
    plot_seed_stability(lopo_per_patient_seed, analysis_dir)
    
    # 3. Image Instability Boxplot
    plot_image_instability(lopo_per_image, analysis_dir)
    
    # 4. Metrics Comparison Across Folds
    plot_metrics_across_folds(clinical_metrics, analysis_dir)
    
    # 5. Patient Prediction Summary
    plot_patient_predictions(lopo_per_patient_final, analysis_dir)
    
    # 6. Generate README
    generate_lopo_readme(lopo_per_patient_final, clinical_metrics, analysis_dir, output_dir)
    
    print(f"All LOPO visualizations saved!")


def plot_patient_confidence_correctness(df, output_dir):
    """
    Scatter plot of confidence vs correctness for each patient.
    Color by true label, shape by correct prediction.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create prediction column
    df = df.copy()
    df['predicted'] = (df['mean_prob_across_seeds'] >= 0.5).astype(int)
    df['correct'] = (df['predicted'] == df['y_true']).astype(int)
    
    # Use std_prob_across_seeds as variability, mean_confidence_across_seeds as confidence
    variability_col = 'std_prob_across_seeds'
    confidence_col = 'mean_confidence_across_seeds'
    patient_col = 'fold_patient_id'
    
    # Create markers: circle for correct, X for incorrect
    for idx, row in df.iterrows():
        marker = 'o' if row['correct'] else 'X'
        color = 'tab:blue' if row['y_true'] == 0 else 'tab:orange'
        ax.scatter(row[variability_col], row[confidence_col],
                   c=color, marker=marker, s=150, alpha=0.8, edgecolors='black')
        ax.annotate(row[patient_col], (row[variability_col], 
                    row[confidence_col]), fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Variability Across Seeds (Std Prob)', fontsize=12)
    ax.set_ylabel('Confidence (Mean Confidence)', fontsize=12)
    ax.set_title('LOPO: Patient Confidence vs Stability\n(Circle=Correct, X=Incorrect)', fontsize=14)
    ax.set_xlim(0, max(0.5, df[variability_col].max() * 1.1))
    ax.set_ylim(0, 1.05)
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='High variability threshold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', 
               markersize=10, label='Class 0 (Correct)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='tab:blue', 
               markersize=10, label='Class 0 (Incorrect)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', 
               markersize=10, label='Class 1 (Correct)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='tab:orange', 
               markersize=10, label='Class 1 (Incorrect)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lopo_patient_confidence_correctness.png'), dpi=150)
    plt.close()
    print("  - Saved: lopo_patient_confidence_correctness.png")


def plot_seed_stability(df, output_dir):
    """
    Error bar plot showing prediction stability across seeds per patient.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Check if we have patient_id or fold_patient_id
    patient_col = 'patient_id' if 'patient_id' in df.columns else 'fold_patient_id'
    prob_col = 'mean_prob' if 'mean_prob' in df.columns else 'patient_mean_prob'
    
    # Compute mean and std per patient
    patient_stats = df.groupby(patient_col).agg({
        prob_col: ['mean', 'std'],
        'y_true': 'first'
    }).reset_index()
    patient_stats.columns = ['patient_id', 'prob_mean', 'prob_std', 'y_true']
    patient_stats = patient_stats.sort_values('prob_mean')
    
    colors = ['tab:blue' if y == 0 else 'tab:orange' for y in patient_stats['y_true']]
    
    x = np.arange(len(patient_stats))
    ax.bar(x, patient_stats['prob_mean'], yerr=patient_stats['prob_std'].fillna(0),
           color=colors, capsize=4, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(patient_stats['patient_id'], rotation=45, ha='right')
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel('Mean Probability (±1 Std)', fontsize=12)
    ax.set_title('LOPO: Patient Prediction Stability Across Seeds', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Color legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor='tab:blue', label='True Class 0'),
                      Patch(facecolor='tab:orange', label='True Class 1')]
    ax.legend(handles=legend_patches + [ax.get_legend_handles_labels()[0][0]], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lopo_seed_stability.png'), dpi=150)
    plt.close()
    print("  - Saved: lopo_seed_stability.png")


def plot_image_instability(df, output_dir):
    """
    Boxplot showing image-level variability distribution per patient.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Aggregate image variability per patient
    image_var = df.groupby(['fold_patient_id', 'image_id']).agg({
        'std_prob_window': 'first'  # Already aggregated at image level
    }).reset_index()
    
    # Sort patients by median variability
    patient_order = image_var.groupby('fold_patient_id')['std_prob_window'].median()
    patient_order = patient_order.sort_values().index.tolist()
    
    sns.boxplot(data=image_var, x='fold_patient_id', y='std_prob_window',
                order=patient_order, ax=ax, palette='viridis')
    
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High instability threshold')
    ax.set_xlabel('Patient ID (Held-Out)', fontsize=12)
    ax.set_ylabel('Image Probability Std (Within Window)', fontsize=12)
    ax.set_title('LOPO: Image-Level Prediction Instability per Patient', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lopo_image_instability.png'), dpi=150)
    plt.close()
    print("  - Saved: lopo_image_instability.png")


def plot_metrics_across_folds(clinical_metrics, output_dir):
    """
    Bar plot showing metrics across all folds.
    """
    # Get image-level metrics
    image_metrics = clinical_metrics[clinical_metrics['level'] == 'image'].copy()
    
    if len(image_metrics) == 0:
        print("  - Skipped: No image-level metrics for fold comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('balanced_accuracy', 'Balanced Accuracy', axes[0, 0]),
        ('sensitivity', 'Sensitivity (Recall)', axes[0, 1]),
        ('specificity', 'Specificity', axes[1, 0]),
        ('auc', 'AUC', axes[1, 1])
    ]
    
    for metric, title, ax in metrics_to_plot:
        if metric in image_metrics.columns:
            values = image_metrics[[metric, 'fold_patient_id']].dropna()
            if len(values) > 0:
                x = np.arange(len(values))
                bars = ax.bar(x, values[metric], color='steelblue', alpha=0.8, edgecolor='black')
                ax.set_xticks(x)
                ax.set_xticklabels(values['fold_patient_id'], rotation=45, ha='right')
                ax.set_ylabel(title)
                ax.set_title(f'{title} per Fold')
                ax.set_ylim(0, 1)
                
                # Add mean line
                mean_val = values[metric].mean()
                ax.axhline(y=mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
                ax.legend()
    
    plt.suptitle('LOPO: Clinical Metrics Across Folds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lopo_metrics_across_folds.png'), dpi=150)
    plt.close()
    print("  - Saved: lopo_metrics_across_folds.png")


def plot_patient_predictions(df, output_dir):
    """
    Summary plot showing patient predictions vs ground truth.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df = df.copy()
    df['predicted'] = (df['mean_prob_across_seeds'] >= 0.5).astype(int)
    df['correct'] = df['predicted'] == df['y_true']
    df = df.sort_values('mean_prob_across_seeds')
    
    # Use fold_patient_id as patient identifier
    patient_col = 'fold_patient_id'
    
    x = np.arange(len(df))
    
    # Bar colors based on correctness
    colors = ['green' if c else 'red' for c in df['correct']]
    
    bars = ax.bar(x, df['mean_prob_across_seeds'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add true label indicators
    for i, (idx, row) in enumerate(df.iterrows()):
        marker = '▲' if row['y_true'] == 1 else '▼'
        color = 'tab:orange' if row['y_true'] == 1 else 'tab:blue'
        ax.text(i, 1.02, marker, ha='center', va='bottom', fontsize=12, color=color)
    
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(df[patient_col], rotation=45, ha='right')
    ax.set_xlabel('Patient ID', fontsize=12)
    ax.set_ylabel('Mean Predicted Probability', fontsize=12)
    ax.set_title('LOPO: Patient Predictions\n(Green=Correct, Red=Incorrect, ▲=True Class 1, ▼=True Class 0)', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend()
    
    # Summary text
    n_correct = df['correct'].sum()
    n_total = len(df)
    accuracy = n_correct / n_total
    ax.text(0.02, 0.95, f'Accuracy: {n_correct}/{n_total} ({accuracy:.1%})',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lopo_patient_predictions.png'), dpi=150)
    plt.close()
    print("  - Saved: lopo_patient_predictions.png")


def generate_lopo_readme(lopo_per_patient_final, clinical_metrics, analysis_dir, output_dir):
    """
    Generate README summarizing LOPO results.
    """
    # Patient-level metrics
    patient_row = clinical_metrics[clinical_metrics['level'] == 'patient'].iloc[0]
    
    # Image-level summary
    image_metrics = clinical_metrics[clinical_metrics['level'] == 'image']
    
    # Patient predictions
    df = lopo_per_patient_final.copy()
    df['predicted'] = (df['mean_prob_across_seeds'] >= 0.5).astype(int)
    df['correct'] = df['predicted'] == df['y_true']
    
    # Use correct column names
    patient_col = 'fold_patient_id'
    variability_col = 'std_prob_across_seeds'
    
    misclassified = df[~df['correct']][[patient_col, 'y_true', 'mean_prob_across_seeds', 
                                         variability_col]].round(4)
    misclassified = misclassified.rename(columns={patient_col: 'patient_id', 
                                                   variability_col: 'variability_across_seeds'})
    
    high_var = df[df[variability_col] > 0.1][[patient_col, 'y_true',
                                               variability_col]].round(4)
    high_var = high_var.rename(columns={patient_col: 'patient_id', 
                                         variability_col: 'variability_across_seeds'})
    
    readme_content = f"""# LOPO Evaluation Results

## Overview
Leave-One-Patient-Out (LOPO) cross-validation was performed to evaluate model generalization.
- **15 folds** (one per patient)
- **5 seeds per fold** (ensemble stability)
- **Windowed metrics** computed around best validation epoch (±2 epochs)

## Patient-Level Clinical Metrics (Overall)
| Metric | Value |
|--------|-------|
| Accuracy | {patient_row['accuracy']:.4f} |
| Sensitivity | {patient_row['sensitivity']:.4f} |
| Specificity | {patient_row['specificity']:.4f} |
| Balanced Accuracy | {patient_row['balanced_accuracy']:.4f} |
| AUC | {patient_row['auc']:.4f} |
| F1 Score | {patient_row['f1']:.4f} |
| True Positives | {patient_row.get('tp', 'N/A')} |
| True Negatives | {patient_row.get('tn', 'N/A')} |
| False Positives | {patient_row.get('fp', 'N/A')} |
| False Negatives | {patient_row.get('fn', 'N/A')} |

## Image-Level Metrics (Mean ± Std Across Folds)
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | {image_metrics['accuracy'].mean():.4f} | {image_metrics['accuracy'].std():.4f} |
| Sensitivity | {image_metrics['sensitivity'].mean():.4f} | {image_metrics['sensitivity'].std():.4f} |
| Specificity | {image_metrics['specificity'].mean():.4f} | {image_metrics['specificity'].std():.4f} |
| Balanced Accuracy | {image_metrics['balanced_accuracy'].mean():.4f} | {image_metrics['balanced_accuracy'].std():.4f} |
| AUC | {image_metrics['auc'].mean():.4f} | {image_metrics['auc'].std():.4f} |

## Misclassified Patients ({len(misclassified)})
{dataframe_to_markdown_simple(misclassified) if len(misclassified) > 0 else "All patients correctly classified!"}

## High Variability Patients (>0.1) ({len(high_var)})
{dataframe_to_markdown_simple(high_var) if len(high_var) > 0 else "No patients with high variability across seeds."}

## Generated Visualizations
1. **lopo_patient_confidence_correctness.png** - Confidence vs Stability scatter
2. **lopo_seed_stability.png** - Prediction stability across seeds
3. **lopo_image_instability.png** - Image-level variability boxplot
4. **lopo_metrics_across_folds.png** - Clinical metrics per fold
5. **lopo_patient_predictions.png** - Patient prediction summary

## Output Files
- `lopo_per_image.xlsx` - Per-image windowed metrics
- `lopo_per_patient_seed.xlsx` - Per-patient-seed metrics
- `lopo_per_patient_final.xlsx` - Per-patient final metrics (aggregated across seeds)
- `lopo_clinical_metrics.csv` - Clinical metrics table
- `lopo_training_logs.csv` - Training logs for all folds/seeds

## Interpretation
- **Sensitivity** = True Positive Rate (correctly identifying disease)
- **Specificity** = True Negative Rate (correctly identifying healthy)
- **High variability patients** may indicate boundary cases or data quality issues
- **Misclassified patients** warrant clinical review

---
*Generated by Dataset Cartography LOPO Module*
"""
    
    readme_path = os.path.join(analysis_dir, 'LOPO_README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  - Saved: LOPO_README.md")


if __name__ == "__main__":
    print("LOPO Visualizations Module")
    print("Run via lopo_runner.py")
