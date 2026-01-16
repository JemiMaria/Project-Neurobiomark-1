"""
Visualization for ALS vs ALS-FTD LOPO Cartography Analysis.

Generates 5 key plots:
1. Confidence vs Correctness scatter (patient-level)
2. Seed Variance - Correctness (bar chart)
3. Seed Variance - Probability (bar chart)
4. Image Disagreement (per patient)
5. Category Summary (pie/bar)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    FIGURES_DIR,
    EASY_CORRECTNESS_THRESHOLD, EASY_CONFIDENCE_THRESHOLD,
    HARD_CORRECTNESS_THRESHOLD, AMBIGUOUS_CORRECTNESS_MIN, AMBIGUOUS_CORRECTNESS_MAX
)


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_confidence_vs_correctness(patient_df, output_dir=None, filename='fig1_confidence_vs_correctness.png'):
    """
    Plot 1: Confidence vs Correctness scatter plot.
    
    - X-axis: Mean Correctness
    - Y-axis: Mean Confidence
    - Color: ALS (blue) vs ALS-FTD (red)
    - Marker size: inversely proportional to variability
    - Threshold lines for Easy/Hard/Ambiguous regions
    
    Args:
        patient_df: DataFrame with patient metrics
        output_dir: Output directory
        filename: Output filename
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get data
    als_patients = patient_df[patient_df['y_true'] == 0]
    alsftd_patients = patient_df[patient_df['y_true'] == 1]
    
    # Marker size based on stability (smaller variability = larger marker)
    def get_marker_size(variability, min_size=50, max_size=300):
        """Inverse mapping: low variability -> large marker."""
        max_var = patient_df['final_std_prob'].max()
        min_var = patient_df['final_std_prob'].min()
        if max_var == min_var:
            return np.full(len(variability), (min_size + max_size) / 2)
        normalized = (variability - min_var) / (max_var - min_var)
        return max_size - normalized * (max_size - min_size)
    
    # Plot ALS (y=0) - blue
    als_sizes = get_marker_size(als_patients['final_std_prob'].values)
    ax.scatter(
        als_patients['final_mean_correctness'],
        als_patients['final_mean_confidence'],
        s=als_sizes,
        c='steelblue',
        alpha=0.7,
        edgecolors='navy',
        linewidths=1.5,
        label='ALS (y=0)',
        zorder=3
    )
    
    # Plot ALS-FTD (y=1) - red
    alsftd_sizes = get_marker_size(alsftd_patients['final_std_prob'].values)
    ax.scatter(
        alsftd_patients['final_mean_correctness'],
        alsftd_patients['final_mean_confidence'],
        s=alsftd_sizes,
        c='indianred',
        alpha=0.7,
        edgecolors='darkred',
        linewidths=1.5,
        label='ALS-FTD (y=1)',
        marker='s',
        zorder=3
    )
    
    # Annotate patient IDs
    for _, row in patient_df.iterrows():
        ax.annotate(
            row['patient_id'],
            (row['final_mean_correctness'], row['final_mean_confidence']),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add threshold lines
    ax.axvline(x=EASY_CORRECTNESS_THRESHOLD, color='green', linestyle='--', alpha=0.5, label='Easy threshold')
    ax.axhline(y=EASY_CONFIDENCE_THRESHOLD, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=HARD_CORRECTNESS_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='Hard threshold')
    
    # Add region labels
    ax.text(0.9, 0.9, 'EASY', fontsize=16, color='green', alpha=0.6, ha='center', va='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.15, 0.15, 'HARD', fontsize=16, color='red', alpha=0.6, ha='center', va='center',
            transform=ax.transAxes, fontweight='bold')
    
    ax.set_xlabel('Mean Correctness')
    ax.set_ylabel('Mean Confidence')
    ax.set_title('ALS vs ALS-FTD: Patient-Level Cartography')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Save
    fig_path = output_dir / filename
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


def plot_seed_variance_correctness(patient_df, output_dir=None, filename='fig2_seed_variance_correctness.png'):
    """
    Plot 2: Seed variance in correctness per patient.
    
    Bar chart showing std_correctness across seeds for each patient.
    Sorted by variance (highest first).
    
    Args:
        patient_df: DataFrame with patient metrics
        output_dir: Output directory
        filename: Output filename
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_publication_style()
    
    # Sort by std_correctness descending
    sorted_df = patient_df.sort_values('final_std_correctness', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by class
    colors = ['indianred' if y == 1 else 'steelblue' for y in sorted_df['y_true']]
    
    bars = ax.bar(
        range(len(sorted_df)),
        sorted_df['final_std_correctness'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df['patient_id'], rotation=45, ha='right')
    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Std Correctness (across seeds)')
    ax.set_title('Seed Variance in Correctness by Patient')
    
    # Add threshold line
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Unstable threshold')
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='ALS (y=0)'),
        Patch(facecolor='indianred', label='ALS-FTD (y=1)'),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='Unstable threshold')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / filename
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


def plot_seed_variance_probability(patient_df, output_dir=None, filename='fig3_seed_variance_probability.png'):
    """
    Plot 3: Seed variance in probability per patient.
    
    Bar chart showing std_prob across seeds for each patient.
    
    Args:
        patient_df: DataFrame with patient metrics
        output_dir: Output directory
        filename: Output filename
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_publication_style()
    
    # Sort by std_prob descending
    sorted_df = patient_df.sort_values('final_std_prob', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by class
    colors = ['indianred' if y == 1 else 'steelblue' for y in sorted_df['y_true']]
    
    bars = ax.bar(
        range(len(sorted_df)),
        sorted_df['final_std_prob'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df['patient_id'], rotation=45, ha='right')
    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Std Probability (across seeds)')
    ax.set_title('Seed Variance in Prediction Probability by Patient')
    
    # Add threshold line
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Unstable threshold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='ALS (y=0)'),
        Patch(facecolor='indianred', label='ALS-FTD (y=1)'),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='Unstable threshold')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / filename
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


def plot_image_disagreement(image_df, output_dir=None, filename='fig4_image_disagreement.png'):
    """
    Plot 4: Image-level disagreement per patient.
    
    Shows number of images that were incorrectly classified per patient.
    
    Args:
        image_df: DataFrame with image-level metrics
        output_dir: Output directory
        filename: Output filename
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_publication_style()
    
    # Count images per patient
    patient_stats = image_df.groupby('patient_id').agg({
        'y_true': 'first',
        'mean_correctness': ['count', 'mean']
    }).reset_index()
    patient_stats.columns = ['patient_id', 'y_true', 'n_images', 'mean_correctness']
    
    # Compute incorrect images
    patient_stats['n_incorrect'] = patient_stats['n_images'] * (1 - patient_stats['mean_correctness'])
    patient_stats['n_incorrect'] = patient_stats['n_incorrect'].round().astype(int)
    patient_stats['n_correct'] = patient_stats['n_images'] - patient_stats['n_incorrect']
    
    # Sort by n_incorrect descending
    sorted_df = patient_stats.sort_values('n_incorrect', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sorted_df))
    width = 0.6
    
    # Stacked bar: correct (green) + incorrect (red)
    bars_correct = ax.bar(x, sorted_df['n_correct'], width, label='Correct', color='seagreen', alpha=0.8)
    bars_incorrect = ax.bar(x, sorted_df['n_incorrect'], width, bottom=sorted_df['n_correct'],
                            label='Incorrect', color='crimson', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['patient_id'], rotation=45, ha='right')
    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Number of Images')
    ax.set_title('Image Classification per Patient (Correct vs Incorrect)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / filename
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


def plot_category_summary(patient_df, output_dir=None, filename='fig5_category_summary.png'):
    """
    Plot 5: Patient category summary.
    
    Pie chart or bar chart showing distribution of Easy/Medium/Ambiguous/Hard.
    
    Args:
        patient_df: DataFrame with patient categorization
        output_dir: Output directory
        filename: Output filename
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Category distribution (pie)
    ax1 = axes[0]
    category_counts = patient_df['category'].value_counts()
    colors_map = {
        'Easy': 'limegreen',
        'Medium': 'gold',
        'Ambiguous': 'orange',
        'Hard': 'crimson'
    }
    colors = [colors_map.get(cat, 'gray') for cat in category_counts.index]
    
    wedges, texts, autotexts = ax1.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.0f%%',
        colors=colors,
        explode=[0.05] * len(category_counts),
        startangle=90
    )
    ax1.set_title('Patient Category Distribution')
    
    # Right: Flag summary (bar)
    ax2 = axes[1]
    flags = {
        'Borderline': patient_df['is_borderline'].sum() if 'is_borderline' in patient_df.columns else 0,
        'Unstable\n(seeds)': patient_df['is_unstable_seeds'].sum() if 'is_unstable_seeds' in patient_df.columns else 0,
        'Unstable\n(images)': patient_df['is_unstable_images'].sum() if 'is_unstable_images' in patient_df.columns else 0,
        'Outlier': patient_df['is_outlier'].sum() if 'is_outlier' in patient_df.columns else 0,
        'Hard': (patient_df['category'] == 'Hard').sum()
    }
    
    flag_colors = ['orange', 'red', 'darkred', 'purple', 'crimson']
    ax2.bar(flags.keys(), flags.values(), color=flag_colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Problem Patient Flags')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, (k, v) in enumerate(flags.items()):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / filename
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


def generate_all_visualizations(patient_final_df, image_df=None, categorized_df=None):
    """
    Generate all 5 visualization plots.
    
    Args:
        patient_final_df: DataFrame with patient metrics
        image_df: DataFrame with image-level metrics (optional)
        categorized_df: DataFrame with categorization (optional, will use patient_final_df if None)
    """
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Plot 1: Confidence vs Correctness
    print("\n  Generating Plot 1: Confidence vs Correctness...")
    plot_confidence_vs_correctness(patient_final_df)
    
    # Plot 2: Seed variance - correctness
    print("\n  Generating Plot 2: Seed Variance (Correctness)...")
    plot_seed_variance_correctness(patient_final_df)
    
    # Plot 3: Seed variance - probability
    print("\n  Generating Plot 3: Seed Variance (Probability)...")
    plot_seed_variance_probability(patient_final_df)
    
    # Plot 4: Image disagreement
    if image_df is not None and len(image_df) > 0:
        print("\n  Generating Plot 4: Image Disagreement...")
        plot_image_disagreement(image_df)
    else:
        print("\n  Skipping Plot 4: No image-level data provided")
    
    # Plot 5: Category summary
    cat_df = categorized_df if categorized_df is not None else patient_final_df
    if 'category' in cat_df.columns:
        print("\n  Generating Plot 5: Category Summary...")
        plot_category_summary(cat_df)
    else:
        print("\n  Skipping Plot 5: No categorization data")
    
    print(f"\n  ✓ All visualizations complete!")
    print(f"  → Saved to: {FIGURES_DIR}")
