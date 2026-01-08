"""
Visualization utilities for Dataset Cartography.

This module creates:
- Scatter plot of confidence vs variability colored by correctness
- Category distribution bar chart
- Sample images from each category
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_cartography.config import (
    IMAGE_FOLDER, VISUALIZATIONS_DIR,
    SCATTER_PLOT_PNG, CATEGORY_DISTRIBUTION_PNG
)


# Color scheme for categories
CATEGORY_COLORS = {
    'Easy': '#2ecc71',      # Green
    'Medium': '#3498db',    # Blue
    'Ambiguous': '#f39c12', # Orange
    'Hard': '#e74c3c'       # Red
}


def create_cartography_scatter_plot(metrics_df, output_path=None, figsize=(12, 10)):
    """
    Create the main cartography scatter plot.
    
    X-axis: Confidence
    Y-axis: Variability
    Color: Correctness (gradient from red to green)
    Marker style: Category
    
    Args:
        metrics_df: DataFrame with metrics (confidence, variability, correctness, category)
        output_path: Path to save the plot
        figsize: Figure size as tuple
        
    Returns:
        matplotlib figure object
    """
    if output_path is None:
        output_path = VISUALIZATIONS_DIR / SCATTER_PLOT_PNG
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with correctness as color
    scatter = ax.scatter(
        metrics_df['confidence'],
        metrics_df['variability'],
        c=metrics_df['correctness'],
        cmap='RdYlGn',  # Red-Yellow-Green colormap
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Correctness', fontsize=12)
    
    # Mark category regions with semi-transparent backgrounds
    # Easy region (high confidence, low variability)
    ax.axhline(y=0.1, color=CATEGORY_COLORS['Easy'], linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.7, color=CATEGORY_COLORS['Easy'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Hard region indicators
    ax.axhline(y=0.2, color=CATEGORY_COLORS['Hard'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Add category legend markers
    for cat in ['Easy', 'Medium', 'Ambiguous', 'Hard']:
        cat_data = metrics_df[metrics_df['category'] == cat]
        if len(cat_data) > 0:
            # Add small markers at category mean positions
            mean_conf = cat_data['confidence'].mean()
            mean_var = cat_data['variability'].mean()
            ax.annotate(
                cat,
                xy=(mean_conf, mean_var),
                fontsize=10,
                fontweight='bold',
                color=CATEGORY_COLORS[cat],
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=CATEGORY_COLORS[cat], alpha=0.8)
            )
    
    # Labels and title
    ax.set_xlabel('Confidence (probability assigned to true label)', fontsize=12)
    ax.set_ylabel('Variability (std of confidence across epochs/runs)', fontsize=12)
    ax.set_title('Dataset Cartography: Sample Difficulty Analysis', fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, metrics_df['variability'].max() * 1.1))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text annotation explaining the plot
    explanation = (
        "Easy: High correctness, low variability (consistent correct predictions)\n"
        "Hard: Low correctness, high variability (consistently wrong)\n"
        "Ambiguous: Medium correctness, high variability (inconsistent predictions)"
    )
    ax.text(
        0.02, 0.98, explanation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save plot
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")
    
    return fig


def create_category_distribution_plot(metrics_df, output_path=None, figsize=(14, 5)):
    """
    Create category distribution bar charts.
    
    Shows:
    - Overall category distribution
    - Distribution by condition (Case vs Control)
    
    Args:
        metrics_df: DataFrame with metrics
        output_path: Path to save the plot
        figsize: Figure size as tuple
        
    Returns:
        matplotlib figure object
    """
    if output_path is None:
        output_path = VISUALIZATIONS_DIR / CATEGORY_DISTRIBUTION_PNG
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    categories = ['Easy', 'Medium', 'Ambiguous', 'Hard']
    colors = [CATEGORY_COLORS[cat] for cat in categories]
    
    # Plot 1: Overall distribution
    ax1 = axes[0]
    dist = metrics_df['category'].value_counts()
    counts = [dist.get(cat, 0) for cat in categories]
    
    bars1 = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1)
    ax1.set_title('Overall Category Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xlabel('Category')
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2 & 3: Distribution by condition
    conditions = metrics_df['condition'].unique()
    
    for idx, condition in enumerate(conditions[:2]):  # Limit to 2 conditions
        ax = axes[idx + 1]
        cond_df = metrics_df[metrics_df['condition'] == condition]
        cond_dist = cond_df['category'].value_counts()
        cond_counts = [cond_dist.get(cat, 0) for cat in categories]
        
        bars = ax.bar(categories, cond_counts, color=colors, edgecolor='black', linewidth=1)
        ax.set_title(f'{condition} Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.set_xlabel('Category')
        
        # Add count labels
        for bar, count in zip(bars, cond_counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Category distribution plot saved to: {output_path}")
    
    return fig


def create_sample_images_grid(metrics_df, output_path=None, samples_per_category=3, figsize=(16, 12)):
    """
    Create a grid showing sample images from each category.
    
    Args:
        metrics_df: DataFrame with metrics
        output_path: Path to save the plot
        samples_per_category: Number of samples to show per category
        figsize: Figure size as tuple
        
    Returns:
        matplotlib figure object
    """
    if output_path is None:
        output_path = VISUALIZATIONS_DIR / "sample_images.png"
    
    categories = ['Easy', 'Medium', 'Ambiguous', 'Hard']
    
    fig, axes = plt.subplots(4, samples_per_category, figsize=figsize)
    
    for row_idx, category in enumerate(categories):
        cat_df = metrics_df[metrics_df['category'] == category]
        
        # Sample images (or take all if less than requested)
        n_samples = min(samples_per_category, len(cat_df))
        
        if n_samples > 0:
            samples = cat_df.sample(n=n_samples, random_state=42)
        else:
            samples = pd.DataFrame()
        
        for col_idx in range(samples_per_category):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(samples):
                sample = samples.iloc[col_idx]
                img_no = int(sample['image_no'])
                
                # Try to load image
                img_path = os.path.join(IMAGE_FOLDER, f"{img_no}.tiff")
                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                    
                    # Add title with sample info
                    title = (f"#{img_no}\n"
                            f"{sample['patient_id']}\n"
                            f"Conf: {sample['confidence']:.2f}")
                    ax.set_title(title, fontsize=9)
                except FileNotFoundError:
                    ax.text(0.5, 0.5, f"Image {img_no}\nnot found",
                           ha='center', va='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, "No sample",
                       ha='center', va='center', fontsize=10)
            
            ax.axis('off')
            
            # Add row label (category) on first column
            if col_idx == 0:
                ax.annotate(
                    category,
                    xy=(-0.2, 0.5),
                    xycoords='axes fraction',
                    fontsize=14,
                    fontweight='bold',
                    color=CATEGORY_COLORS[category],
                    ha='right',
                    va='center',
                    rotation=90
                )
    
    plt.suptitle('Sample Images from Each Category', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sample images grid saved to: {output_path}")
    
    return fig


def create_all_visualizations(metrics_df):
    """
    Create all visualization plots.
    
    Args:
        metrics_df: DataFrame with metrics
        
    Returns:
        list of figure objects
    """
    print("\nCreating visualizations...")
    
    figures = []
    
    # 1. Main cartography scatter plot
    fig1 = create_cartography_scatter_plot(metrics_df)
    figures.append(fig1)
    
    # 2. Category distribution
    fig2 = create_category_distribution_plot(metrics_df)
    figures.append(fig2)
    
    # 3. Sample images (only if images exist)
    if os.path.exists(IMAGE_FOLDER):
        try:
            fig3 = create_sample_images_grid(metrics_df)
            figures.append(fig3)
        except Exception as e:
            print(f"Could not create sample images grid: {e}")
    
    print(f"\nCreated {len(figures)} visualization(s)")
    
    return figures


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("Use create_all_visualizations(metrics_df) after computing metrics.")
