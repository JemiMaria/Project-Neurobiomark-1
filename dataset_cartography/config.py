"""
Configuration settings for Dataset Cartography analysis.

This file contains all hyperparameters and paths used throughout the project.
Modify these settings to adapt the analysis to your specific dataset.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (parent of this config file)
BASE_DIR = Path(__file__).parent.parent

# Input paths - UPDATE THESE TO YOUR ACTUAL PATHS
IMAGE_FOLDER = r"C:\path\to\your\images"  # Folder containing 1.tiff, 2.tiff, etc.
EXCEL_FILE = r"C:\path\to\your\metadata.xlsx"  # Excel file with image metadata

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
LOPO_DIR = OUTPUT_DIR / "lopo"  # LOPO evaluation outputs

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Image settings
IMAGE_SIZE = 224  # Resize images to 224x224
NUM_IMAGES = 190  # Total number of images in dataset

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Excel column names
COL_IMAGE_NO = "Image No"
COL_CASE_ID = "Case ID"
COL_REGION = "Region"
COL_CATEGORY = "Category"
COL_CONDITION = "Condition"

# Class labels
POSITIVE_CLASS = "Case"  # Label for positive class (disease/condition)
NEGATIVE_CLASS = "Control"  # Label for negative class (healthy)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model architecture
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1  # Binary classification (using BCEWithLogitsLoss)
DROPOUT_RATE = 0.3  # Dropout before final classifier

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Hyperparameters (FIXED across all runs) - tuned to prevent overfitting
LEARNING_RATE = 5e-5  # Slower learning rate for stability
WEIGHT_DECAY = 1e-4   # Moderate L2 regularization
BATCH_SIZE = 8        # Smaller batches for regularization effect
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 3  # Stop if val_loss doesn't improve for 3 consecutive epochs

# Label smoothing for regularization (reduces overconfidence)
LABEL_SMOOTHING = 0.05  # Kept for reference, but not used (replaced by pos_weight)

# Learning rate scheduler settings
LR_SCHEDULER_FACTOR = 0.5    # Reduce LR by half
LR_SCHEDULER_PATIENCE = 2    # After 2 epochs without improvement

# Class imbalance handling
# Using pos_weight (balanced class weights) computed dynamically from training data
# pos_weight = n_negative / n_positive (weights the positive class higher when minority)
# This replaces label smoothing for handling class imbalance
USE_BALANCED_WEIGHTS = True  # Enable dynamic pos_weight computation

# Multiple runs configuration
NUM_RUNS = 5
RANDOM_SEEDS = [1, 2, 3, 4, 5]

# Validation split (patient-level split)
VALIDATION_SPLIT = 0.2  # 20% of patients for validation
TEST_SPLIT = 0.1        # 10% of patients for test (optional)

# =============================================================================
# CARTOGRAPHY CONFIGURATION
# =============================================================================

# Windowed epoch strategy settings
# Window is centered on best validation loss epoch: [t* - WINDOW_HALF, t* + WINDOW_HALF]
WINDOW_HALF_SIZE = 2  # Window = ±2 epochs around best val_loss epoch (5 epochs total)

# Legacy settings (deprecated - kept for backward compatibility)
METRIC_START_EPOCH = 15  # 0-indexed, so this is epoch 16
METRIC_END_EPOCH = 20    # Exclusive, so epochs 16-20 (last 5)

# Category thresholds
# Easy: correctness >= 0.8 AND variability < 0.1
EASY_CORRECTNESS_THRESHOLD = 0.8  # correctness >= 0.8
EASY_VARIABILITY_THRESHOLD = 0.1  # variability < 0.1

# Hard: correctness < 0.5 AND variability >= 0.2
HARD_CORRECTNESS_THRESHOLD = 0.5  # correctness < 0.5
HARD_VARIABILITY_THRESHOLD = 0.2  # variability >= 0.2

# Ambiguous: correctness >= 0.5 AND variability >= 0.2
AMBIGUOUS_CORRECTNESS_THRESHOLD = 0.5  # correctness >= 0.5
AMBIGUOUS_VARIABILITY_THRESHOLD = 0.2  # variability >= 0.2

# Medium: variability < 0.2 AND correctness between 0.5 and 0.8
MEDIUM_CORRECTNESS_MIN = 0.5
MEDIUM_CORRECTNESS_MAX = 0.8
MEDIUM_VARIABILITY_THRESHOLD = 0.2  # variability < 0.2

# =============================================================================
# OUTPUT FILES
# =============================================================================

METRICS_CSV = "cartography_metrics.csv"
TRAINING_LOG_CSV = "training_logs.csv"
CARTOGRAPHY_PER_IMAGE_XLSX = "cartography_per_image.xlsx"
CARTOGRAPHY_PER_PATIENT_XLSX = "cartography_per_patient.xlsx"
SCATTER_PLOT_PNG = "cartography_scatter_plot.png"
CATEGORY_DISTRIBUTION_PNG = "category_distribution.png"
DOCUMENTATION_FILE = "CARTOGRAPHY_REPORT.html"


def create_output_directories():
    """Create all necessary output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOPO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created at: {OUTPUT_DIR}")


def validate_paths():
    """Check if input paths exist and are valid."""
    errors = []
    
    if not os.path.exists(IMAGE_FOLDER):
        errors.append(f"Image folder not found: {IMAGE_FOLDER}")
    
    if not os.path.exists(EXCEL_FILE):
        errors.append(f"Excel file not found: {EXCEL_FILE}")
    
    if errors:
        print("=" * 60)
        print("CONFIGURATION ERROR - Please update config.py:")
        print("=" * 60)
        for error in errors:
            print(f"  - {error}")
        print("\nUpdate IMAGE_FOLDER and EXCEL_FILE in config.py")
        print("=" * 60)
        return False
    
    return True


if __name__ == "__main__":
    # Quick test of configuration
    print("Dataset Cartography Configuration")
    print("=" * 40)
    print(f"Image folder: {IMAGE_FOLDER}")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    validate_paths()
