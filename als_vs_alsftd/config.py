"""
Configuration for ALS vs ALS-FTD Windowed Cartography + LOPO Pipeline.

Task: Binary classification
- ALS (Discordant) → y=0 (negative)
- ALS-FTD (Concordant) → y=1 (positive)
- Controls excluded entirely
"""

from pathlib import Path
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (parent of als_vs_alsftd)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "raw_images"
METADATA_PATH = DATA_DIR / "image_keys.xlsx"
IMAGE_KEYS_PATH = METADATA_PATH  # Alias for compatibility

# Output directories
OUTPUT_DIR = Path(__file__).parent / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
CARTOGRAPHY_DIR = OUTPUT_DIR / "cartography"
LOPO_DIR = OUTPUT_DIR / "lopo"
LOPO_SPLITS_DIR = LOPO_DIR  # Alias for compatibility
TRAINING_LOGS_DIR = LOGS_DIR  # Alias for compatibility
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
FIGURES_DIR = PLOTS_DIR  # Alias for compatibility
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# Create directories
for directory in [OUTPUT_DIR, LOGS_DIR, CARTOGRAPHY_DIR, LOPO_DIR, 
                  METRICS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Excel column names (matching existing pipeline)
COL_IMAGE_NO = "Image No"
COL_CASE_ID = "Case ID"
COL_CATEGORY = "Category"
COL_CONDITION = "Condition"

# Label mapping for ALS vs ALS-FTD
# Concordant → ALS-FTD → y=1 (positive)
# Discordant → ALS → y=0 (negative)
POSITIVE_CATEGORY = "Concordant"  # ALS-FTD
NEGATIVE_CATEGORY = "Discordant"  # ALS
POSITIVE_LABEL = 1  # ALS-FTD
NEGATIVE_LABEL = 0  # ALS

# Categories to include (exclude controls)
INCLUDED_CATEGORIES = [POSITIVE_CATEGORY, NEGATIVE_CATEGORY]

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1  # Binary classification
PRETRAINED = True
DROPOUT_RATE = 0.3

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 8  # Smaller for small dataset

# Training
NUM_EPOCHS = 25
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 1e-4

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2

# Multiple seeds for stability analysis
NUM_SEEDS = 5
RANDOM_SEEDS = [1, 2, 3, 4, 5]
SEEDS = RANDOM_SEEDS  # Alias for compatibility

# =============================================================================
# WINDOWED CARTOGRAPHY CONFIGURATION
# =============================================================================

# Window is centered on best validation loss epoch: [t* - WINDOW_HALF, t* + WINDOW_HALF]
WINDOW_HALF_SIZE = 2  # Window = ±2 epochs around best val_loss epoch (5 epochs total)

# =============================================================================
# PATIENT CATEGORIZATION THRESHOLDS
# =============================================================================

# Easy: high correctness (≥0.8) + high confidence (≥0.8)
EASY_CORRECTNESS_THRESHOLD = 0.8
EASY_CONFIDENCE_THRESHOLD = 0.8

# Medium: 0.6–0.8 correctness + stable (std < 0.1)
MEDIUM_CORRECTNESS_MIN = 0.6
MEDIUM_CORRECTNESS_MAX = 0.8
MEDIUM_STD_THRESHOLD = 0.1

# Ambiguous: 0.4–0.6 correctness + moderate confidence
AMBIGUOUS_CORRECTNESS_MIN = 0.4
AMBIGUOUS_CORRECTNESS_MAX = 0.6

# Hard: low correctness (<0.4)
HARD_CORRECTNESS_THRESHOLD = 0.4

# Outlier: distance from median > 2σ
OUTLIER_SIGMA_THRESHOLD = 2.0

# Borderline: |mean_prob - 0.5| ≤ 0.1
BORDERLINE_MARGIN = 0.1

# Unstable (across seeds): flip_rate > 30% OR std_correctness > 0.2
UNSTABLE_FLIP_RATE_THRESHOLD = 0.3
UNSTABLE_STD_CORRECTNESS_THRESHOLD = 0.2

# Unstable (within images): std_prob_window > 0.15
UNSTABLE_STD_PROB_THRESHOLD = 0.15

# =============================================================================
# IMAGENET NORMALIZATION
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
