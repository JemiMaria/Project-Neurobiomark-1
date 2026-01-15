"""
Configuration for Grouped CV EfficientNetB0 Pipeline

Central configuration file for all pipeline settings.
Modify these values to customize the pipeline behavior.
"""

from pathlib import Path
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (parent of grouped_cv_eb0)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths - Using same paths as LOPO pipeline
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "raw_images"  # Folder containing 1.tif, 2.tif, etc.
METADATA_PATH = DATA_DIR / "image_keys.xlsx"  # Same Excel as LOPO pipeline

# Excel column names (matching LOPO pipeline)
COL_IMAGE_NO = "Image No"      # Image number (1, 2, 3, ...)
COL_CASE_ID = "Case ID"        # e.g., "SD028-12 Concord BA46"
COL_CONDITION = "Condition"    # "Case" (ALS=1) or "Control" (0)

# Class labels
POSITIVE_CLASS = "Case"        # Label for ALS (label=1)
NEGATIVE_CLASS = "Control"     # Label for Control (label=0)

# Output directories
OUTPUT_DIR = Path(__file__).parent / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"
THRESHOLDS_DIR = OUTPUT_DIR / "thresholds"
EVAL_DIR = OUTPUT_DIR / "eval"
OPTUNA_DIR = OUTPUT_DIR / "optuna"
CAMS_DIR = OUTPUT_DIR / "cams"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# Create directories
for d in [OUTPUT_DIR, SPLITS_DIR, THRESHOLDS_DIR, EVAL_DIR, OPTUNA_DIR, 
          CAMS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================

N_FOLDS = 5
RANDOM_SEED = 42

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# EfficientNetB0 settings
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1  # Binary classification (1 logit for BCEWithLogitsLoss)
PRETRAINED = True

# Dropout for classifier head
DROPOUT_RATE = 0.3

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Image settings
IMAGE_SIZE = 224  # EfficientNetB0 default input size
BATCH_SIZE = 16

# Training phases
FREEZE_EPOCHS = 3  # Phase A: head-only training
FINETUNE_EPOCHS = 20  # Phase B: backbone fine-tuning
TOTAL_EPOCHS = FREEZE_EPOCHS + FINETUNE_EPOCHS

# Learning rates
HEAD_LR = 1e-3  # Learning rate for classifier head
BACKBONE_LR_MULTIPLIER = 0.1  # Backbone LR = HEAD_LR * multiplier

# Optimizer settings
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0  # Set > 0 for label smoothing (e.g., 0.1)

# Early stopping
PATIENCE = 7
MIN_DELTA = 1e-4

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3

# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================

# Augmentation level: 'light', 'medium', 'strong'
AUG_LEVEL = "medium"

# Use stain augmentation (requires RandStainNA)
USE_STAIN_AUG = False

# Augmentation parameters (tunable by Optuna)
MAX_ROTATE = 20  # degrees
P_NOISE = 0.1  # probability of noise augmentation
USE_ELASTIC = False  # elastic deformation (rare/mild)

# =============================================================================
# THRESHOLD TUNING CONFIGURATION
# =============================================================================

# Clinical constraint: minimum sensitivity
MIN_SENSITIVITY = 0.70

# Threshold search range
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEPS = 81  # Number of thresholds to evaluate

# =============================================================================
# OPTUNA CONFIGURATION
# =============================================================================

OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = None  # Seconds, or None for no timeout
OPTUNA_PRUNER_WARMUP = 5  # Epochs before pruning starts
OPTUNA_STUDY_NAME = "als_classification_cv"

# Search space bounds
OPTUNA_HEAD_LR_RANGE = (1e-4, 1e-2)
OPTUNA_BACKBONE_LR_MULT_RANGE = (0.01, 0.3)
OPTUNA_WEIGHT_DECAY_RANGE = (1e-6, 1e-3)
OPTUNA_DROPOUT_RANGE = (0.1, 0.5)
OPTUNA_LABEL_SMOOTHING_RANGE = (0.0, 0.2)
OPTUNA_FREEZE_EPOCHS_RANGE = (3, 5)

# =============================================================================
# CAM CONFIGURATION
# =============================================================================

# Target layer for CAM (EfficientNetB0 last conv layer)
CAM_TARGET_LAYER = "features.8"  # Last block of EfficientNetB0

# Top percentage of activated pixels for consistency metrics
CAM_TOP_PERCENTILE = 10

# =============================================================================
# IMAGENET NORMALIZATION (for EfficientNetB0)
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
