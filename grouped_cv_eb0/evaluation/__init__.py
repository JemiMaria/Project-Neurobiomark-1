"""
Evaluation module for grouped CV pipeline.
"""

from .metrics import (
    compute_patient_level_metrics,
    compute_image_level_predictions,
    aggregate_to_patient_level,
    compute_fold_metrics,
    compute_cv_summary,
    compute_wilson_ci
)
from .threshold import tune_threshold_per_fold, find_optimal_threshold
from .calibration import compute_calibration_metrics, plot_reliability_diagram, save_calibration_results
from .reliability import (
    compute_borderline_patients,
    compute_patient_instability,
    generate_evaluation_report
)

__all__ = [
    'compute_patient_level_metrics',
    'compute_image_level_predictions',
    'aggregate_to_patient_level',
    'compute_fold_metrics',
    'compute_cv_summary',
    'compute_wilson_ci',
    'tune_threshold_per_fold',
    'find_optimal_threshold',
    'compute_calibration_metrics',
    'plot_reliability_diagram',
    'save_calibration_results',
    'compute_borderline_patients',
    'compute_patient_instability',
    'generate_evaluation_report',
]
