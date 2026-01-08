"""
Leave-One-Patient-Out (LOPO) Cross-Validation Module.

This module implements LOPO evaluation for medical imaging:
- Hold out one patient at a time
- Train on remaining patients
- Evaluate on held-out patient
- Compute clinical metrics across all folds
- Reliability checks at patient level
"""

from .lopo_runner import run_lopo_evaluation
from .lopo_metrics import compute_clinical_metrics, compute_patient_level_metrics
from .lopo_visualize import create_lopo_visualizations
from .lopo_reliability_checks import run_reliability_checks

__all__ = [
    'run_lopo_evaluation',
    'compute_clinical_metrics',
    'compute_patient_level_metrics',
    'create_lopo_visualizations',
    'run_reliability_checks'
]
