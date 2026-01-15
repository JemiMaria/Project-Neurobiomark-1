"""
Interpretability module for CAM generation.
"""

from .layercam import LayerCAM, generate_layercam_outputs
from .guided_gradcam import GuidedGradCAM, generate_guided_gradcam_outputs
from .cam_analysis import (
    compute_cam_consistency,
    compute_class_mean_cams,
    compute_cam_outliers,
    plot_cam_comparison
)

__all__ = [
    'LayerCAM',
    'generate_layercam_outputs',
    'GuidedGradCAM',
    'generate_guided_gradcam_outputs',
    'compute_cam_consistency',
    'compute_class_mean_cams',
    'compute_cam_outliers',
    'plot_cam_comparison',
]
