"""
LayerCAM implementation for model interpretability.

LayerCAM is a gradient-weighted class activation mapping technique that
produces more fine-grained localization than Grad-CAM.

Reference: Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps
for Localization" (2021)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVICE, CAMS_DIR, CAM_TARGET_LAYER, IMAGENET_MEAN, IMAGENET_STD


class LayerCAM:
    """
    LayerCAM: Gradient-weighted class activation mapping.
    
    Produces finer-grained localization than Grad-CAM by using
    element-wise multiplication of positive gradients with activations.
    """
    
    def __init__(self, model, target_layer_name=None, device=None):
        """
        Initialize LayerCAM.
        
        Args:
            model: PyTorch model (EfficientNetB0)
            target_layer_name: Name of target layer (default from config)
            device: torch device
        """
        self.model = model
        self.device = device or DEVICE
        self.target_layer_name = target_layer_name or CAM_TARGET_LAYER
        
        self.model.eval()
        self.model.to(self.device)
        
        # Get target layer
        self.target_layer = self._get_layer(self.target_layer_name)
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _get_layer(self, layer_name):
        """Get layer by name (e.g., 'features.8')."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Generate LayerCAM heatmap for input.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class (None for predicted class)
            
        Returns:
            np.ndarray: CAM heatmap [H, W] in range [0, 1]
        """
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = 0  # Binary classification
        
        # Backward pass
        target = output[0, class_idx] if output.dim() > 1 else output[0]
        target.backward()
        
        # LayerCAM computation
        # ReLU on gradients (keep only positive gradients)
        gradients = F.relu(self.gradients)
        
        # Element-wise multiplication
        cam = gradients * self.activations
        
        # Sum over channels
        cam = cam.sum(dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam
    
    def generate_overlay(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Generate CAM overlay on original image.
        
        Args:
            image: Original image [H, W, C] in range [0, 255]
            cam: CAM heatmap [H, W] in range [0, 1]
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            np.ndarray: Overlay image [H, W, C]
        """
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        np.ndarray: Image [H, W, C] in range [0, 255]
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    tensor = tensor.clamp(0, 1)
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def generate_layercam_outputs(model, dataloader, output_dir=None, device=None, max_images=None):
    """
    Generate LayerCAM outputs for all images in dataloader.
    
    Saves:
    - Per-image heatmap overlays
    - Per-patient averaged CAM maps
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        output_dir: Output directory
        device: torch device
        max_images: Maximum images to process (None for all)
        
    Returns:
        dict: {patient_id: list of CAM arrays}
    """
    if output_dir is None:
        output_dir = CAMS_DIR / "layercam"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = DEVICE
    
    print(f"\n{'='*60}")
    print("GENERATING LAYERCAM OUTPUTS")
    print(f"{'='*60}")
    
    # Initialize LayerCAM
    layercam = LayerCAM(model, device=device)
    
    # Storage for patient-level averaging
    patient_cams = {}
    patient_images = {}
    
    n_processed = 0
    
    for batch in tqdm(dataloader, desc="Generating LayerCAM"):
        images = batch['image']
        patient_ids = batch['patient_id']
        image_paths = batch['image_path']
        
        for i in range(len(images)):
            if max_images is not None and n_processed >= max_images:
                break
            
            img_tensor = images[i:i+1]
            pid = patient_ids[i]
            img_path = Path(image_paths[i])
            
            # Generate CAM
            cam = layercam(img_tensor)
            
            # Denormalize image for visualization
            img_denorm = denormalize_image(images[i])
            
            # Generate overlay
            overlay = layercam.generate_overlay(img_denorm, cam)
            
            # Save individual image
            img_name = img_path.stem
            
            # Create patient subfolder
            patient_dir = output_dir / "per_image" / str(pid)
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original and overlay
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_denorm)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title("LayerCAM")
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(patient_dir / f"{img_name}_layercam.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            # Store for patient averaging
            if pid not in patient_cams:
                patient_cams[pid] = []
                patient_images[pid] = []
            patient_cams[pid].append(cam)
            patient_images[pid].append(img_denorm)
            
            n_processed += 1
        
        if max_images is not None and n_processed >= max_images:
            break
    
    # Generate patient-averaged CAMs
    print("\n  Generating patient-averaged CAMs...")
    patient_avg_dir = output_dir / "per_patient"
    patient_avg_dir.mkdir(parents=True, exist_ok=True)
    
    for pid, cams in patient_cams.items():
        avg_cam = np.mean(cams, axis=0)
        avg_image = np.mean(patient_images[pid], axis=0).astype(np.uint8)
        avg_overlay = layercam.generate_overlay(avg_image, avg_cam)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(avg_image)
        axes[0].set_title(f"Patient {pid} (Avg Image)")
        axes[0].axis('off')
        
        axes[1].imshow(avg_cam, cmap='jet')
        axes[1].set_title("Avg LayerCAM")
        axes[1].axis('off')
        
        axes[2].imshow(avg_overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(patient_avg_dir / f"patient_{pid}_layercam_avg.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"\n  ✓ Processed {n_processed} images")
    print(f"  ✓ Generated {len(patient_cams)} patient-averaged CAMs")
    print(f"  ✓ Outputs saved to: {output_dir}")
    
    return patient_cams
