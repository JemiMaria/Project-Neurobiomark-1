"""
Guided Grad-CAM implementation for model interpretability.

Guided Grad-CAM combines:
1. Guided Backpropagation (fine-grained pixel-level attribution)
2. Grad-CAM (coarse localization)

The fusion produces high-resolution class-discriminative visualizations.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017)
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


class GuidedBackprop:
    """
    Guided Backpropagation for fine-grained attribution.
    
    Modifies ReLU backward pass to only propagate positive gradients
    where the input was also positive.
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or DEVICE
        self.model.eval()
        self.model.to(self.device)
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to modify ReLU backward pass."""
        def relu_backward_hook(module, grad_input, grad_output):
            # Only propagate positive gradients
            return (torch.clamp(grad_output[0], min=0.0),)
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_full_backward_hook(relu_backward_hook)
                self.hooks.append(hook)
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Compute guided backpropagation.
        
        Args:
            input_tensor: Input image [1, C, H, W]
            class_idx: Target class
            
        Returns:
            np.ndarray: Gradient image [H, W, C]
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward
        if class_idx is None:
            class_idx = 0
        target = output[0, class_idx] if output.dim() > 1 else output[0]
        target.backward()
        
        # Get gradients
        gradients = input_tensor.grad[0].cpu().numpy()
        gradients = np.transpose(gradients, (1, 2, 0))
        
        return gradients
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()


class GradCAM:
    """
    Grad-CAM for coarse localization.
    
    Uses gradients flowing into the final convolutional layer
    to produce a coarse localization map.
    """
    
    def __init__(self, model, target_layer_name=None, device=None):
        self.model = model
        self.device = device or DEVICE
        self.target_layer_name = target_layer_name or CAM_TARGET_LAYER
        
        self.model.eval()
        self.model.to(self.device)
        
        self.target_layer = self._get_layer(self.target_layer_name)
        self.activations = None
        self.gradients = None
        
        self._register_hooks()
    
    def _get_layer(self, layer_name):
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Compute Grad-CAM.
        
        Args:
            input_tensor: Input image [1, C, H, W]
            class_idx: Target class
            
        Returns:
            np.ndarray: CAM heatmap [H, W]
        """
        input_tensor = input_tensor.to(self.device)
        
        # Forward
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward
        if class_idx is None:
            class_idx = 0
        target = output[0, class_idx] if output.dim() > 1 else output[0]
        target.backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam


class GuidedGradCAM:
    """
    Guided Grad-CAM: Fusion of Guided Backprop and Grad-CAM.
    
    Produces high-resolution, class-discriminative visualizations
    by element-wise multiplication of the two methods.
    """
    
    def __init__(self, model, target_layer_name=None, device=None):
        self.model = model
        self.device = device or DEVICE
        
        self.gradcam = GradCAM(model, target_layer_name, device)
        self.guided_bp = GuidedBackprop(model, device)
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Compute Guided Grad-CAM.
        
        Args:
            input_tensor: Input image [1, C, H, W]
            class_idx: Target class
            
        Returns:
            tuple: (guided_gradcam, gradcam, guided_bp)
        """
        # Get Grad-CAM
        gradcam = self.gradcam(input_tensor, class_idx)
        
        # Get Guided Backprop
        guided_bp = self.guided_bp(input_tensor.clone().detach(), class_idx)
        
        # Fusion: element-wise multiplication
        # Expand gradcam to 3 channels
        gradcam_3d = np.stack([gradcam] * 3, axis=-1)
        
        # Normalize guided backprop
        guided_bp_norm = guided_bp - guided_bp.min()
        if guided_bp_norm.max() > 0:
            guided_bp_norm = guided_bp_norm / guided_bp_norm.max()
        
        # Multiply
        guided_gradcam = guided_bp_norm * gradcam_3d
        
        # Normalize result
        if guided_gradcam.max() > 0:
            guided_gradcam = guided_gradcam / guided_gradcam.max()
        
        return guided_gradcam, gradcam, guided_bp
    
    def generate_overlay(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """Generate CAM overlay on original image."""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalize image tensor for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    tensor = tensor.clamp(0, 1)
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def generate_guided_gradcam_outputs(model, dataloader, output_dir=None, device=None, max_images=None):
    """
    Generate Guided Grad-CAM outputs for all images.
    
    Saves:
    - Per-image: original, Grad-CAM, Guided BP, Guided Grad-CAM
    - Per-patient averaged CAM maps
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        output_dir: Output directory
        device: torch device
        max_images: Maximum images to process
        
    Returns:
        dict: {patient_id: list of CAM arrays}
    """
    if output_dir is None:
        output_dir = CAMS_DIR / "guided_gradcam"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = DEVICE
    
    print(f"\n{'='*60}")
    print("GENERATING GUIDED GRAD-CAM OUTPUTS")
    print(f"{'='*60}")
    
    # Initialize
    guided_gradcam = GuidedGradCAM(model, device=device)
    
    # Storage
    patient_cams = {}
    patient_images = {}
    
    n_processed = 0
    
    for batch in tqdm(dataloader, desc="Generating Guided Grad-CAM"):
        images = batch['image']
        patient_ids = batch['patient_id']
        image_paths = batch['image_path']
        
        for i in range(len(images)):
            if max_images is not None and n_processed >= max_images:
                break
            
            img_tensor = images[i:i+1]
            pid = patient_ids[i]
            img_path = Path(image_paths[i])
            
            # Generate CAMs
            ggcam, gcam, gbp = guided_gradcam(img_tensor)
            
            # Denormalize
            img_denorm = denormalize_image(images[i])
            
            # Generate overlay
            overlay = guided_gradcam.generate_overlay(img_denorm, gcam)
            
            # Save
            img_name = img_path.stem
            patient_dir = output_dir / "per_image" / str(pid)
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(img_denorm)
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gcam, cmap='jet')
            axes[0, 1].set_title("Grad-CAM")
            axes[0, 1].axis('off')
            
            # Guided BP (show as grayscale of magnitude)
            gbp_gray = np.mean(np.abs(gbp), axis=-1)
            if gbp_gray.max() > 0:
                gbp_gray = gbp_gray / gbp_gray.max()
            axes[1, 0].imshow(gbp_gray, cmap='gray')
            axes[1, 0].set_title("Guided Backprop")
            axes[1, 0].axis('off')
            
            # Guided Grad-CAM
            ggcam_gray = np.mean(ggcam, axis=-1)
            axes[1, 1].imshow(ggcam_gray, cmap='jet')
            axes[1, 1].set_title("Guided Grad-CAM")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(patient_dir / f"{img_name}_guided_gradcam.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            # Store for averaging
            if pid not in patient_cams:
                patient_cams[pid] = []
                patient_images[pid] = []
            patient_cams[pid].append(gcam)
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
        avg_overlay = guided_gradcam.generate_overlay(avg_image, avg_cam)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(avg_image)
        axes[0].set_title(f"Patient {pid} (Avg)")
        axes[0].axis('off')
        
        axes[1].imshow(avg_cam, cmap='jet')
        axes[1].set_title("Avg Grad-CAM")
        axes[1].axis('off')
        
        axes[2].imshow(avg_overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(patient_avg_dir / f"patient_{pid}_guided_gradcam_avg.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"\n  ✓ Processed {n_processed} images")
    print(f"  ✓ Generated {len(patient_cams)} patient-averaged CAMs")
    print(f"  ✓ Outputs saved to: {output_dir}")
    
    return patient_cams
