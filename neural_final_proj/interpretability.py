import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn

def visualize_saliency_map(model, input_tensor, target_class=None, device=None):
    """
    Generates the Saliency Map for a given input tensor.
    """
    model.eval()
    
    target_device = device if device is not None else input_tensor.device
    input_tensor = input_tensor.clone().detach().to(target_device) 

    input_tensor.requires_grad_()
    
    output = model(input_tensor)
    if target_class is None: target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    target_logit = output[0, target_class]
    target_logit.backward()
    
    saliency = input_tensor.grad.data.abs().max(dim=1, keepdim=True)[0]
    
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

class GradCAM:
    """Implements the Grad-CAM visualization technique."""
    def __init__(self, model, target_layer_name):
        self.model = model.eval()
        self.target_layer = self._find_target_layer(model, target_layer_name)
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _find_target_layer(self, model, name):
        def get_module_by_name(module, name_list):
            if not name_list:
                return module
            sub_module = getattr(module, name_list[0])
            return get_module_by_name(sub_module, name_list[1:])

        return get_module_by_name(model, name.split('.'))

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_a = self.target_layer.register_forward_hook(forward_hook)
        self.hook_g = self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
            self.model.zero_grad()
            # input_tensor = input_tensor.unsqueeze(0)
            
            output = self.model(input_tensor)
            
            predicted_label = output.argmax(dim=1).item()
            if target_class is None:
                target_class = predicted_label
            
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            
            output.backward(gradient=one_hot, retain_graph=True) 

            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = nn.functional.relu(cam)
            
            cam = nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam, predicted_label
    
    def close(self):
        """Removes the forward and backward hooks registered during initialization."""
        self.hook_a.remove()
        self.hook_g.remove()

# --- Find Viz Functions ---

def find_viz_indices(model_preds, model_name, target_labels_indices, true_labels, class_names):
    """
    Searches for one correct and one misclassified index per target class
    and returns a minimal map of (title: index) for plotting.
    """
    
    true_labels_np = np.array(true_labels)
    model_preds_np = np.array(model_preds)
    
    viz_map = {}
    
    print(f"\n--- Searching Samples using {model_name} Results ---")

    for target_label_idx in target_labels_indices:
        target_class_name = class_names[target_label_idx]
    
        is_target_class = (true_labels_np == target_label_idx)
        
        is_correct_pred = (model_preds_np == true_labels_np)
        
        correct_indices = np.where(is_target_class & is_correct_pred)[0]
        correct_found_idx = correct_indices[0] if correct_indices.size > 0 else None
        
        is_misclassified = (model_preds_np != true_labels_np)
        misclassified_indices = np.where(is_target_class & is_misclassified)[0]
        incorrect_found_idx = misclassified_indices[0] if misclassified_indices.size > 0 else None
        
        if correct_found_idx is not None:
            title = f"{model_name}: Correct: {target_class_name}"
            viz_map[title] = correct_found_idx
            print(f"FOUND: {target_class_name} - CORRECT (Index: {correct_found_idx})")

        if incorrect_found_idx is not None:
            predicted_class_name = class_names[model_preds_np[incorrect_found_idx]]
            title = f"{model_name}: Misclassified: {target_class_name} -> {predicted_class_name}"
            viz_map[title] = incorrect_found_idx
            print(f"FOUND: {target_class_name} - MISCLASSIFIED (Index: {incorrect_found_idx}) -> PREDICTED: {predicted_class_name}")

    return viz_map

# --- Plotting Functions ---

def plot_interpretability_results(
    model, 
    model_name, 
    input_tensor, 
    sample_img_tensor, 
    actual_label, 
    class_names, 
    models_dir
):
    """
    Loads the best model checkpoint, generates Saliency Map and Grad-CAM, 
    and plots the three visualizations for a single input image.
    """
    # Load Model Checkpoint
    model_path = os.path.join(models_dir, f"{model_name}-original.pt")

    # Load model state onto the CPU (as DEVICE is 'cpu')
    load_result = model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
        print(f"WARNING: Model {model_name} had mismatched keys during load.")
    
    model.eval()

    # Generate Interpretability Maps
    saliency = visualize_saliency_map(model, input_tensor)
    
    grad_cam_instance = GradCAM(model, model.target_layer_name)
    cam, predicted_label_idx = grad_cam_instance(input_tensor)
    
    # Process Labels
    predicted_class = class_names[predicted_label_idx]
    actual_class = class_names[actual_label]

    # --- ADDED: Determine Status for Title ---
    if actual_label == predicted_label_idx:
        status_line = "CORRECT"
    else:
        status_line = "MISCLASSIFIED"

    # Denormalize Image for Display
    # Standard ImageNet mean/std used in the transforms
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Permute (C, H, W) to (H, W, C) and denormalize
    img_np = sample_img_tensor.permute(1, 2, 0).numpy() * std + mean
    img_np = np.clip(img_np, 0, 1)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original: {actual_class} | Prediction: {predicted_class}", fontsize=14)
    axes[0].axis('off')

    # Saliency Map
    axes[1].imshow(img_np)
    axes[1].imshow(saliency, cmap='hot', alpha=0.5) 
    axes[1].set_title(f"Saliency Map (Pixel Gradient)", fontsize=14)
    axes[1].axis('off')

    # Grad-CAM
    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap='jet', alpha=0.6) 
    axes[2].set_title(f"Grad-CAM (Feature Activation)", fontsize=14)
    axes[2].axis('off')
    
    fig.suptitle(f"Model Interpretability: {model_name}", fontsize=18, y=1.02)
    plt.figtext(x=0.5, y=-0.02,s=f"Status: {status_line}", fontsize=14, ha="center")
    plt.tight_layout()
    plt.show()