import os
import copy
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.ndimage
import shutil 
from sklearn.metrics import classification_report

# Import required model types for type checking
from .core_models import CustomCNN, ResNetSmall 

from .basic_utils import count_parameters, evaluate_model

# --- Model Loading (Specific to Problem B Checkpoints) ---

def load_models_for_attack(cfg, model_a_class, model_b_class, class_names, ckpt_dir, ckpt_a_name, ckpt_b_name):
    """Load trained model checkpoints from Part A."""
    device = cfg.device

    # Initialize models (Requires model classes from core_models.py)
    model_a = model_a_class(num_classes=cfg.num_classes).to(device)
    model_b = model_b_class(num_classes=cfg.num_classes).to(device)

    ckpt_a = os.path.join(ckpt_dir, ckpt_a_name)
    ckpt_b = os.path.join(ckpt_dir, ckpt_b_name)

    print(f"Loading Model A from {ckpt_a}")
    model_a.load_state_dict(torch.load(ckpt_a, map_location=device))
    print(f"Loading Model B from {ckpt_b}")
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device))
    
    if cfg.target_class_name in class_names:
        target_idx = class_names.index(cfg.target_class_name)
    else:
        raise ValueError(f"Target class '{cfg.target_class_name}' not found. Classes: {class_names}")

    return model_a, model_b, target_idx


# --- Core Attack Algorithms (White-Box) ---

def clamp_tensor(x, mean, std):
    """Clamps the tensor in image space (0 to 1) then renormalizes."""
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    img = x * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    x_norm = (img - mean) / std
    return x_norm

def fgsm_attack(model, x, y, eps, targeted, target_labels=None):
    """Fast Gradient Sign Method (FGSM) implementation."""
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    
    outputs = model(x_adv)
    
    if targeted:
        loss = criterion(outputs, target_labels)
        grad_sign = -torch.sign(torch.autograd.grad(loss, x_adv)[0])
    else:
        loss = criterion(outputs, y)
        grad_sign = torch.sign(torch.autograd.grad(loss, x_adv)[0])
        
    x_adv = x_adv + eps * grad_sign
    
    return clamp_tensor(x_adv.detach(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def pgd_attack(model, x, y, eps, alpha, steps, targeted, target_labels=None):
    """Projected Gradient Descent (PGD) implementation."""
    model.eval()
    x_orig = x.clone().detach()
    
    x_adv = x_orig + (torch.rand_like(x_orig) * 2 * eps - eps)
    x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps)
    x_adv = clamp_tensor(x_adv, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        
        if targeted:
            loss = criterion(outputs, target_labels)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv - alpha * torch.sign(grad)
        else:
            loss = criterion(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + alpha * torch.sign(grad)

        x_adv = x_adv.detach()
        
        eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + eta
        
        x_adv = clamp_tensor(x_adv, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    return x_adv.detach()


# --- Experiment Runner ---

def generate_adversarial_set(
    model, val_loader, attack_type, eps, targeted, target_class_idx, num_samples, alpha=None, steps=None
):
    """Generates and evaluates adversarial samples from the validation set."""
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    collected = []
    
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            logits_clean = model(inputs)
            preds_clean = logits_clean.argmax(dim=1)
        mask_correct = preds_clean.eq(targets)
        if mask_correct.sum().item() == 0: continue

        inputs_c = inputs[mask_correct]
        targets_c = targets[mask_correct]
        logits_clean = logits_clean[mask_correct]
        preds_clean = preds_clean[mask_correct]

        target_labels = torch.full_like(targets_c, target_class_idx, device=device) if targeted else None

        if attack_type == "fgsm":
            adv = fgsm_attack(model, inputs_c, targets_c, eps, targeted, target_labels)
        elif attack_type == "pgd":
            adv = pgd_attack(model, inputs_c, targets_c, eps, alpha, steps, targeted, target_labels)
        else: raise ValueError("Unknown attack type")

        with torch.no_grad():
            logits_adv = model(adv)
            preds_adv = logits_adv.argmax(dim=1)

        for i in range(inputs_c.size(0)):
            x0 = inputs_c[i].detach().cpu()
            xa = adv[i].detach().cpu()
            y_true = int(targets_c[i].item())
            y_clean = int(preds_clean[i].item())
            y_adv = int(preds_adv[i].item())
            lc = logits_clean[i].detach().cpu().numpy()
            la = logits_adv[i].detach().cpu().numpy()

            diff = (xa - x0).view(-1)
            l2 = torch.norm(diff, p=2).item()
            linf = torch.norm(diff, p=float("inf")).item()

            success = (y_adv == target_class_idx) if targeted else (y_adv != y_true)

            collected.append({
                "x_clean": x0, "x_adv": xa, "y_true": y_true, "y_pred_clean": y_clean,
                "y_pred_adv": y_adv, "logits_clean": lc, "logits_adv": la, "l2": l2,
                "linf": linf, "success": success, "attack_type": attack_type,
                "targeted": targeted, "target_idx": target_class_idx,
            })

            if len(collected) >= num_samples: return collected

    return collected


def transferability_test(examples, model_other, targeted, target_idx):
    """Test attack transferability between models."""
    device = next(model_other.parameters()).device
    model_other.to(device)
    model_other.eval()

    total = 0
    success = 0

    for ex in examples:
        xa = ex["x_adv"].unsqueeze(0).to(device)
        y_true = ex["y_true"]
        with torch.no_grad():
            logits = model_other(xa)
            pred = logits.argmax(dim=1).item()

        if targeted: suc = (pred == target_idx)
        else: suc = (pred != y_true)

        success += int(suc)
        total += 1

    rate = success / max(total, 1)
    return rate

# Adversarial Training Function

def train_robust_model(model, model_name, train_loader, val_loader, cfg):
    """
    Trains a model using Adversarial Training (PGD-AT).
    PGD examples are generated online and fed to the optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    
    start_time_total = time.time()
    
    print(f"\n--- Starting ADVERSARIAL TRAINING for {model_name} ---")
    
    for epoch in range(cfg.num_epochs):
        model.train()
        total_train, correct_train = 0, 0
        
        # Start Time
        start_time_epoch = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            
            # 1. GENERATE ADVERSARIAL EXAMPLES ONLINE (Untargeted PGD)
            # pgd_attack must be available here
            x_adv = pgd_attack(
                model, inputs, labels, cfg.PGD_EPS, cfg.PGD_ALPHA, cfg.PGD_STEPS, 
                targeted=False, target_labels=None
            )
            
            # 2. Train on Adversarial Data
            optimizer.zero_grad()
            outputs = model(x_adv)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 3. Track clean accuracy (using the adversarial outputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # End Time
        epoch_time = time.time() - start_time_epoch
        
        # Evaluation (Using clean data for standard evaluation)
        # evaluate_model must be available here
        final_loss, val_accuracy, all_preds, all_labels = evaluate_model(model, val_loader, criterion, cfg.device)
        train_accuracy = 100 * correct_train / total_train

        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = os.path.join(cfg.out_dir, f"{model_name}-robust.pt")
            torch.save(model.state_dict(), checkpoint_path)

    # Final Time
    total_training_time = time.time() - start_time_total
    
    return {
        'model_name': model_name,
        'total_training_time': total_training_time,
        'final_val_accuracy': best_accuracy,
    }

# --- CORE VISUALIZATION UTILITIES ---

def denormalize(tensor):
    """Denormalizes a single or batched (C, H, W) tensor to (H, W, C) numpy array."""
    # Constants defined inside the function as requested
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    # Squeeze to (C, H, W) if it's batched (1, C, H, W)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    img_np = tensor.cpu().detach().permute(1, 2, 0).numpy()
    
    # Denormalize
    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
    
    # Clip to [0, 1] range for plotting
    return np.clip(img_np, 0, 1)


def get_top_preds_formatted(logits_array, class_names, top_k=2):
    """Converts logits (as NumPy array) to softmax probabilities and returns the top K as a formatted string."""
    
    # Convert NumPy array to PyTorch tensor
    logits_tensor = torch.from_numpy(logits_array)
    
    # Squeeze the tensor to be (num_classes) if it was batched (1, num_classes)
    logits_tensor = logits_tensor.squeeze()
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits_tensor, dim=0) 
    top_p, top_class_idx = probs.topk(top_k, dim=0)
    
    # Format string
    output = []
    for p, idx in zip(top_p.tolist(), top_class_idx.tolist()):
        output.append(f"{class_names[idx]}: {p:.3f}")
    return "\n".join(output)


def plot_comparison_row(ax_row, img_tensor_batched, model, class_names, cfg_device, title, target_layer_name='features.2'):
    """Plots the image, saliency, and Grad-CAM for a given model and input."""
    # Local imports as requested
    from .interpretability import GradCAM, visualize_saliency_map
    from .core_models import CustomCNN, ResNetSmall # Re-import here for isinstance check
    
    # 1. Prediction for title
    with torch.no_grad():
        output = model(img_tensor_batched)
    predicted_label_idx = output.argmax(dim=1).item()
    predicted_class = class_names[predicted_label_idx]
    
    # 2. Denormalize Image for Display
    img_np = denormalize(img_tensor_batched)
    
    # --- Col 1: Original Image ---
    ax_row[0].imshow(img_np)
    ax_row[0].set_title(f'{title}\nPred: {predicted_class}', fontsize=12)
    ax_row[0].axis('off')
    
    # --- Col 2: Saliency Map ---
    saliency_map = visualize_saliency_map(
        model, img_tensor_batched, 
        target_class=predicted_label_idx, 
        device=cfg_device
    )
    
    ax_row[1].imshow(img_np) # Plot original image as background
    ax_row[1].imshow(saliency_map, cmap='jet', alpha=0.5) # Overlay saliency map
    ax_row[1].set_title(f'Saliency Map (Target: {predicted_class})', fontsize=12)
    ax_row[1].axis('off')
    
    # --- Col 3: Grad-CAM Heatmap ---
    
    # FIX 1: Dynamically determine the target layer based on model type
    if isinstance(model, ResNetSmall):
        target_layer_name = 'layer3'
    elif isinstance(model, CustomCNN):
        target_layer_name = 'features.2' 
    else:
        # Fallback for unexpected models
        target_layer_name = 'layer4'
        
    cam = GradCAM(model, target_layer_name)
    # GradCAM is assumed to use the __call__ method, returning a tuple (heatmap_np, ...)
    heatmap_np = cam(img_tensor_batched, target_class=predicted_label_idx)[0]
    del cam # Cleanup hooks immediately
    
    ax_row[2].imshow(img_np) # Plot original image as background
    
    # Resize heatmap to match image dimensions (e.g., 64x64). 
    h_target, w_target = img_np.shape[:2]
    h_cam, w_cam = heatmap_np.shape
    resized_heatmap = scipy.ndimage.zoom(heatmap_np, (h_target / h_cam, w_target / w_cam), order=1)
    
    ax_row[2].imshow(resized_heatmap, cmap='jet', alpha=0.5) # Overlay Grad-CAM
    ax_row[2].set_title(f'Grad-CAM (Target: {predicted_class})', fontsize=12)
    ax_row[2].axis('off')


def plot_single_adversarial_sample(sample, model, model_name, cfg, class_names, output_filepath=None):
    """
    Creates a 2x3 visualization comparing an original image and a successful 
    adversarial example, including Saliency and Grad-CAM maps, with a 
    summary text box at the bottom (footer).
    
    If output_filepath is None, the plot is displayed inline using plt.show().
    """
    
    # 2. Extract and prepare tensors for plotting
    sample_img_tensor = sample['x_clean'].to(cfg.device)
    x_adv_tensor = sample['x_adv'].to(cfg.device)
    actual_label_idx = sample['y_true'] 

    # Create batched tensors for model input (required for model(input))
    input_tensor_batched = sample_img_tensor.unsqueeze(0)
    x_adv_batched = x_adv_tensor.unsqueeze(0)

    # 3. Extract Quantitative Metrics for Annotation
    clean_top_preds = get_top_preds_formatted(sample['logits_clean'], class_names)
    adv_top_preds = get_top_preds_formatted(sample['logits_adv'], class_names)

    norm_linf = sample['linf']
    norm_l2 = sample['l2']
    attack_success = sample['success']

    # 4. Assemble Three Column Text Sections
    target_status = 'Targeted' if sample['targeted'] else 'Untargeted'
    status_text = "FAILURE" if not attack_success else "SUCCESS"

    metrics_summary_text = f"""
Attack: {sample['attack_type']}_{target_status}
(Status: {status_text})
L_inf Norm: {norm_linf:.5f}
L2 Norm: {norm_l2:.5f}
"""
    original_preds_text = f"""
Original Predictions
{clean_top_preds}
"""

    adversarial_preds_text = f"""
Adversarial Predictions
{adv_top_preds}
"""

    # 5. Create Figure and Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10)) 
    fig.suptitle(f'Interpretability Comparison: {model_name} | {sample["attack_type"].upper()} {target_status} Attack', fontsize=16)

    # Top Row: Original Image
    plot_comparison_row(
        ax_row=axes[0], 
        img_tensor_batched=input_tensor_batched, 
        model=model, 
        class_names=class_names, 
        cfg_device=cfg.device,
        title=f'Original Image (True: {class_names[actual_label_idx]})'
    )

    # Bottom Row: Adversarial Example
    plot_comparison_row(
        ax_row=axes[1], 
        img_tensor_batched=x_adv_batched, 
        model=model, 
        class_names=class_names, 
        cfg_device=cfg.device,
        title=f'Adversarial Example'
    )

    # 6. Add the Metrics Text Box as a Footer in three columns
    
    y_coord = 0.02 # Normalized y-coordinate for the footer

    # Col 1: Attack Summary (Left)
    plt.figtext(0.15, y_coord, metrics_summary_text, 
        fontsize=14, 
        ha='left', va='bottom'
    )

    # Col 2: Original Predictions (Center)
    plt.figtext(0.34, y_coord, original_preds_text, 
        fontsize=14, 
        ha='left', va='bottom'
    )

    # Col 3: Adversarial Predictions (Right)
    plt.figtext(0.5, y_coord, adversarial_preds_text, 
        fontsize=14, 
        ha='left', va='bottom'
    )

    plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.2, wspace=0.1)

    
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
        plt.close(fig) # Close figure to free memory
    else:
        plt.show()


def plot_all_adversarial_samples_to_dir(all_results, model_a, model_b, cfg, class_names, sub_dir_name = "adversarial_plots"):
    """
    Iterates through ALL adversarial examples (successful and failures), generates a plot for 
    each one, and saves them to a subdirectory in cfg.out_dir. Logs all attempts.
    """
    
    target_dir = os.path.join(cfg.out_dir, sub_dir_name)
    
    # 1. Delete the directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"Deleted existing directory: {target_dir}")

    # 2. Create the clean directory
    os.makedirs(target_dir, exist_ok=True)
    
    # 3. FIX: Ensure all figures are closed before and after the batch save
    # This prevents figures from displaying in the notebook output
    plt.close('all') 
    
    for key, result in all_results.items():
        attack_name, atype = key.split('_')
        
        # Determine Target Abbreviation
        is_targeted = result.get('Adv_A Examples', [{}])[0].get('targeted', False) if result.get('Adv_A Examples') else False
        target_tag = 'T' if is_targeted else 'UT'
        
        # --- Process Model A Examples ---
        model_name = "ModelA"
        model = model_a
        for i, sample in enumerate(result.get('Adv_A Examples', [])):
            is_successful = sample.get('success')
            status_tag_abbr = 'S' if is_successful else 'F' 
            status_tag_full = 'SUCCESS' if is_successful else 'FAILURE'
            true_class = class_names[sample['y_true']]
            
            log_message = f"ATTEMPT {i+1} ({model_name} / {attack_name}_{target_tag}) | True Class: {true_class} | Status: {status_tag_full}"
            print(log_message)

            # Construct unique filename
            filename = f"{attack_name}_{target_tag}_{model_name}_{status_tag_abbr}_Sample{i:02d}.png"
            filepath = os.path.join(target_dir, filename) 
            plot_single_adversarial_sample(
                sample=sample,
                model=model,
                model_name=model_name,
                cfg=cfg,
                class_names=class_names,
                output_filepath=filepath
            )
        
        # --- Process Model B Examples ---
        model_name = "ModelB"
        model = model_b
        for i, sample in enumerate(result.get('Adv_B Examples', [])):
            is_successful = sample.get('success')
            status_tag_abbr = 'S' if is_successful else 'F' 
            status_tag_full = 'SUCCESS' if is_successful else 'FAILURE'
            true_class = class_names[sample['y_true']]
            
            log_message = f"ATTEMPT {i+1} ({model_name} / {attack_name}_{target_tag}) | True Class: {true_class} | Status: {status_tag_full}"
            print(log_message)

            # Construct unique filename
            filename = f"{attack_name}_{target_tag}_{model_name}_{status_tag_abbr}_Sample{i:02d}.png"
            filepath = os.path.join(target_dir, filename) 
            plot_single_adversarial_sample(
                sample=sample,
                model=model,
                model_name=model_name,
                cfg=cfg,
                class_names=class_names,
                output_filepath=filepath
            )

    print(f"\nAll adversarial plots (SUCCESS and FAILURE) saved to {target_dir}")