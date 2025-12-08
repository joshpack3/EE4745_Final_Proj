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
from sklearn.metrics import classification_report

from .project_utils import count_parameters, evaluate_model

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

# --- Visualization/Plotting Functions (Problem B specific saving) ---

def show_and_save_overlay(image_tensor, heatmap, out_path, alpha=0.4):
    """Helper to plot heatmap over image and save the result."""
    import scipy.ndimage
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = image_tensor.cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0)) # Shape (H, W, C)

    img_h, img_w, _ = img.shape
    heatmap = scipy.ndimage.zoom(heatmap, (img_h / heatmap.shape[0], img_w / heatmap.shape[1]), order=1)

    heat = plt.cm.jet(heatmap)[..., :3]
    overlay = alpha * heat + (1 - alpha) * img
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(overlay)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_example_images(examples, class_names, out_dir, prefix):
    """Saves the clean vs. adversarial image pairs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for idx, ex in enumerate(examples):
        x0 = ex["x_clean"].numpy()
        xa = ex["x_adv"].numpy()

        def denorm(x):
            img = std * x + mean
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))
            return img

        img0 = denorm(x0)
        imga = denorm(xa)

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(img0)
        axs[0].axis("off")
        axs[0].set_title(
            f"Clean {class_names[ex['y_true']]} / pred {class_names[ex['y_pred_clean']]}"
        )
        axs[1].imshow(imga)
        axs[1].axis("off")
        axs[1].set_title(
            f"Adv pred {class_names[ex['y_pred_adv']]}"
        )
        plt.tight_layout()
        fig.savefig(out_dir / f"{prefix}_pair_{idx}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


def save_interpretability_maps(model, img_clean, img_adv, label_true, label_adv, class_names, out_dir, prefix, GradCAM_Class, Saliency_Func):
    """Computes and saves Saliency and GradCAM for clean and adversarial images."""
    device = next(model.parameters()).device
    model.to(device)
    
    # 1. Determine target layer for GradCAM based on model type (Requires model class knowledge)
    if 'CustomCNN' in model.__class__.__name__:
        target_layer_name = 'features.8'
    elif 'ResNetSmall' in model.__class__.__name__:
        target_layer_name = 'layer4.1.bn2' # Assuming this is the final BN layer
    else:
        raise ValueError("Unknown model type for GradCAM")

    # 2. Generate Interpretability Maps using imported functions/classes
    sal_clean = Saliency_Func(model, img_clean, label_true, device)
    sal_adv = Saliency_Func(model, img_adv, label_adv, device)
    
    grad_cam = GradCAM_Class(model, target_layer_name)
    
    cam_clean, _ = grad_cam(img_clean, target_class=label_true)
    cam_adv, _ = grad_cam(img_adv, target_class=label_adv)
    
    grad_cam.close() 

    # 3. Process Labels and Paths
    label_true_name = class_names[label_true]
    label_adv_name = class_names[label_adv]
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True) 

    # 4. Save results 
    
    # Save raw saliency maps
    for name, sal in [("clean", sal_clean), ("adv", sal_adv)]:
        p = base / f"{prefix}_saliency_{name}_{label_true_name}_to_{label_adv_name}.png"
        # This save command now works because 'base' exists:
        plt.figure(figsize=(4, 4)); plt.axis("off"); plt.imshow(sal, cmap="hot")
        plt.tight_layout(); plt.savefig(p, bbox_inches="tight", pad_inches=0); plt.close()

    # Save GradCAM overlays 
    # (show_and_save_overlay handles its own sub-directory creation if needed)
    show_and_save_overlay(img_clean, cam_clean, str(base / f"{prefix}_gradcam_clean_{label_true_name}.png"))
    show_and_save_overlay(img_adv, cam_adv, str(base / f"{prefix}_gradcam_adv_{label_adv_name}.png"))

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