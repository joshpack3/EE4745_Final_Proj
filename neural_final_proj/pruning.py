import os
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from .basic_utils import evaluate_model

def count_sparse_weights(model):
    """Counts the total number of weights and the number of zero (pruned) weights."""
    total_weights = 0
    zero_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                total_weights += module.weight.numel()
                # Check for the actual 'weight' tensor (post-pruning)
                weight_tensor = module.weight.data
                zero_weights += torch.sum(weight_tensor == 0).item()
    
    sparsity = zero_weights / total_weights if total_weights > 0 else 0
    return total_weights, zero_weights, sparsity

def prune_and_evaluate(model_class, model_ckpt_path, sparsity_level, cfg, train_loader, val_loader, evaluate_model_func, finetune_epochs=5):
    """
    Applies pruning, evaluates pre-finetune, performs fine-tuning with epoch monitoring, 
    and evaluates post-finetune.
    """
    # 1. Setup Model
    device = cfg.device
    model_name = model_class.__name__
    criterion = nn.CrossEntropyLoss()
    
    model = model_class(num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    
    model_pruned = copy.deepcopy(model)
    
    # 2. Apply Pruning (Global Unstructured L1)
    parameters_to_prune = []
    for name, module in model_pruned.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity_level,
    )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    # 3. Pre-Finetune Evaluation
    _, acc_pre, _, _ = evaluate_model_func(model_pruned, val_loader, criterion, device)
    total_w, zero_w, sparsity = count_sparse_weights(model_pruned)
    
    # Checkpoint setup
    best_finetune_ckpt_path = os.path.join(cfg.out_dir, f"{model_name}-pruned-{int(sparsity_level*100)}%-best.pt")

    # Save the initial pruned state to ensure file existence and proper baseline
    torch.save(model_pruned.state_dict(), best_finetune_ckpt_path)
    
    # 4. Fine-Tuning
    optimizer = optim.Adam(model_pruned.parameters(), lr=0.0001)

    best_finetune_acc = 0
    
    print(f"\n--- Fine-Tuning ({model_name} @ {sparsity*100:.0f}% Sparsity) ---")
    print(f"Initial Acc: {acc_pre:.2f}%")
    
    for epoch in range(finetune_epochs):
        model_pruned.train()
        start_time_epoch = time.time()
        
        # Training Step
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_pruned(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation Step (Validation)
        epoch_time = time.time() - start_time_epoch
        _, current_acc, _, _ = evaluate_model_func(model_pruned, val_loader, criterion, device)
        
        # Checkpoint and Tracking
        if current_acc > best_finetune_acc:
            best_finetune_acc = current_acc
            torch.save(model_pruned.state_dict(), best_finetune_ckpt_path)
            status = "(New Best)"
        else:
            status = ""
            
        print(f"Epoch {epoch+1}/{finetune_epochs} | Val Acc: {current_acc:.2f}% | Time: {epoch_time:.2f}s {status}")

    # 5. Final Evaluation and Metric Collection
    
    # Load the best weights found (either the initial pruned state or the best fine-tuned state)
    model_pruned.load_state_dict(torch.load(best_finetune_ckpt_path, map_location=device))
    acc_post = best_finetune_acc 
    
    # Calculate model size on disk
    model_size_mb = os.path.getsize(best_finetune_ckpt_path) / (1024 * 1024)
    print(f"Final Best Acc (Post-Finetune): {acc_post:.2f}%")

    return {
        'sparsity_target': sparsity_level,
        'sparsity_actual': sparsity,
        'acc_pre_finetune': acc_pre,
        'acc_post_finetune': acc_post,
        'param_count': total_w, 
        'model_size_mb': model_size_mb,
        'model_ckpt_path': best_finetune_ckpt_path
    }


def measure_inference_latency(model, cfg, batch_size=1, num_runs=100, warm_up=10):
    """Measures the mean inference latency on CPU."""
    model.eval()
    
    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, 3, cfg.image_size, cfg.image_size).to(cfg.device)
    
    # Warm-up runs
    for _ in range(warm_up):
        _ = model(dummy_input)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model(dummy_input)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
    times_ms = np.array(times) * 1000 # Convert to milliseconds
    
    mean_latency = np.mean(times_ms)
    std_latency = np.std(times_ms)
    
    return mean_latency, std_latency