import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# --- Utility Functions ---

def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Data Loading and Preparation Functions ---

def prepare_data_loaders(data_root, image_size, batch_size, train_dir='train', valid_dir='valid'):
    """Prepares and returns the data loaders and class names."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # NOTE: Default folder names are 'train' and 'valid'
    train_dataset = ImageFolder(root=os.path.join(data_root, train_dir), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(data_root, valid_dir), transform=val_transform)
    
    # Using num_workers=0 for compatibility with notebook/Windows environments
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_dataset.classes, val_dataset

def evaluate_model(model, loader, criterion, device):
    """Evaluates the model on a given data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels

# --- Core Training Function (Problem A) ---

def train_model(model, model_name, train_loader, val_loader, num_epochs, device, log_dir, models_dir):
    """Handles the full training, logging, and checkpointing process."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Setup environment
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    
    start_time_total = time.time()
    best_accuracy = 0.0
    
    train_accuracies, val_accuracies = [], []
    
    print(f"\n--- Training {model_name} (Params: {count_parameters(model)}) on {device} ---")
    
    for epoch in range(num_epochs):
        # ... [Training loop logic] ... (Omitted for brevity)
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time_epoch = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_time = time.time() - start_time_epoch
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Validation
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | Time: {epoch_time:.2f}s")
        
        # Log to TensorBoard
        writer.add_scalar(f'Loss/Train/{model_name}', running_loss/len(train_loader), epoch)
        writer.add_scalar(f'Accuracy/Validation/{model_name}', val_accuracy, epoch)
        
        # Save best model checkpoint
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = os.path.join(models_dir, f"{model_name}-original.pt")
            robust_checkpoint_path = os.path.normpath(os.path.abspath(checkpoint_path))
            torch.save(model.state_dict(), robust_checkpoint_path)

    total_training_time = time.time() - start_time_total
    
    # Load the BEST model weights before the final evaluation for reporting
    best_checkpoint_path = os.path.join(models_dir, f"{model_name}-original.pt")
    
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print(f"Loaded best checkpoint for {model_name} before final evaluation.")

    # Set model to evaluation mode
    model.eval()

    # Final Evaluation for reporting
    final_loss, final_acc, all_preds, all_labels = evaluate_model(model, val_loader, criterion, device)
    
    return {
        'model_name': model_name,
        'parameter_count': count_parameters(model),
        'total_training_time': total_training_time,
        'final_val_accuracy': final_acc,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'all_labels': all_labels, 
        'all_preds': all_preds,
        'best_checkpoint_path': os.path.join(models_dir, f"{model_name}-original.pt")
    }