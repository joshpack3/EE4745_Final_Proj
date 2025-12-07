import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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

# --- Model 1: Custom CNN ---

class CustomCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    Requires num_classes during initialization.
    """
    def __init__(self, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        # Note: Assumes 64x64 input leading to 8x8 spatial output for the flattening layer.
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
        self.target_layer_name = 'features.8' 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

# --- Model 2: Lightweight ResNet Block Definition ---

class BasicBlock(nn.Module):
    """
    Basic block used in small ResNet architectures.
    """
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNetSmall(nn.Module):
    """
    A lightweight ResNet-style network.
    Requires num_classes during initialization.
    """
    def __init__(self, num_classes):
        super(ResNetSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Helper to define layers
        def _make_layer(in_planes, planes, stride, num_blocks):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(BasicBlock(in_planes, planes, stride))
                in_planes = planes
            return nn.Sequential(*layers)
        
        # ResNet layer blocks 
        self.layer1 = _make_layer(16, 16, stride=1, num_blocks=2) 
        self.layer2 = _make_layer(16, 32, stride=2, num_blocks=2) 
        self.layer3 = _make_layer(32, 64, stride=2, num_blocks=2) 
        self.layer4 = _make_layer(64, 128, stride=2, num_blocks=2) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.linear = nn.Linear(128, num_classes)

        self.target_layer_name = 'layer4.1.bn2' 

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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
    
    # NOTE: Assumes folder names are 'train' and 'valid'
    train_dataset = ImageFolder(root=os.path.join(data_root, train_dir), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(data_root, valid_dir), transform=val_transform)
    
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

# --- Core Training Function ---

def train_model(model, model_name, train_loader, val_loader, num_epochs, device, class_names, log_dir, models_dir):
    """Handles the full training, logging, and checkpointing process."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Setup environment
    # 1: Clean the log directory if it exists
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    # 2: Create the log and models directory
    os.makedirs(log_dir, exist_ok=True)
    # 3: Initialize the writer AFTER the log_dir is created
    os.makedirs(models_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    
    start_time_total = time.time()
    best_accuracy = 0.0
    
    train_accuracies, val_accuracies = [], []
    
    print(f"\n--- Training {model_name} (Params: {count_parameters(model)}) on {device} ---")
    
    for epoch in range(num_epochs):
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
            torch.save(model.state_dict(), checkpoint_path)

    total_training_time = time.time() - start_time_total
    
    # Final Evaluation for reporting
    final_loss, final_acc, all_preds, all_labels = evaluate_model(model, val_loader, criterion, device)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'model_name': model_name,
        'parameter_count': count_parameters(model),
        'total_training_time': total_training_time,
        'final_val_accuracy': final_acc,
        'per_class_report': report,
        'confusion_matrix': cm,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'all_labels': all_labels, 
        'all_preds': all_preds,
        'best_checkpoint_path': os.path.join(models_dir, f"{model_name}-original.pt")
    }


# --- Interpretability Functions ---

def visualize_saliency_map(model, input_tensor, target_class=None):
    """Generates the Saliency Map for a given input tensor."""
    model.eval()
    input_tensor = input_tensor.clone().detach().to(input_tensor.device) 
    input_tensor.requires_grad_()
    
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
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

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
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
    # 1. Load Model Checkpoint
    model_path = os.path.join(models_dir, f"{model_name}-original.pt")
    # Load model state onto the CPU (as DEVICE is 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode

    # 2. Generate Interpretability Maps
    # NOTE: visualize_saliency_map and GradCAM are assumed to be imported from models.py
    saliency = visualize_saliency_map(model, input_tensor)
    
    grad_cam_instance = GradCAM(model, model.target_layer_name)
    cam, predicted_label_idx = grad_cam_instance(input_tensor)
    
    # 3. Process Labels
    predicted_class = class_names[predicted_label_idx]
    actual_class = class_names[actual_label]

    # 4. Denormalize Image for Display
    # Standard ImageNet mean/std used in the transforms
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Permute (C, H, W) to (H, W, C) and denormalize
    img_np = sample_img_tensor.permute(1, 2, 0).numpy() * std + mean
    img_np = np.clip(img_np, 0, 1)

    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original: {actual_class}\nPrediction: {predicted_class}", fontsize=14)
    axes[0].axis('off')

    # Saliency Map (Highlights pixel importance)
    axes[1].imshow(img_np)
    axes[1].imshow(saliency, cmap='hot', alpha=0.5) 
    axes[1].set_title(f"Saliency Map (Pixel Gradient)", fontsize=14)
    axes[1].axis('off')

    # Grad-CAM (Highlights relevant feature region)
    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap='jet', alpha=0.6) 
    axes[2].set_title(f"Grad-CAM (Feature Activation)", fontsize=14)
    axes[2].axis('off')
    
    fig.suptitle(f"Model Interpretability: {model_name}", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()