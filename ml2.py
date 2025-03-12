import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define dataset paths
original_dataset_path = "/kaggle/input/breast-cancer-dataset/BreaKHis_v1/histology_slides/breast"
train_path = "/kaggle/working/preprocessed_dataset/train"
validation_path = "/kaggle/working/preprocessed_dataset/validation"

categories = ["benign", "malignant"]

# Create directories
for category in categories:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(validation_path, category), exist_ok=True)

# Function to collect & split images
def collect_images(source_dir, train_dest, val_dest, category, val_split=0.2):
    all_images = [os.path.join(root, file) for root, _, files in os.walk(source_dir) for file in files if file.endswith((".jpg", ".png"))]
    train_images, val_images = train_test_split(all_images, test_size=val_split, random_state=42)
    for img in train_images:
        shutil.copy(img, os.path.join(train_dest, category, os.path.basename(img)))
    for img in val_images:
        shutil.copy(img, os.path.join(val_dest, category, os.path.basename(img)))

for category in categories:
    collect_images(os.path.join(original_dataset_path, category), train_path, validation_path, category)

# Transformations & DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
validation_dataset = datasets.ImageFolder(root=validation_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# CNN Model (ResNet-18)
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.model = timm.create_model("resnet18", pretrained=True, num_classes=2)
    def forward(self, x):
        return self.model(x)

# ViT Model
class ViT_Model(nn.Module):
    def __init__(self):
        super(ViT_Model, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    def forward(self, x):
        return self.model(x)

# Swin Transformer Model
class Swin_Model(nn.Module):
    def __init__(self):
        super(Swin_Model, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=2)
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, val_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    
    return model

# Train models
cnn_model = train_model(CNN_Model(), train_loader, val_loader)
vit_model = train_model(ViT_Model(), train_loader, val_loader)
swin_model = train_model(Swin_Model(), train_loader, val_loader)

# Ensemble Learning
def ensemble_predict(models, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)
        model.eval()
    
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = [torch.softmax(model(images), dim=1) for model in models]
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            preds = torch.argmax(avg_output, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"✅ Ensemble Model Accuracy: {accuracy:.4f}")
    return accuracy

ensemble_accuracy = ensemble_predict([cnn_model, vit_model, swin_model], val_loader)

# Model Evaluation
def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
    return accuracy_score(true_labels, predictions)

cnn_accuracy = evaluate_model(cnn_model, val_loader)
vit_accuracy = evaluate_model(vit_model, val_loader)
swin_accuracy = evaluate_model(swin_model, val_loader)

# Display Accuracy Results
print(f"✅ CNN Model Accuracy: {cnn_accuracy:.4f}")
print(f"✅ ViT Model Accuracy: {vit_accuracy:.4f}")
print(f"✅ Swin Transformer Accuracy: {swin_accuracy:.4f}")

# Comparison Graph
model_names = ["CNN", "ViT", "Swin Transformer", "Ensemble"]
accuracies = [cnn_accuracy, vit_accuracy, swin_accuracy, ensemble_accuracy]

plt.figure(figsize=(8,5))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Comparison of Individual Models vs Ensemble Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Performance Boost
best_individual_model_acc = max(cnn_accuracy, vit_accuracy, swin_accuracy)
performance_boost = ensemble_accuracy - best_individual_model_acc

print(f"📊 Best Individual Model Accuracy: {best_individual_model_acc:.4f}")
print(f"📈 Performance Boost with Ensemble: {performance_boost:.4f} (Improvement of {performance_boost * 100: .2f}%)")
