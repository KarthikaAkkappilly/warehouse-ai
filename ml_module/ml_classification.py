#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from collections import defaultdict
import random
import os

ROOT = "/content/drive/MyDrive/warehouse_ai"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Pretrained Weights
weights = ResNet50_Weights.DEFAULT

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean,
                         std=weights.transforms().std)
])

test_transform = weights.transforms()

# Dataset
data_path = os.path.join(ROOT, "ml_module", "dataset")
full_dataset = datasets.ImageFolder(data_path)

# Train/test split
class_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(full_dataset.imgs):
    class_to_indices[label].append(idx)

test_indices = []
train_indices = []

for cls, indices in class_to_indices.items():
    random.shuffle(indices)
    n_test = max(1, int(0.2 * len(indices)))
    test_indices.extend(indices[:n_test])
    train_indices.extend(indices[n_test:])

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model
model = resnet50(weights=weights)

# Fine-tune last ResNet block
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
model = model.to(device)

# Loss & Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

print("Training Finished.")

# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
print(f"\nTest Accuracy: {accuracy:.2f}%\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Inference demo
img_path = os.path.join(ROOT, "ml_module", "dataset", "fragile", "000002.jpg")
img = Image.open(img_path).convert("RGB")
img_tensor = test_transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    _, pred = torch.max(output, 1)

print(f"\nImage: {img_path}")
print(f"Predicted class: {full_dataset.classes[pred.item()]}")

# Save Model
save_path = os.path.join(ROOT, "ml_module", "resnet50_warehouse.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved successfully at {save_path}")


# #Limitations of the Machine Learning Model
# 
# The main limitation of this machine learning model is the **small size of the dataset**, which contains only a few images per class. While data augmentation increases diversity, the model may still struggle with real-world variations in object appearance, lighting, and orientation. As a result, test metrics can be highly sensitive: a single misclassification significantly impacts accuracy, precision, and recall.
# 
# Additionally, the model is trained using only **static images**. In a real warehouse scenario, objects may appear at different distances, under occlusion, or partially visible, which may reduce the model’s reliability.
# 
# Finally, only the **last ResNet block is fine-tuned** to prevent overfitting. While this approach works well for small datasets, it may limit the model’s ability to fully adapt to warehouse-specific features, such as complex textures or mixed materials.
# 
# Advantage: Despite these limitations, the model effectively demonstrates classification capabilities with **high accuracy** and **reliable inference** on new images.
