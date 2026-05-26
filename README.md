# Plant Disease Prediction using CNN Image Classifier

A deep learning–based image classification system for detecting plant diseases from leaf images using CNN and hybrid deep learning architectures.

---

# Problem Statement

Plant diseases significantly reduce agricultural productivity and crop quality. Traditional disease identification methods rely on manual inspection by experts, which can be time-consuming, expensive, and inaccessible to small-scale farmers.

The objective of this project is to build a deep learning–based image classification system capable of automatically identifying plant diseases from leaf images. The system uses Convolutional Neural Networks (CNNs) and hybrid deep learning architectures to classify healthy and diseased plant leaves across multiple categories.

The project aims to:
- Automate plant disease detection
- Improve early diagnosis accuracy
- Reduce dependency on manual inspection
- Support precision agriculture practices

---

# Project Architecture

```text
                +----------------------+
                |  PlantVillage Dataset |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Data Preprocessing   |
                | - Resize (224x224)  |
                | - Normalization     |
                | - Augmentation      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Dataset Splitting    |
                | Train : 70%         |
                | Validation : 15%    |
                | Test : 15%          |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | DataLoader Pipeline  |
                | Batch Size : 32      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Deep Learning Models |
                | - Simple CNN         |
                | - ResNet18           |
                | - ResNet50           |
                | - ViT                |
                | - AllCNN             |
                | - Attention Models   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Training & Evaluation|
                | Adam Optimizer       |
                | CrossEntropy Loss    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Prediction Results   |
                +----------------------+
```

---

# Dataset Information

| Attribute | Value |
|---|---|
| Dataset Name | PlantVillage |
| Total Images | 54,305 |
| Number of Classes | 38 |
| Dataset Size | 2.33 GB |

Dataset Source:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

---

# Data Preprocessing

The following preprocessing techniques were applied:

- Image resizing to `224 × 224`
- Pixel normalization
- Data augmentation:
  - Random Rotation
  - Width Shift
  - Height Shift
  - Shear Transformation
  - Zoom Augmentation
  - Horizontal Flipping

---

# Dataset Split

| Dataset | Percentage |
|---|---|
| Training Set | 70% |
| Validation Set | 15% |
| Test Set | 15% |

---

# Tools & Technologies

## Programming Language
- Python

## Deep Learning Framework
- PyTorch
- Torchvision

## Data Processing Libraries
- NumPy
- Pandas

## Visualization
- Matplotlib

## Machine Learning Utilities
- Scikit-learn

## Development Environment
- Jupyter Notebook

---

# Models Used

## CNN-Based Models
- Simple CNN
- ResNet18
- ResNet50
- Vision Transformer (ViT)
- AllCNN

## Hybrid Models
- ResNet18 with Attention
- CNN with Attention

---

# Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Cross Entropy Loss |
| Learning Rate | 0.003 |
| Epochs | 5 |
| Batch Size | 32 |

---

# Backend Workflow

## 1. Data Loading

The dataset is loaded using PyTorch `DataLoader` for efficient batch processing.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

---

## 2. Data Transformation Pipeline

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 3. Model Initialization

Example using ResNet18:

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=True)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 38)
```

---

## 4. Loss Function & Optimizer

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.003
)
```

---

## 5. Training Loop

```python
for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")
```

---

## 6. Model Evaluation

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy}%")
```

---

# Results

- Successfully trained multiple CNN and hybrid deep learning models.
- ResNet18 achieved the best performance on the test dataset.
- The model demonstrated strong classification performance across 38 plant disease categories.
- Attention-based models improved feature extraction and disease localization.
- The project demonstrates the practical application of deep learning in precision agriculture.

---

---

# Future Improvements

- Web application deployment
- Mobile application integration
- Real-time camera-based disease detection
- Lightweight model optimization for edge devices
- Improved hybrid transformer architectures

---

# Conclusion

This project demonstrates the effectiveness of CNN-based and hybrid deep learning models for automated plant disease classification. The system can accurately identify multiple plant diseases from leaf images and has strong potential for real-world agricultural assistance systems.

---

# Author

## Arkadip Kansabanik

Ramakrishna Mission Vivekananda Educational & Research Institute

---
