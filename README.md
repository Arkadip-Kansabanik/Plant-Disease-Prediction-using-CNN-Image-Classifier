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
- Tensorflow

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


---

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



---
