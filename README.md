# Image Classification using Convolutional Neural Networks (PyTorch)

> A complete end-to-end implementation of a Convolutional Neural Network (CNN) for multi-class image classification on the CIFAR-10 dataset using PyTorch.

This project was built from scratch to understand the complete deep learning pipeline—from raw image tensors to learned feature representations, gradient-based optimization, and inference on unseen images.

---

# Overview

Unlike projects that rely on pretrained models, this implementation builds a CNN manually to understand every component involved in supervised image classification.

The project covers:

- Image preprocessing
- Dataset & DataLoader pipeline
- Convolutional Neural Network architecture
- Forward propagation
- Cross Entropy Loss
- Backpropagation
- Adam Optimization
- Validation
- Model serialization
- Inference on custom images

Every layer of the network was intentionally implemented to understand how CNNs learn hierarchical visual representations.

---

# Project Architecture

```
Input Image (32×32×3)
          │
          ▼
┌──────────────────────┐
│ Conv2D (3 → 16)      │
└──────────────────────┘
          │
          ▼
ReLU
          │
          ▼
MaxPool (2×2)
          │
          ▼
┌──────────────────────┐
│ Conv2D (16 → 32)     │
└──────────────────────┘
          │
          ▼
ReLU
          │
          ▼
MaxPool (2×2)
          │
          ▼
Flatten
          │
          ▼
Linear (1152 → 128)
          │
          ▼
ReLU
          │
          ▼
Linear (128 → 10)
          │
          ▼
Raw Logits
          │
          ▼
CrossEntropyLoss
```

---

# 📂 Project Structure

```
Image-Classification-CNN/

│
├── data/
│
├── model.py
├── train.py
├── predict.py
├── test.py
│
├── cnn_model.pth
│
├── requirements.txt
└── README.md
```

---

# Dataset

The model is trained on the **CIFAR-10** dataset.

### Dataset Statistics

- 60,000 RGB Images
- Resolution: 32 × 32
- 10 Classes

Classes:

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

Training Images:

```
50,000
```

Testing Images:

```
10,000
```

---

# Model Design

The network follows a simple hierarchical feature extraction pipeline.

### Feature Extraction

```
Conv2D

↓

ReLU

↓

MaxPool

↓

Conv2D

↓

ReLU

↓

MaxPool
```

The convolution layers progressively learn

- edges
- corners
- textures
- local visual patterns

before compressing them into a learned feature representation.

---

### Classification Head

```
Flatten

↓

Fully Connected

↓

ReLU

↓

Fully Connected

↓

10 Logits
```

The classifier maps extracted visual features into one of the ten CIFAR-10 categories.

---

# ⚙ Training Configuration

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |

---

# Training Result

Training Loss

```
Epoch 1  → 1.3622

Epoch 2  → 1.0233

Epoch 3  → 0.8891

Epoch 4  → 0.7969

Epoch 5  → 0.7250

Epoch 6  → 0.6586

Epoch 7  → 0.5946

Epoch 8  → 0.5384

Epoch 9  → 0.4916

Epoch 10 → 0.4407
```

Validation Accuracy

```
67.27%
```

The steadily decreasing training loss indicates stable optimization and successful gradient-based learning.

---

# Learning Pipeline

```
Image

↓

DataLoader

↓

CNN

↓

Logits

↓

CrossEntropy Loss

↓

Backpropagation

↓

Gradients

↓

Adam Optimizer

↓

Updated Parameters

↓

Repeat
```

This iterative optimization enables the network to progressively minimize classification error across training epochs.

---

# Model Saving

The trained parameters are stored using PyTorch's `state_dict()` serialization.

```python
torch.save(model.state_dict(), "cnn_model.pth")
```

The model can later be restored for inference without retraining.

---

# Inference

Custom RGB images can be classified by

1. Loading the trained weights
2. Applying the same preprocessing pipeline
3. Performing a forward pass
4. Selecting the class with the highest logit

```python
_, predicted = torch.max(outputs, 1)
```

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- PIL
- NumPy

---

# Key Concepts Demonstrated

- Convolution Operation
- Feature Maps
- Weight Sharing
- ReLU Activation
- Max Pooling
- Flatten Layer
- Fully Connected Networks
- Forward Propagation
- Cross Entropy Loss
- Backpropagation
- Gradient Descent
- Adam Optimization
- Model Evaluation
- Inference Pipeline

---

# Future Improvements

Potential architectural improvements include

- Batch Normalization
- Dropout Regularization
- Learning Rate Scheduling
- Data Augmentation
- Early Stopping
- Transfer Learning (ResNet, EfficientNet)
- Mixed Precision Training
- GPU Training
- Hyperparameter Optimization

---

# Results

This project demonstrates the complete implementation of a CNN from first principles without relying on pretrained feature extractors.

Beyond achieving competitive performance on CIFAR-10 for a lightweight architecture, the primary objective was to build a strong conceptual understanding of how convolutional neural networks transform raw pixel data into learned semantic representations through gradient-based optimization.

The implementation serves as a foundation for more advanced computer vision architectures such as ResNet, DenseNet, EfficientNet, and Vision Transformers.