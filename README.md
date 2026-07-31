# 🧠 Image Classification using Convolutional Neural Networks (PyTorch)

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Torchvision](https://img.shields.io/badge/Torchvision-CIFAR10-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 📌 Overview

This repository presents an end-to-end implementation of a **Convolutional Neural Network (CNN)** for multi-class image classification using **PyTorch**.

Unlike projects that simply import pretrained architectures, this implementation builds the complete CNN pipeline from first principles to develop a deep understanding of how convolutional neural networks learn hierarchical visual representations through gradient-based optimization.

The project covers every stage of the deep learning workflow, including data preprocessing, convolution operations, feature extraction, optimization, validation, model serialization, and inference.

---

# Objectives

- Build a CNN completely from scratch
- Understand convolution mathematically and computationally
- Learn feature extraction through stacked convolutional layers
- Implement the complete training pipeline
- Understand gradient flow and backpropagation
- Perform inference on unseen images
- Build a strong foundation before moving to Transfer Learning (ResNet, EfficientNet)

---

# 🏗 CNN Architecture

```
                     Input Image
                  (3 × 32 × 32)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ Conv2D                          │
        │ 3 Channels → 16 Filters         │
        │ Kernel = 3 × 3                  │
        └─────────────────────────────────┘
                         │
                         ▼
                     ReLU Activation
                         │
                         ▼
                 Max Pooling (2 × 2)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ Conv2D                          │
        │16 Channels → 32 Filters         │
        │Kernel = 3 × 3                   │
        └─────────────────────────────────┘
                         │
                         ▼
                     ReLU Activation
                         │
                         ▼
                 Max Pooling (2 × 2)
                         │
                         ▼
                     Flatten
                         │
                         ▼
               Fully Connected (1152→128)
                         │
                         ▼
                     ReLU
                         │
                         ▼
                Fully Connected (128→10)
                         │
                         ▼
                     Raw Logits
                         │
                         ▼
               CrossEntropyLoss
```

---

# 📐 Tensor Shape Evolution

| Layer | Output Shape |
|---------|-------------|
| Input | 3 × 32 × 32 |
| Conv1 | 16 × 30 × 30 |
| ReLU | 16 × 30 × 30 |
| MaxPool | 16 × 15 × 15 |
| Conv2 | 32 × 13 × 13 |
| ReLU | 32 × 13 × 13 |
| MaxPool | 32 × 6 × 6 |
| Flatten | 1152 |
| FC1 | 128 |
| FC2 | 10 |

---

# Learning Pipeline

```
               Image
                 │
                 ▼
            DataLoader
                 │
                 ▼
          Convolution Layers
                 │
                 ▼
          Feature Extraction
                 │
                 ▼
          Fully Connected Layer
                 │
                 ▼
             Raw Logits
                 │
                 ▼
       Cross Entropy Loss
                 │
                 ▼
          Backpropagation
                 │
                 ▼
             Gradients
                 │
                 ▼
          Adam Optimizer
                 │
                 ▼
        Updated CNN Parameters
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
├── test.py
├── predict.py
│
├── cnn_model.pth
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The model is trained on the **CIFAR-10** benchmark dataset.

## Dataset Statistics

| Property | Value |
|-----------|--------|
| Images | 60,000 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Classes | 10 |
| Resolution | 32 × 32 RGB |

Classes

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

---

# ⚙ Training Configuration

| Parameter | Value |
|------------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 32 |

---

# 📈 Training Progress

| Epoch | Loss |
|-------:|------:|
| 1 | 1.3622 |
| 2 | 1.0233 |
| 3 | 0.8891 |
| 4 | 0.7969 |
| 5 | 0.7250 |
| 6 | 0.6586 |
| 7 | 0.5946 |
| 8 | 0.5384 |
| 9 | 0.4916 |
| 10 | 0.4407 |

The training loss decreases consistently across epochs, indicating stable optimization and successful convergence.

---

# Performance

| Metric | Score |
|----------|-------|
| Validation Accuracy | **67.27 %** |

Although intentionally lightweight, the network demonstrates successful feature learning and generalization on unseen images.

---

# 🔬 Model Training Workflow

```
Dataset

↓

Mini Batch

↓

CNN Forward Pass

↓

Logits

↓

CrossEntropyLoss

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

---

# 💾 Saving the Model

The learned parameters are stored using PyTorch's serialization mechanism.

```python
torch.save(model.state_dict(), "cnn_model.pth")
```

The trained model can later be restored without retraining.

---

# Inference Pipeline

```
Custom Image

↓

Resize

↓

Normalization

↓

Tensor

↓

Batch Dimension

↓

CNN

↓

Logits

↓

Argmax

↓

Predicted Class
```

Example

```python
_, predicted = torch.max(outputs, 1)
```

---

# 🛠 Technology Stack

- Python
- PyTorch
- Torchvision
- Pillow
- NumPy

---

# 🧩 Deep Learning Concepts Demonstrated

✅ Convolution Operation

✅ Learnable Filters

✅ Feature Maps

✅ ReLU Activation

✅ Max Pooling

✅ Flatten Layer

✅ Fully Connected Networks

✅ Forward Propagation

✅ Cross Entropy Loss

✅ Backpropagation

✅ Gradient Computation

✅ Adam Optimization

✅ Model Evaluation

✅ Model Serialization

✅ Image Inference

---

# Future Improvements

The current implementation intentionally keeps the architecture lightweight to emphasize understanding rather than benchmark performance.

Possible improvements include:

- Batch Normalization
- Dropout Regularization
- Data Augmentation
- Learning Rate Scheduler
- Early Stopping
- Mixed Precision Training
- GPU Training
- TensorBoard Logging
- Hyperparameter Optimization
- Transfer Learning (ResNet, EfficientNet)
- Vision Transformers (ViT)

---

# 📷 Sample Prediction

| Image | Prediction |
|---------|------------|
| 🐱 Your Cat | **Cat ✅** |

> *(Replace this section with screenshots from your own predictions.)*

---

# Key Takeaways

This project demonstrates the complete lifecycle of supervised image classification using Convolutional Neural Networks.

Rather than treating deep learning as a collection of high-level APIs, the implementation focuses on understanding how convolutional layers progressively transform raw pixel values into hierarchical feature representations and how these representations are optimized through gradient-based learning.

Building the network from scratch establishes a strong conceptual foundation for more advanced computer vision architectures such as ResNet, DenseNet, EfficientNet, Vision Transformers, and multimodal foundation models.

---

# 👨‍💻 Author

**Gausul Wara**

Machine Learning Engineer | Deep Learning | Computer Vision | LLM Engineering

Building production-ready AI systems while developing a deep understanding of modern machine learning architectures.

---

## ⭐ If you found this repository useful, consider giving it a star!