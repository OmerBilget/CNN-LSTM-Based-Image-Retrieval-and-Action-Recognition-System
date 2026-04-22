# CNN-LSTM-Based-Image-Retrieval-and-Action-Recognition-System
This project is a deep learning case study exploring multiple approaches for image classification, image retrieval, and human action recognition. It includes models based on CNN and LSTM architectures. The study demonstrates how different deep learning techniques can be applied to visual understanding tasks using TensorFlow/Keras.

# Overview
This repository presents a comprehensive deep learning case study exploring multiple computer vision tasks, including:

- Image Classification 
- Content-Based Image Retrieval
- Human Action Recognition

The study investigates different deep learning architectures such as CNN, 3D CNN, LSTM, and hybrid CNN-LSTM models to understand both spatial and spatiotemporal features in visual data.

---

# Methodology

### 1. Image Classification (MNIST)
- Dataset: MNIST handwritten digits
- Model: Convolutional Neural Network (CNN)
- Purpose: Learn spatial features for digit recognition

---

### 2. Image Retrieval  (CIFAR-10 database)
- Approach: Content-Based Image Retrieval (CBIR)
- Feature Extraction: CNN-based embeddings
- Evaluation Metrics:
  - Top-1 Accuracy
  - Top-5 Accuracy
  - Top-5 (Any match)

---

### 3. Action Recognition (UCF-101 subset)
- Dataset: 10-class human activity 
  (Basketball, Biking, Bowling, HighJump, HorseRiding, JavelinThrow, RopeClimbing, Skiing, SkyDiving, TennisSwing)

- Models Implemented:
  - 3D CNN
  - 3D CNN + LSTM (Hybrid)
  - LSTM (sequence-based)
  - Transfer Learning (improved variant)

- Purpose:
  Capture temporal dynamics in video sequences using spatiotemporal learning.

---

# Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Scikit-learn

---

# Results

# 1. MNIST Classification
| Metric     | Value   |
|-----------|--------|
| Accuracy  | 99.69% |
| Precision | 99.69% |
| Recall    | 99.69% |
| F1 Score  | 99.69% |

---

# 2. Image Retrieval Performance
| Metric            | Value   |
|------------------|--------|
| Top-1 Accuracy   | 81.34% |
| Top-5 Accuracy   | 80.53% |
| Top-5 (Any Hit)  | 90.88% |

---

# 3. Action Recognition Results

# Model Comparison

| Model            | Accuracy |
|------------------|---------|
| LSTM             | 42.0%   |
| LSTM(v2)         | 52.4%   |
| 3D CNN           | 52.7%   |
| 3D CNN (v2)      | 64.8%   |
| 3D CNN + LSTM    | 78.3%   |

---
# Detailed Metrics (Best Model: 3D CNN + LSTM)

| Metric     | Value |
|-----------|------|
| Accuracy  | 78%  |
| Precision | 84%  |
| Recall    | 78%  |
| F1 Score  | 78%  |

---

# Key Insights

- CNN performs extremely well on spatial tasks (MNIST, Retrieval)
- LSTM alone struggles with complex visual temporal data
- 3D CNN improves spatial-temporal feature extraction
- Hybrid CNN-LSTM models significantly boost performance
- Transfer learning provides competitive results with less training
---

