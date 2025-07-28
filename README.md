# Optimize and Deploy Deep Learning Models on Edge Devices

Welcome to our official repository for the graduation project titled **"Optimize and Deploy Deep Learning Models on Edge Devices"**
  
**Mentored by:** Si-Vision

![alt text](images/cloudvsedge.png)

---

## 📘 Abstract

This project explores efficient deployment of deep learning models on resource-constrained edge devices, particularly the NVIDIA Jetson Nano. It addresses computational, memory, and energy limitations using a full pipeline of optimization techniques, including pruning, quantization, knowledge distillation, low-rank approximation, and TensorRT acceleration. We evaluated this pipeline on architectures like VGG11, MobileNetV2, EfficientNet, NASNet, and AlexNet, achieving up to 18× speedup with minimal accuracy loss.

---

## 📌 Overview

Deep neural networks (DNNs) have achieved outstanding performance in numerous applications. However, their computational demands hinder deployment on low-power devices. This project proposes a complete pipeline integrating:

- Model pruning (structured and unstructured)
- Quantization (post-training & quantization-aware)
- Low-rank approximations (SVD-based)
- Knowledge distillation (KD)
- TensorRT acceleration

These methods are applied to 5 popular architectures (VGG, MobileNet, NASNet, AlexNet, EfficientNet) and tested on Jetson Nano:
- **AlexNet**
- **VGG11 & VGG16**
- **MobileNetV2**
- **EfficientNet-B0**
- **NASNet-Mobile**
- **YOLOv8s**
- **LLMs** (e.g., Qwen2.5-0.5B, TinyLlama-1.1B)


In addition to the five CNN architectures, we deployed real-time applications to further evaluate the effectiveness of the optimization and deployment pipeline on the Jetson Nano. These applications include:

- `YOLOv8s` for real-time object detection

- `MobileNet SSD` for lightweight object detection

- Face detection and recognition, using `facedetect` for detection and `InceptionResNetV1` for generating face embeddings

These practical use cases demonstrate the feasibility and performance benefits of running optimized deep learning models on resource-constrained edge devices.

---

## 🔧 Optimization Techniques

### 📏 Quantization
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- TensorRT precision calibration (INT8, FP16)

### 🔧 Pruning
- Structured channel pruning
- Iterative pruning
- FastNAS pruning
- Torch-Pruning with dependency graphs

### 🧊 Low-Rank Approximation
- SVD-based layer compression
- CP-like decomposition for Conv2D

### 🧪 Knowledge Distillation
- Student-teacher training with NASNet
- Used for both classification and detection tasks

---

## 📊 Results Highlights

- **MobileNetV2**:  
  From 9.88MB to **0.47MB**, and throughput improved from **31.9 FPS** to **66.1 FPS** (TensorRT + FP16).

- **VGG11**:  
  Achieved **16× speedup** with minimal (2%) accuracy drop using structured pruning and quantization and tensorRT acceleration.

- **YOLOv8s**:  
  Real-time object detection with quantization and KD, while reducing model size and inference time significantly.

- **LLMs on Jetson Nano**:  
  Benchmarked models like TinyLlama and Qwen2.5-0.5B with memory and latency evaluations for edge NLP.

📊 **Full Excel Results Sheet:** [View on OneDrive](https://docs.google.com/spreadsheets/d/1Db0EXINeAfmpou3PeocLsoQTGAfQ5cSlHY8U9TEEHKI/edit?gid=0#gid=0)

(See detailed tables in `/results` and thesis for full metrics)

---

## ⚙️ Environment & Deployment Platform 

- **Device**: NVIDIA Jetson Nano
- **Power**: 5V 3A MicroUSB
- **Frameworks**: PyTorch, TensorFlow Lite, ONNX, TensorRT
- **TensorRT** for high-speed inference
- **ONNX Runtime** as baseline
- **PyTorch** & **TensorFlow Lite** as training and conversion tools
- **Datasets:** CIFAR-10, ImageNet, COCO, Pascal VOC
- 

---

## 📬 Contact

For questions, suggestions, or collaborations, feel free to open an issue or contact us directly.

---
