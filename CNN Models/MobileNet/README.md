# Optimize and Deploy Deep Learning Models - MobileNet Documentation

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Model Implementation and Transfer Learning](#2-model-implementation-and-transfer-learning)
    - [2.1 TensorFlow](#21-tensorflow)
    - [2.2 PyTorch](#22-pytorch)
- [4. Optimization Techniques](#4-optimization-techniques)
    - [4.1 Quantization](#41-quantization)
    - [4.2 Pruning](#42-pruning)
      - [4.2.1 TensorFlow Pruning](#421-tensorflow-pruning)
      - [4.2.2 PyTorch Pruning](#422-pytorch-pruning)
      - [4.2.3 FastNAS with ModelOpt](#423-fastnas-with-modelopt)
      - [4.2.4 Useful Links and References](#424-useful-links-and-references)
    - [4.3 Knowledge Distillation](#43-knowledge-distillation)
- [5. Model Formats Conversions](#5-model-formats-conversions)

## 1. Introduction

- We began our work by training the MobileNet model on a simple binary classification task using the Cats vs. Dogs dataset provided by TensorFlow Datasets (`tfds`). We used a pre-trained mobilenet model on image net weights and did transfer learning to change the model to binary classification. Once the model was successfully trained, we proceeded to apply various optimization techniques to improve its efficiency for deployment.

- After applying optimizations, we observed that the binary classification task (cats vs. dogs) was relatively easy. Therefore, we switched to the CIFAR-10 dataset, which includes 10 classes. Additionally, we applied MobileNet to another 10-class classification problem: classifying spoken digits (0 to 9). The audio data was converted to spectrograms to match MobileNet’s input format.

- We worked with 2 frameworks (tensorflow and pytorch)

## 2. Model implementation and transfer learning
### 2.1. Tensorflow
The model implementation using transfer learning and evaluation can be found at: 

Notebook: [MobileNet CIFAR-10 Notebook (Tensorflow)](Tensorflow/MobileNet_10Classes.ipynb)
### 2.2 Pytorch
In pytorch we implemented the model similarly and used transfer learning to adjust the model to classify on the CIFAR-10 dataset and also in Pytorch we have to implement train and evaluate functions.

Notebook: [MobileNet CIFAR-10 Notebook (Pytorch)](Pytorch/mobileNet_Pytorch.ipynb)

## 4. Optimization Techniques
### 4.1 Quantization

The first optimization technique we explored was **Quantization**. To facilitate this, we converted the model to the TensorFlow Lite (`.tflite`) format and applied three different types of quantization:

1. **Float16 Quantization (fp16)**: This technique reduces the precision of model weights to 16-bit floating point, decreasing model size while maintaining accuracy.
2. **Dynamic Range Quantization (int8)**: This approach quantizes only the model's weights to 8-bit integers during inference.
3. **Full Integer Quantization (static int8)**: Both model weights and activations are quantized to 8-bit integers, providing the highest level of optimization for reduced model size and faster inference.

These quantization techniques aim to reduce the model’s memory footprint and improve inference speed while maintaining acceptable accuracy levels for deployment on resource-constrained devices like the Jetson Nano. 

Notebook: [Tflite Quantization Notebook](Tensorflow/Cifar10_Quant.ipynb)

### 4.2 Pruning
1. We applied pruning using the prune API in both TensorFlow and PyTorch. In TensorFlow, we used the `prune_low_magnitude` function from the `tensorflow_model_optimization` library. However, we encountered an issue when applying this function to a pretrained MobileNet model from the Keras library. The function operates at the tensor level, meaning it requires the model to be built from scratch for pruning to work. Due to this limitation, we switched to PyTorch for pruning.

2. In PyTorch, we used the `torch.nn.utils.prune` library to apply both structured and unstructured pruning with various sparsity levels. After pruning, we measured the model's sparsity and observed a significant drop in accuracy. To recover the lost accuracy, we performed additional fine-tuning epochs. While we achieved satisfactory accuracy after fine-tuning, we encountered a new challenge which is leveraging the model's sparsity. We are currently still working on optimizing inference time by skipping zero values during computations and reducing the model size.
Notebook: [Pytorch Pruning Notebook](Pytorch/MobileNet_pytorch_pruning.ipynb)

3. We also applied pruning using the FastNAS method from the `ModelOpt` library. This method is specifically designed for Computer Vision models and identifies an optimal subnet that meets deployment constraints (such as FLOPs and model size) while minimizing accuracy loss. Given our pretrained MobileNet model, FastNAS efficiently pruned convolutional and linear layers to reduce model complexity. After pruning, we fine-tuned the resulting subnet to recover the model's accuracy. The output subnet had less parameters than the original model and thus smaller size and faster inference time.
Notebook: [ModelOpt Pruning Notebook](Pytorch/model_opt_prune.ipynb)
#### **Useful Links and References**
- **Pruning concepts**
    - [How to Prune Neural Networks with PyTorch](https://towardsdatascience.com/how-to-prune-neural-networks-with-pytorch-ebef60316b91/) 
- **NVIDIA TensorRT Acceleration**
    - [Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) 
    - [Sparsity in INT8: Training Workflow and Best Practices for NVIDIA TensorRT Acceleration](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/)
- **ModelOpt and FastNAS**  
  - Model-Format-Conversions\Model_Format_Conversion.ipynb

### 4.3 Knowledge Distillation
We initially attempted knowledge distillation from MobileNet to a simpler architecture, but the results were not promising. Since MobileNet is already a lightweight and optimized model, there was little room for further improvement in terms of efficiency or performance.

To explore the effectiveness of knowledge distillation further, we took a different approach:
We used a pre-trained ResNet model (a larger and more complex architecture) as the teacher and applied transfer learning to adapt it for CIFAR-10 classification. We then distilled the knowledge from the ResNet teacher to a MobileNet student model.

Key Insights from Knowledge Distillation:
- Faster Convergence: Knowledge distillation significantly reduced the number of training epochs required for MobileNet to reach the same accuracy as training it from scratch on the CIFAR-10 dataset.
- Limited Optimization Gains: Although knowledge distillation accelerated training, we did not observe additional benefits in terms of model size or inference speed, as MobileNet is already an optimized architecture.
This experiment highlighted that knowledge distillation can be a valuable technique for speeding up training but offers limited advantages when the target model (student) is already highly efficient like MobileNet.

Notebook: [ResNet Notebook](Tensorflow/resNet.ipynb)

Notebook: [Knowledge Distillation Notebook](Tensorflow/KD.ipynb)

## 5. Model Formats Conversions
1. H5 to ONNX: [tf conversion notebook](Model-Format-Conversions/Model_Format_Conversion.ipynb)
2. ONNX to protobuff: [tf conversion notebook](Model-Format-Conversions/Model_Format_Conversion.ipynb)
3. Protobuff to tflite
```python
original_model_path = '/content/drive/MyDrive/saved_models/tf_model'

# Load the model
model = tf.saved_model.load(original_model_path)
# Explicitly get the function
concrete_func = model.signatures["serving_default"]
# Convert using the concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TFLite model
tflite_original_model_path = "/content/drive/My Drive/saved_models/test.tflite"
with open(tflite_original_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_original_model_path}")
```
4. Torch (pth) to ONNX: [pytorch conversion module](Model-Format-Conversions/conversions.py)
5. ONNX to tensorRT (trt): [pytorch conversion module](Model-Format-Conversions/conversions.py)

For conversion 4 and 5 there are usage examples found at: [conversions example usage](Model-Format-Conversions/test_conversion.ipynb)
