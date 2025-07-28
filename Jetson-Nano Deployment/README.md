# Deep Learning Model Deployment on Jetson Nano
<img src="Photos/coverphoto.jpg" alt="Project cover">

![JetPack](https://img.shields.io/badge/JetPack-4.6.1-blue)
![Python](https://img.shields.io/badge/Python-3.6.9-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-red)
![TorchVision](https://img.shields.io/badge/TorchVision-0.9.0-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.1-Green)

## Contents
- [Introduction](#introduction)
- [Environment Setup (Jetson Initial Setup)](#environment-setup-jetson-initial-setup)
- [Installation of Key Dependencies](#installation-of-key-dependencies)
  - [1. Python 3 and Virtual Environment Setup](#1-python-3-and-virtual-environment-setup)
  - [2. Creating a Swap File](#2-creating-a-swap-file)
  - [3. OpenCV with CUDA Support](#3-opencv-with-cuda-support)
  - [4. Installing jtop for System Monitoring](#4-installing-jtop-for-system-monitoring)
  - [5. Installing PyTorch and TorchVision](#5-installing-pytorch-and-torchvision)
  - [6. Installing onnxruntime-gpu](#6-installing-onnxruntime-gpu)
- [Installed Packages and Versions](#installed-packages-and-versions)
- [Issues Faced and Their Solution](#issues-faced-and-their-solution)
  - [1. Error in installing PyCUDA and NVCC Not Found in PATH](#1-error-in-installing-pycuda-and-nvcc-not-found-in-path)
  - [2. Illegal Instruction (Core Dumped)](#2-illegal-instruction-core-dumped)
- [Converting to TensorRT Using trtexec](#converting-to-tensorrt-using-trtexec)
- [Inference Scripts](#inference-scripts)
  - [1. TensorRT](#1-tensorrt)
  - [2. ONNX Runtime (CPU & GPU)](#2-onnx-runtime-cpu--gpu)

## Introduction
This documentation presents the ongoing process of deploying and benchmarking deep learning models on the NVIDIA Jetson Nano. The objective is to evaluate the performance of various models before and after optimization for edge inference.
## Environment Setup (Jetson Initial Setup)
[Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro)

To begin the deployment process, we prepared the Jetson Nano by setting up the operating system using NVIDIA's JetPack SDK.
Below are the steps followed for the initial setup:

1. **Download JetPack Image**  
   We downloaded the official JetPack SD card image (version **4.6.1**) from NVIDIA's embedded downloads page:  
   [https://developer.nvidia.com/embedded/downloads](https://developer.nvidia.com/embedded/downloads)

2. **Format the SD Card**  
   The SD card was formatted using the official SD Card Formatter tool provided by the SD Association:  
   [https://www.sdcard.org/downloads/formatter_4/eula_windows/](https://www.sdcard.org/downloads/formatter_4/eula_windows/)

3. **Flash the Image to SD Card**  
   The JetPack image was flashed onto the formatted SD card using **balenaEtcher**:  
   [https://etcher.balena.io/](https://etcher.balena.io/)

After flashing, the SD card was inserted into the Jetson Nano and the device was powered on for first-time setup and configuration.

## Installation of Key Dependencies

This section covers the installation of essential tools required for running and benchmarking deep learning models on Jetson Nano. These include setting up Python with virtual environments and expanding swap space to support package compilation (e.g., OpenCV).

### 1. Python 3 and Virtual Environment Setup

Install `pip3` and create a Python virtual environment to isolate your project dependencies.

```bash
sudo apt-get update
sudo apt-get install python3-pip
pip3 install virtualenv

# Create a virtual environment using system-wide site packages
python3 -m virtualenv -p python3 env --system-site-packages

# Activate the environment
source env/bin/activate
```
> Note: The --system-site-packages flag allows access to Jetson-specific packages (like CUDA-enabled libraries) installed globally.

### 2. Creating a Swap File (Recommended for Compilation Tasks)

Jetson Nano has limited RAM (4GB), which can cause large compilations (e.g., OpenCV from source) to fail. Creating a swap file helps prevent memory issues during these tasks.

```bash
# Allocate a 4GB swap file
sudo fallocate -l 4G /var/swapfile

# Set proper permissions
sudo chmod 600 /var/swapfile

# Format the file as swap space
sudo mkswap /var/swapfile

# Enable the swap file
sudo swapon /var/swapfile

# Make the change permanent across reboots
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```
🔄Reboot the Jetson. After rebooting check swap space
```bash
free -h
```

✅ After compilation tasks are completed, it is recommended to disable the swap file to reduce SD card wear
```bash
sudo swapoff /var/swapfile
```
### 3. OpenCV with CUDA support
Install these Dependencies before installing OpenCV:
```bash
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf“
sudo ldconfig
sudo apt-get install build-essential cmake git unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgtk2.0-dev libcanberra-gtk*
sudo apt-get install python3-dev python3-numpy python3-pip
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install libv4l-dev v4l-utils
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install libavresample-dev libvorbis-dev libxine2-dev
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install liblapack-dev libeigen3-dev gfortran
sudo apt-get install libhdf5-dev protobuf-compiler
sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev
```
Download OpenCV:
```bash
Download OpenCV:
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip 
unzip opencv.zip 
unzip opencv_contrib.zip
mv opencv-4.5.1 opencv
mv opencv_contrib-4.5.1 opencv_contrib
rm opencv.zip
rm opencv_contrib.zip
```
Build OpenCV:
```bash
cd ~/opencv
mkdir build
cd build 
```
The following block is a cmake command configuring the build process for compiling OpenCV from source on a system like the Jetson Nano.
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 -D WITH_OPENCL=OFF -D WITH_CUDA=ON -D CUDA_ARCH_BIN=5.3 -D CUDA_ARCH_PTX="" -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_NEON=ON -D WITH_QT=OFF -D WITH_OPENMP=ON -D WITH_OPENGL=ON -D BUILD_TIFF=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D WITH_TBB=ON -D BUILD_TBB=ON -D BUILD_TESTS=OFF -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_LIBV4L=ON -D OPENCV_ENABLE_NONFREE=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_python3=TRUE -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF ..
```
Compile the source code after configuration with cmake. This command will take a long time (around 2 hours)

```bash
make -j4
```
Finish the process:
```bash
cd ~
sudo rm -r /usr/include/opencv4/opencv2
cd ~/opencv/build
sudo make install
sudo ldconfig
make clean
sudo apt-get update 
```
Verify OpenCV Installation it should be version 4.5.1
```bash
#open python3 shell
python3
import cv2
cv2._version_
```
📸 To test a camera on Jetson Nano (Not done yet but can be useful if we get a camera)
```bash
ls /dev/video0   #csi camera
ls /dev/video*   # show you a list of cameras
```
Take a Photo:
```bash
# for testing CSI camera
nvgstcapture-1.0 --orientation=2
# V4L2 USB camera 
nvgstcapture-1.0 --camsrc=0 --cap-dev-node=1
```
### 4. Installing jtop for System Monitoring
**jtop** is a useful system monitoring tool designed for Jetson devices. It provides a real-time display of system performance metrics, including CPU, GPU, memory usage, and also you can use it to check that OpenCV is installed with cuda support. To install **jtop** on your Jetson Nano:
```bash
sudo -H pip3 install -U jetson-stats
sudo reboot

# After reboot you can use
jtop
```
### 5. Installing PyTorch and TorchVision
Check NVIDIA's [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

To install **PyTorch** and **TorchVision** on Jetson Nano, it is essential to ensure compatibility with the installed **JetPack** version. In this case, we're using **JetPack 4.6.1** and we installed **PyTorch v1.8 - torchvision v0.9.0**

Installation:
```bash
# substitute the link URL and wheel filename from the desired torch version above
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
Then install torchvision:
```bash
$ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
$ git clone --branch <version> https://github.com/pytorch/vision torchvision   # see the website for version of torchvision to download
$ cd torchvision
$ export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
$ python3 setup.py install --user
$ cd ../  # attempting to load torchvision from build dir will result in import error
$ pip install 'pillow<7'
```
Verification Script:
```python
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))

import torchvision
print(torchvision.__version__)
```
### 6. Installing onnxruntime-gpu
```bash
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
python3 -m pip install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```

## Installed Packages and Versions

| Package / Library     | Version       | Notes                                  |
|-----------------------|---------------|----------------------------------------|
| JetPack SDK           | 4.6.1         | OS image for Jetson with libraries and APIs|
| TensorRT              | 8.2.1         | Default with JetPack 4.6.1             |
| CUDA                  | 10.2          | Default with JetPack 4.6.1             |
| cuDNN                 | 8.2.1         | Default with JetPack 4.6.1             |
| Python3               | 3.6.9         | Default with JetPack 4.6.1             |
| NumPy                 | 1.19.5        | Required for numerical operations      |
| OpenCV (CUDA support) | 4.5.1         | Image and video processing             |
| PyTorch               | 1.8.0         | Deep learning framework                |
| TorchVision           | 0.9.0         | Image-specific utilities for PyTorch   |
| ONNX                  | 1.11.0        | Interoperability format for models     |

## Issues faced and their solution
### 1. Error in installing PyCUDA and NVCC Not Found in PATH
Check out the king [DONKEY CAR](https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/)

Setup some environment variables so nvcc is on $PATH. Add the following lines to your ~/.bashrc file.
```bash
# Add this to your .bashrc file
export CUDA_HOME=/usr/local/cuda
# Adds the CUDA compiler to the PATH
export PATH=$CUDA_HOME/bin:$PATH
# Adds the libraries
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH
```
test changes
```bash
source ~/.bashrc
nvcc --version
```
then switch to virtual environment and
```bash
pip install pycuda
```
### 2. illegal instruction (core dumped)
Add the following lines to your ~/.bashrc file.
```bash
export OPENBLAS_CORETYPE=ARMV8
```
## Converting to tensorRT using trtexec
To optimize the ONNX model for inference on the Jetson Nano using TensorRT, you can generate a TensorRT engine file (`.trt`) with the `trtexec` command-line tool.

**🔧 Command:**
```bash
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.trt --workspace=1024
```
### 1. TensorRT

Run inference using NVIDIA TensorRT for optimized GPU execution.

📄 **Script:** [`trt_inference.py`](Inference%20Scripts/trt_inference.py)

---

### 2. ONNX Runtime (CPU & GPU)

Run inference on ONNX models using ONNX Runtime, supporting both CPU and GPU execution.

📄 **Script:** [`onnx_inference.py`](Inference%20Scripts/onnx_inference.py)