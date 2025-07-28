import onnxruntime as ort
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import time

# Parameters
MODEL_PATH = "mobilenet_cifar10.onnx"
BATCH_SIZE = 4
WARMUP_SIZE = 100
TEST_SIZE = 1000

# 1. Configure GPU provider options
gpu_options = {
    'device_id': 0,  # Use GPU 0
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}

# 2. Set provider priority (GPU first, then CPU)
providers = [
    ('CUDAExecutionProvider', gpu_options),
    'CPUExecutionProvider'  # Fallback
]

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load CIFAR-10 test set
dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

# Split into warmup and test subsets
warmup_set = Subset(dataset, list(range(WARMUP_SIZE)))
test_set = Subset(dataset, list(range(WARMUP_SIZE, WARMUP_SIZE + TEST_SIZE)))

# DataLoaders
warmup_loader = DataLoader(warmup_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 3. Load ONNX model with GPU priority
try:
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print("Using GPU:", session.get_providers())
except Exception as e:
    print(f"GPU failed: {e}\nFalling back to CPU")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Warmup
print("Warming up GPU...")
for imgs, _ in warmup_loader:
    imgs = imgs.numpy().astype(np.float32)
    # 4. Ensure correct memory layout (NCHW for GPU)
    if imgs.shape[1] != 3:  # If channels aren't first
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # NHWC to NCHW
    session.run([output_name], {input_name: imgs})
print("Warmup complete.")

# Inference + Timing
print("Running inference...")
total = 0
correct = 0
start_time = time.perf_counter()

for imgs, labels in test_loader:
    imgs = imgs.numpy().astype(np.float32)
    # 5. Maintain consistent memory layout
    if imgs.shape[1] != 3:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    
    preds = session.run([output_name], {input_name: imgs})[0]
    preds = np.argmax(preds, axis=1)
    correct += (preds == labels.numpy()).sum()
    total += len(labels)

end_time = time.perf_counter()
avg_time = (end_time - start_time) / total

# Results
print(f"\nAccuracy: {100 * correct / total:.2f}%")
print(f"Total inference time: {end_time - start_time:.2f} s")
print(f"Average time per image: {avg_time * 1000:.2f} ms")
print(f"Execution provider: {session.get_providers()[0]}")
