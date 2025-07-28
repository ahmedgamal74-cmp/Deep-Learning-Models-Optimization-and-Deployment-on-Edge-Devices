########################## IMPORTS ###################################
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import time

################### CHANGE PARAMETERS HERE #########################
MODEL = "mobilenet_original.trt"
BATCH_SIZE = 1
TEST_IMAGES = 1000  # MAX 10000
####################################################################

print("Model under evaluation: ", MODEL)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers for input and output
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
        if engine.binding_is_input(binding):
            inputs.append({'binding': binding, 'device_mem': device_mem})
        else:
            outputs.append({'binding': binding, 'device_mem': device_mem})
        bindings.append(int(device_mem))
    return inputs, outputs, bindings, stream

# Run inference on a single image with timing
def infer(context, bindings, inputs, outputs, stream, input_image):
    start_time = time.perf_counter()
    
    cuda.memcpy_htod_async(inputs[0]['device_mem'], input_image, stream)
    context.execute_async(batch_size=BATCH_SIZE, bindings=bindings, stream_handle=stream.handle)
    output = np.empty([1, 10], dtype=np.float32)
    cuda.memcpy_dtoh_async(output, outputs[0]['device_mem'], stream)
    stream.synchronize()
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    
    return output, inference_time

# Main
engine = load_engine(MODEL)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Load CIFAR-10 and preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
cifar10 = CIFAR10(root='./data', train=False, transform=transform, download=True)
subset_index = list(range(TEST_IMAGES))
reduced_dataset = Subset(cifar10, subset_index)
loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=False)

correct = 0
total = 0
inference_times = []

# WARMUP
print("Warming up...")
for i, (img, _) in enumerate(loader):
    if i == 100: 
        break
    dummy_input = img.numpy().astype(np.float32)
    _ = infer(context, bindings, inputs, outputs, stream, dummy_input)[0]  # Discard timing during warmup
print("Warmup complete.\n")

# Main inference loop
print("Starting inference...")
for i, (img, label) in enumerate(loader):
    img_np = img.numpy().astype(np.float32)
    output, inf_time = infer(context, bindings, inputs, outputs, stream, img_np)
    inference_times.append(inf_time)
    
    pred = np.argmax(output)
    if pred == label.item():
        correct += 1
    total += 1
    
    if i % 100 == 0:
        print(f"Processed {i} images | Current avg time: {np.mean(inference_times)*1000:.2f} ms")

# Calculate statistics
total_time = sum(inference_times)
avg_time = np.mean(inference_times)
min_time = np.min(inference_times)
max_time = np.max(inference_times)
std_dev = np.std(inference_times)

print(f"\nPerformance Metrics:")
print(f"Accuracy: {100 * correct / total:.2f}%")
print(f"\nInference Timing (per image):")
print(f"Total time: {total_time:.4f} seconds")
print(f"Average time: {avg_time * 1000:.4f} ms")
print(f"Minimum time: {min_time * 1000:.4f} ms")
print(f"Maximum time: {max_time * 1000:.4f} ms")
print(f"Standard deviation: {std_dev * 1000:.4f} ms")
print(f"\nThroughput: {total/total_time:.2f} images/second")
