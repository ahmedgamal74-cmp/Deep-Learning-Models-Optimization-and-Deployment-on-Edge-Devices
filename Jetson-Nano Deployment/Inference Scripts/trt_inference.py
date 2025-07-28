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
MODEL = "mobilenet_pruned_fastnas.trt"
BATCH_SIZE = 1
TEST_IMAGES = 1000 # MAX 10000
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

# Run inference on a single image
def infer(context, bindings, inputs, outputs, stream, input_image):
    cuda.memcpy_htod_async(inputs[0]['device_mem'], input_image, stream)
    context.execute_async(batch_size=BATCH_SIZE, bindings=bindings, stream_handle=stream.handle)
    output = np.empty([1, 10], dtype=np.float32)
    cuda.memcpy_dtoh_async(output, outputs[0]['device_mem'], stream)
    stream.synchronize()
    return output

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
warmup_data_index = list(range(1000))
reduced_data_index = list(range(1000, 1000+TEST_IMAGES))
warmup_dataset = Subset(cifar10, warmup_data_index)
reduced_dataset = Subset(cifar10, reduced_data_index)
warmup_loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=False)
correct = 0
total = 0

# WARMUP
print("Warming up...")
for i, (img, _) in enumerate(warmup_loader):
    dummy_input = img.numpy().astype(np.float32)
    infer(context, bindings, inputs, outputs, stream, dummy_input)
    if i % 100 == 0:
    	print(f"warming up, used {i+100} images")
print("Warmup complete.\n")

# Start timing
start_time = time.time()

for i, (img, label) in enumerate(test_loader):
    img_np = img.numpy().astype(np.float32)
    output = infer(context, bindings, inputs, outputs, stream, img_np)
    pred = np.argmax(output)
    if pred == label.item():
        correct += 1
    total += 1
    if i % 100 == 0:
        print(f"Processed {i+100} images")

# End timing
end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / total

print(f"\nAccuracy: {100 * correct / total:.2f}%")
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time * 1000:.2f} ms")
