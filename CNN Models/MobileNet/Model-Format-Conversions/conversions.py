import torch
import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import cv2
import time

def torch2onnx(onnx_path, model, device, opset_version=17):
    """
    Converts a PyTorch model to the ONNX format and saves it to the specified path.

    Parameters:
    -----------
    onnx_path : str
        The file path where the ONNX model will be saved.
    model : torch.nn.Module
        The PyTorch model to be converted.
    device : torch.device
        The device (CPU or GPU) on which the model should be placed before export.
    opset_version : int, optional
        The ONNX opset version to use for the export (default is 17).

    Exceptions:
    -----------
    - Prints an error message if the export process fails.

    Example:
    --------
    >>> model = MyModel()  # Replace with your PyTorch model
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> torch2onnx("model.onnx", model, device)
    """
    try:
        # Export the model to ONNX
        torch.onnx.export(
            model=model,
            args=torch.randn(1, 3, 224, 224).to(device),
            f=onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            opset_version=opset_version
        )
        print(f"Model successfully exported to: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")

def check_onnx_model(onnx_path):
    """
    Validates an ONNX model by checking its structure and consistency.

    Parameters:
    -----------
    onnx_path : str
        The file path of the ONNX model to be validated.

    Exceptions:
    -----------
    - Prints an error message if the ONNX model check fails.

    Example:
    --------
    >>> check_onnx_model("model.onnx")
    """
    try:
        onnx.checker.check_model(onnx_path)
        print(f"Model successfully checked: {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX model check: {e}")



def onnx2trt(onnx_path, trt_path, fp_16=True):
    """
    Converts an ONNX model to a TensorRT engine for optimized deployment.

    Parameters:
    -----------
    onnx_path : str
        Path to the input ONNX model file.
        
    trt_path : str
        Path to save the output TensorRT engine file.

    fp_16 : bool, optional (default=True)
        Enable FP16 precision if supported by the hardware for faster inference.

    Description:
    ------------
    This function converts a given ONNX model into a TensorRT engine,
    which is highly optimized for NVIDIA GPUs. It supports FP16 precision
    if the hardware allows, reducing memory consumption and improving performance.

    Notes:
    ------
    - The function uses a 1GB memory workspace limit (`1 << 30` bytes).
    - FP16 optimization is enabled by default if supported.
    - TODO: Future enhancement includes adding INT8 support for further optimization.

    Example Usage:
    --------------
    >>> import tensorrt as trt
    >>> onnx2trt("model.onnx", "model.trt")
    TensorRT engine saved to model.trt

    Raises:
    -------
    RuntimeError: If there are issues parsing the ONNX model.
    """
    # Create TensorRT Logger to capture warnings and errors
    logger = trt.Logger(trt.Logger.WARNING)

    # Build TensorRT engine from ONNX
    # Create TensorRT builder (Main object responsible for building the TensorRT engine.)
    builder = trt.Builder(logger)

    # Initializes a TensorRT network.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Parses the ONNX model into a TensorRT-compatible structure.
    parser = trt.OnnxParser(network, logger)

    # Load the ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(1)

    # Set the builder configuration
    config = builder.create_builder_config()

    # Set workspace size (TensorRT 10+ uses set_memory_pool_limit) 1GB workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Set precision (FP16 if supported)
    if builder.platform_has_fast_fp16 and fp_16:
        print("FP16 supported, enabling FP16 optimization.")
        config.set_flag(trt.BuilderFlag.FP16)
    # Todo: Add INT8 support

    # Build and serialize the engine
    print("Building TensorRT engine. This might take a while...")
    serialized_engine = builder.build_serialized_network(network, config)

    # Save the engine
    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {trt_path}")

def load_engine(engine_path, logger):
    """
    Load a TensorRT engine from a serialized file.

    Parameters:
    - engine_path (str): Path to the serialized TensorRT engine (.plan file).
    - logger (trt.Logger): TensorRT logger instance for error and warning reporting.

    Returns:
    - trt.ICudaEngine: Deserialized TensorRT engine ready for inference.
    """
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(f.read())

    
def infer(batch, input_shape, output_shape, context, d_output):
    """
    Run inference on a batch of images using a TensorRT context.

    Parameters:
    - batch (np.ndarray): Input batch of images (shape: (batch_size, C, H, W)).
    - input_shape (tuple): Expected input tensor shape (C, H, W).
    - output_shape (tuple): Expected output tensor shape.
    - context (trt.IExecutionContext): TensorRT execution context for inference.
    - d_output (cuda.DeviceAllocation): Pre-allocated GPU memory for output data.

    Returns:
    - np.ndarray: Stacked predictions from the model for the entire batch.
    """
    batch = batch.astype(np.float32)
    predictions = []
    for img in batch:
        # Ensure image shape matches (1, 3, 224, 224)
        img = img.reshape(input_shape)

        #print(f"Processing image with shape: {img.shape}")

        # Allocate memory for each image
        d_input = cuda.mem_alloc(img.nbytes)

        # Copy to GPU memory
        cuda.memcpy_htod(d_input, img)

        # Execute inference
        context.execute_v2(bindings=[int(d_input), int(d_output)])

        # Retrieve results
        output_data = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_data, d_output)
        predictions.append(output_data)
    return np.vstack(predictions)


def trt_inference(engine_path, data_loader):
    """
    Perform TensorRT inference on a dataset and evaluate model accuracy and speed.

    Parameters:
    - engine_path (str): Path to the serialized TensorRT engine (.plan file).
    - data_loader (torch.utils.data.DataLoader): DataLoader providing input images and labels.

    Returns:
    - tuple: (accuracy (float), inference_time (float))
    """
    # Load TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_path, logger)
    context = engine.create_execution_context()
    # Get TensorRT input/output shapes
    input_shape = context.get_tensor_shape(engine.get_tensor_name(0))
    output_shape = context.get_tensor_shape(engine.get_tensor_name(1))
    # Allocate GPU memory for input/output
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().nbytes))
    output_data = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)
    # Evaluate on CIFAR-10 test set
    correct = 0
    total = 0
    print("Running inference on the test dataset...")
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.numpy()

            # Ensure input shape is (batch_size, 3, 224, 224)
            if images.shape[2:] != (224, 224):
                images = np.array([cv2.resize(img.transpose(1, 2, 0), (224, 224)).transpose(2, 0, 1) for img in images])

            # Run inference
            predictions = infer(images, input_shape, output_shape, context, d_output)

            # Get predicted class for each sample
            predicted_classes = np.argmax(predictions, axis=1)

            # Update accuracy
            correct += (predicted_classes == labels.numpy()).sum()
            total += labels.size(0)
    stop_time = time.time()
    inference_time = stop_time - start_time
    # Print accuracy
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Inference Time: {inference_time:.2f} seconds")
    return accuracy, inference_time
