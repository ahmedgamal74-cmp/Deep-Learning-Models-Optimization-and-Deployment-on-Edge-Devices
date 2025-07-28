import argparse
import time
import numpy as np
import os
import psutil
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score

# Optional imports (only loaded when needed)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

class ModelRunner:
    def __init__(self, args):
        self.args = args
        self.setup_transforms()
        self.load_dataset()
        self.model_size = self.get_model_size()
        
        if args.engine == 'onnx':
            if not ONNX_AVAILABLE:
                raise RuntimeError("ONNX Runtime not available")
            self.setup_onnx()
        elif args.engine == 'tensorrt':
            if not TRT_AVAILABLE:
                raise RuntimeError("TensorRT not available")
            self.setup_tensorrt()
        else:
            raise ValueError(f"Unknown engine: {args.engine}")

    def get_model_size(self):
        """Get model size in MB"""
        if not os.path.exists(self.args.model_path):
            return 0
        return os.path.getsize(self.args.model_path) / (1024 * 1024)

    def setup_transforms(self):
        """Setup image transformations"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def load_dataset(self):
        """Load CIFAR-10 dataset"""
        dataset = CIFAR10(root='./data', train=False, transform=self.transform, download=True)
        
        # Split into warmup and test subsets
        warmup_indices = list(range(self.args.warmup_size))
        test_indices = list(range(self.args.warmup_size, 
                                 self.args.warmup_size + self.args.test_size))
        
        self.warmup_loader = DataLoader(
            Subset(dataset, warmup_indices),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            Subset(dataset, test_indices),
            batch_size=self.args.batch_size,
            shuffle=False
        )

    def setup_onnx(self):
        """Initialize ONNX runtime session"""
        provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }

        providers = ['CPUExecutionProvider']
        if self.args.device == 'gpu':
            providers.insert(0, ('CUDAExecutionProvider', provider_options))

        self.session = ort.InferenceSession(self.args.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def setup_tensorrt(self):
        """Initialize TensorRT engine"""
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        
        def load_engine(engine_path):
            with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

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

        self.engine = load_engine(self.args.model_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def preprocess_batch(self, batch):
        """Convert batch to correct format for the engine"""
        batch = batch.numpy().astype(np.float32)
        if self.args.engine == 'onnx' and batch.shape[1] != 3:
            batch = np.transpose(batch, (0, 3, 1, 2))  # NHWC to NCHW
        return batch

    def infer_onnx(self, batch):
        """Run ONNX inference"""
        return self.session.run([self.output_name], {self.input_name: batch})[0]

    def infer_tensorrt(self, batch):
        """Run TensorRT inference"""
        cuda.memcpy_htod_async(self.inputs[0]['device_mem'], batch, self.stream)
        self.context.execute_async(
            batch_size=self.args.batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        output = np.empty([self.args.batch_size, 10], dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.outputs[0]['device_mem'], self.stream)
        self.stream.synchronize()
        return output

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def warmup(self):
        """Warmup the inference engine"""
        print("Warming up...")
        for imgs, _ in self.warmup_loader:
            batch = self.preprocess_batch(imgs)
            if self.args.engine == 'onnx':
                self.infer_onnx(batch)
            else:
                self.infer_tensorrt(batch)
        print("Warmup complete.")

    def run_benchmark(self):
        """Run benchmark and return metrics"""
        correct = 0
        total = 0
        latency_list = []
        all_preds = []
        all_labels = []
        memory_usage = []
        
        print("Running inference...")
        start_time = time.perf_counter()
        
        for imgs, labels in self.test_loader:
            batch_start = time.perf_counter()
            
            batch = self.preprocess_batch(imgs)
            
            if self.args.engine == 'onnx':
                outputs = self.infer_onnx(batch)
            else:
                outputs = self.infer_tensorrt(batch)
                
            preds = np.argmax(outputs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            correct += (preds == labels.numpy()).sum()
            total += len(labels)
            
            latency_list.append(time.perf_counter() - batch_start)
            memory_usage.append(self.get_memory_usage())
        
        total_time = time.perf_counter() - start_time
        
        # Calculate metrics
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        throughput = total / total_time
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        
        return {
            'accuracy': 100 * correct / total,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_time': total_time,
            'avg_latency': np.mean(latency_list),
            'throughput': throughput,
            'model_size_mb': self.model_size,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'engine': self.args.engine,
            'device': self.args.device if self.args.engine == 'onnx' else 'gpu'
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Generic Model Inference Script')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model file (.onnx or .trt)')
    parser.add_argument('--engine', type=str, required=True,
                        choices=['onnx', 'tensorrt'],
                        help='Inference engine to use')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'],
                        help='Device to use for ONNX (ignored for TensorRT)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--warmup-size', type=int, default=100,
                        help='Number of warmup images')
    parser.add_argument('--test-size', type=int, default=1000,
                        help='Number of test images')
    return parser.parse_args()

def main():
    args = parse_args()
    
    runner = ModelRunner(args)
    runner.warmup()
    results = runner.run_benchmark()
    
    print("\n===== Benchmark Results =====")
    print(f"Engine: {results['engine'].upper()} ({'GPU' if results['device'] == 'gpu' else 'CPU'})")
    print(f"Model Size: {results['model_size_mb']:.2f} MB")
    print("\n===== Performance Metrics =====")
    print(f"Total inference time: {results['total_time']:.2f} seconds")
    print(f"Average latency: {results['avg_latency'] * 1000:.2f} ms")
    print(f"Throughput: {results['throughput']:.2f} images/sec")
    print("\n===== Memory Usage =====")
    print(f"Average memory: {results['avg_memory_mb']:.2f} MB")
    print(f"Peak memory: {results['max_memory_mb']:.2f} MB")
    print("\n===== Accuracy Metrics =====")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == '__main__':
    main()
