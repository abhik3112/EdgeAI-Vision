import time
import torch
import os
import torchvision.models as models
from torch.ao.quantization import get_default_qconfig, prepare, convert

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def benchmark_inference(model, input_tensor, num_iterations=100):
    model.eval()
    
    # Warm-up runs to clear JIT/cache overhead
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_iterations) * 1000.0
    fps = num_iterations / total_time

    return avg_latency_ms, fps

if __name__ == "__main__":
    # Dummy input representing a single video frame / image
    input_tensor = torch.randn(1, 3, 224, 224)

    # 1. Profile Baseline FP32 Model
    fp32_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    torch.save(fp32_model.state_dict(), "temp_fp32.pt")
    fp32_size = get_file_size_mb("temp_fp32.pt")
    fp32_latency, fp32_fps = benchmark_inference(fp32_model, input_tensor)
    os.remove("temp_fp32.pt")

    # 2. Profile Quantized INT8 Model
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = models.mobilenet_v3_small(weights=None)
    quantized_model.qconfig = get_default_qconfig('qnnpack')
    quantized_model = prepare(quantized_model, inplace=False)
    quantized_model = convert(quantized_model, inplace=False)
    
    # Save & measure quantized size
    torch.save(quantized_model.state_dict(), "temp_int8.pt")
    int8_size = get_file_size_mb("temp_int8.pt")
    int8_latency, int8_fps = benchmark_inference(quantized_model, input_tensor)
    os.remove("temp_int8.pt")

    # 3. Print Professional Comparison Table
    print("\n" + "="*50)
    print(f"{'Performance Metric':<25} | {'FP32 Baseline':<12} | {'INT8 Quantized':<12}")
    print("="*50)
    print(f"{'Model Size (MB)':<25} | {fp32_size:<12.2f} | {int8_size:<12.2f}")
    print(f"{'Latency per Frame (ms)':<25} | {fp32_latency:<12.2f} | {int8_latency:<12.2f}")
    print(f"{'Throughput (FPS)':<25} | {fp32_fps:<12.2f} | {int8_fps:<12.2f}")
    print("="*50 + "\n")