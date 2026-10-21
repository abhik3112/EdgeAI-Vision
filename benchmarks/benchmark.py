import os
import sys
import time
import numpy as np
import torch
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import load_mobilenet_v3


def get_file_size_mb(filepath):
    """Calculates file size in Megabytes."""
    return os.path.getsize(filepath) / (1024 * 1024)


def benchmark_pytorch(model, input_tensor, num_iterations=100):
    model.eval()
    with torch.no_grad():
        for _ in range(15):
            _ = model(input_tensor)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    total_time = time.time() - start_time

    avg_latency = (total_time / num_iterations) * 1000.0
    fps = num_iterations / total_time
    return avg_latency, fps


def benchmark_onnx(onnx_path, dummy_np_input, num_iterations=100):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Warm-up runs
    for _ in range(15):
        session.run(None, {input_name: dummy_np_input})

    start_time = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_name: dummy_np_input})
    total_time = time.time() - start_time

    avg_latency = (total_time / num_iterations) * 1000.0
    fps = num_iterations / total_time
    return avg_latency, fps


def run_benchmark():
    onnx_path = "weights/mobilenet_v3.onnx"
    if not os.path.exists(onnx_path):
        print("[!] ONNX model not found. Run `python quantization/export_onnx.py` first.")
        return

    # PyTorch Setup
    dummy_torch = torch.randn(1, 3, 224, 224)
    py_model = load_mobilenet_v3(pretrained=True)

    weights_dir = "weights"
    temp_pt_path = os.path.join(weights_dir, "temp_mobilenet.pt")
    torch.save(py_model.state_dict(), temp_pt_path)
    pt_size = get_file_size_mb(temp_pt_path)
    if os.path.exists(temp_pt_path):
        os.remove(temp_pt_path)

    onnx_size = get_file_size_mb(onnx_path)

    print("[-] Benchmarking PyTorch FP32 Baseline...")
    py_latency, py_fps = benchmark_pytorch(py_model, dummy_torch)

    print("[-] Benchmarking ONNX Runtime Engine...")
    dummy_np = dummy_torch.numpy().astype(np.float32)
    onnx_latency, onnx_fps = benchmark_onnx(onnx_path, dummy_np)

    speedup = py_latency / onnx_latency if onnx_latency > 0 else 1.0

    print("\n" + "=" * 65)
    print(f"{'Performance Metric':<25} | {'PyTorch FP32':<15} | {'ONNX Runtime':<15}")
    print("=" * 65)
    print(f"{'Model File Size (MB)':<25} | {pt_size:<15.2f} | {onnx_size:<15.2f}")
    print(f"{'Latency per Frame (ms)':<25} | {py_latency:<15.2f} | {onnx_latency:<15.2f}")
    print(f"{'Throughput (FPS)':<25} | {py_fps:<15.2f} | {onnx_fps:<15.2f}")
    print("=" * 65)
    print(f"[+] Execution Speedup via Graph Optimization: {speedup:.2f}x")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_benchmark()