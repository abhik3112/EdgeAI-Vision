import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import load_mobilenet_v3


def export_to_onnx():
    print("[-] Loading FP32 MobileNetV3 model for ONNX export...")
    model = load_mobilenet_v3(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    os.makedirs("../weights", exist_ok=True)
    onnx_path = "../weights/mobilenet_v3.onnx"

    print(f"[-] Exporting computation graph to ONNX format at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"[+] ONNX Export successfully completed! Saved to {onnx_path}")


if __name__ == "__main__":
    export_to_onnx()