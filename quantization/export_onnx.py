import torch
import torchvision.models as models

def export_to_onnx():
    print("[-] Loading model for ONNX export...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()

    # Create dummy input matching your expected frame dimensions [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "../weights/mobilenet_v3.onnx"

    print(f"[-] Exporting model to ONNX format at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("[+] ONNX export complete successfully!")

if __name__ == "__main__":
    export_to_onnx()