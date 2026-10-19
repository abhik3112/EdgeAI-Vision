import os
import torch
import torchvision.models as models
from torch.ao.quantization import get_default_qconfig, prepare_qat, prepare, convert
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

def get_model():
    # Load a lightweight pre-trained vision model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()
    return model

def calibrate_and_quantize(model, calibration_dataloader):
    """
    Applies Post-Training Static Quantization (PTQ) to a PyTorch model.
    """
    print("[-] Starting quantization preparation...")
    
    # 1. Set the backend engine (fbgemm for x86 server/dev, qnnpack for ARM edge devices like Snapdragon/Raspberry Pi)
    torch.backends.quantized.engine = 'qnnpack'

    # 2. Fuse modules where applicable (Conv + BN + ReLU fusion speeds up inference)
    # For MobileNetV3, PyTorch provides fused variants or manual fusion mapping.
    model.qconfig = get_default_qconfig('qnnpack')
    
    # 3. Prepare the model for quantization (inserts observers)
    prepared_model = prepare(model, inplace=False)
    prepared_model.eval()

    # 4. Calibrate with representative data (feeds dummy or real input to compute scales & zero-points)
    print("[-] Running calibration...")
    with torch.no_grad():
        for images, _ in calibration_dataloader:
            prepared_model(images)

    # 5. Convert to quantized model (transforms weights and activations to INT8)
    print("[-] Converting to INT8 quantized model...")
    quantized_model = convert(prepared_model, inplace=False)
    
    return quantized_model

if __name__ == "__main__":
    # 1. Initialize model
    model = get_model()

    # 2. Create synthetic calibration data (Replace with a small dataloader of actual images)
    # Shape: [Batch_Size, Channels, Height, Width] -> e.g., Batch of 32 images (3, 224, 224)
    dummy_calibration_data = torch.randn(32, 3, 224, 224)
    calib_dataset = TensorDataset(dummy_calibration_data, torch.zeros(32))
    calib_loader = DataLoader(calib_dataset, batch_size=8)

    # 3. Quantize
    quantized_model = calibrate_and_quantize(model, calib_loader)

    # 4. Save the quantized model state dict
    os.makedirs("../weights", exist_ok=True)
    torch.save(quantized_model.state_dict(), "../weights/mobilenet_v3_quantized.pt")
    print("[+] Quantized model successfully saved to weights/mobilenet_v3_quantized.pt")