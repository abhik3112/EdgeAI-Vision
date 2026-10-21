import os
import sys
import torch
import torchvision.models as models
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to import from models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import load_mobilenet_v3


def calibrate_model(prepared_model, calibration_loader):
    """
    Passes representative calibration data through the model.
    The observers inserted during prepare() record the minimum and maximum
    activation values to calculate scale factors and zero-points for INT8 conversion.
    """
    print("[-] Running calibration over representative dataset...")
    prepared_model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared_model(images)
    print("[+] Calibration complete.")


def run_static_quantization():
    print("[-] Step 1: Loading FP32 MobileNetV3 model...")
    fp32_model = load_mobilenet_v3(pretrained=True)

    # 1. Set backend engine: 'qnnpack' is engineered for ARM architecture (Qualcomm Snapdragon/Raspberry Pi)
    torch.backends.quantized.engine = 'qnnpack'

    # 2. Assign Quantization Configuration (QConfig) for INT8 activation & weight observers
    fp32_model.qconfig = get_default_qconfig('qnnpack')

    # 3. Prepare model: Inserts observer modules into layers to monitor activation distributions
    print("[-] Step 2: Preparing model for quantization (attaching observers)...")
    prepared_model = prepare(fp32_model, inplace=False)

    # 4. Create dummy calibration data (32 synthetic images matching [Batch, Channels, Height, Width])
    # In production, this would be 100-500 representative real images from your domain dataset.
    calibration_data = torch.randn(32, 3, 224, 224)
    calib_dataset = TensorDataset(calibration_data, torch.zeros(32))
    calib_loader = DataLoader(calib_dataset, batch_size=8)

    # 5. Calibrate
    calibrate_model(prepared_model, calib_loader)

    # 6. Convert: Transforms FP32 operators into INT8 quantized counterparts
    print("[-] Step 3: Converting model layers from FP32 to INT8 precision...")
    quantized_model = convert(prepared_model, inplace=False)

    # 7. Save weights
    os.makedirs("../weights", exist_ok=True)
    quant_weights_path = "../weights/mobilenet_v3_quantized.pt"
    torch.save(quantized_model.state_dict(), quant_weights_path)
    print(f"[+] Static INT8 Quantization successful! Saved to {quant_weights_path}")

    return quantized_model


if __name__ == "__main__":
    run_static_quantization()