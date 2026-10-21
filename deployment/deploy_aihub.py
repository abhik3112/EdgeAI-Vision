import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import qai_hub as hub
import torch
from models.base_model import load_mobilenet_v3

def run_qualcomm_cloud_deployment():
    print("[-] Step 1: Loading PyTorch MobileNetV3-Small model...")
    model = load_mobilenet_v3(pretrained=True)
    model.eval()

    # Define input specs matching your preprocessing shape [1, 3, 224, 224]
    input_shape = (1, 3, 224, 224)
    sample_input = torch.randn(*input_shape)

    print("[-] Step 1.1: Tracing model for AI Hub export...")
    # Trace the PyTorch model into TorchScript format as required by qai_hub
    traced_model = torch.jit.trace(model, sample_input)

    # Choose a target device from your list (e.g., Samsung Galaxy S24 running Snapdragon 8 Gen 3)
    target_device_name = "Samsung Galaxy S24"
    print(f"[-] Step 2: Targeting physical device: {target_device_name}...")
    device = hub.Device(target_device_name)

    print("[-] Step 3: Submitting model compilation job to Qualcomm AI Hub...")
    compile_job = hub.submit_compile_job(
        model=traced_model,  # Pass the traced torch script model here!
        input_specs=dict(image=input_shape),
        device=device,
    )
    print(f"[+] Compile Job Submitted! ID: {compile_job.job_id}")
    print("[-] Waiting for compilation to finish on Qualcomm cloud...")
    
    target_model = compile_job.get_target_model()
    print("[+] Compilation complete! Model optimized for Hexagon NPU.")

    print("[-] Step 4: Submitting on-device profiling job...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    print(f"[+] Profile Job Submitted! ID: {profile_job.job_id}")
    print("[-] Waiting for profiling results...")

    profile_report = profile_job.download_profile()
    print("\n" + "="*50)
    print("🚀 QUALCOMM AI HUB ON-DEVICE HARDWARE REPORT 🚀")
    print("="*50)
    print(profile_report)
    print("="*50)

if __name__ == "__main__":
    run_qualcomm_cloud_deployment()