import torch
import torchvision.models as models

def load_mobilenet_v3(pretrained=True):
    """
    Loads MobileNetV3-Small architecture.
    MobileNetV3 is optimized for mobile hardware and edge accelerators.
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        print("[+] Loaded pre-trained MobileNetV3-Small weights.")
    else:
        model = models.mobilenet_v3_small(weights=None)
        print("[+] Initialized MobileNetV3-Small with random weights.")

    # Set model to evaluation mode (disables Dropout and fixes Batch Normalization layers)
    model.eval()
    return model

if __name__ == "__main__":
    # Quick sanity check
    model = load_mobilenet_v3(pretrained=True)
    
    # Create a dummy image tensor matching standard ImageNet dimensions [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"[+] Model Output Shape: {output.shape} (Matches 1000 ImageNet classes)")