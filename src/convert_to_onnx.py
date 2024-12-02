import os
import torch
import segmentation_models_pytorch as smp

# Define directory for saving models
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Paths for PyTorch and ONNX models
MODEL_PATH = os.path.join(SAVE_DIR, "nail_segmentation_unet.pth")
ONNX_PATH = os.path.join(SAVE_DIR, "nail_segmentation_unet.onnx")

# Load the PyTorch model
model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Define a dummy input tensor
dummy_input = torch.randn(1, 3, 1024, 1024)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print(f"Model exported to ONNX at '{ONNX_PATH}'.")
