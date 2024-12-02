import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp

# Define directory for saving models
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Paths for PyTorch and ONNX models
MODEL_PATH = os.path.join(SAVE_DIR, "nail_segmentation_unet.pth")
IMAGE_PATH = "images/HandPicture.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT, IMG_WIDTH = 1024, 1024

# Load the model
print("Loading model...")
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocess the input image
def preprocess_image(image_path, img_height, img_width):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_size

# Postprocess the predicted mask
def postprocess_mask(mask_tensor, original_size):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)
    return mask_image

# Postprocess the predicted heatmap
def postprocess_heatmap(heatmap_tensor, original_size):
    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_size, Image.Resampling.BICUBIC)
    return heatmap_resized

# Predict
print("Predicting...")
input_image, original_size = preprocess_image(IMAGE_PATH, IMG_HEIGHT, IMG_WIDTH)
input_image = input_image.to(DEVICE)

with torch.no_grad():
    prediction = model(input_image)
    prediction = torch.sigmoid(prediction)

# Postprocess mask and heatmap
predicted_mask = postprocess_mask(prediction, original_size)
predicted_heatmap = postprocess_heatmap(prediction, original_size)

# Create masked image (overlay mask on original)
original_image = Image.open(IMAGE_PATH).convert("RGB")
overlay = np.array(original_image).astype(np.float32) * 0.7 + np.array(predicted_mask.convert("RGB")).astype(np.float32) * 0.3
overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# Visualize results
plt.figure(figsize=(15, 5))

# Masked Image
plt.subplot(1, 3, 1)
plt.imshow(overlay)
plt.title("Masked Image")
plt.axis("off")

# Predicted Mask
plt.subplot(1, 3, 2)
plt.imshow(predicted_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

# Predicted Heatmap
plt.subplot(1, 3, 3)
plt.imshow(predicted_heatmap, cmap="hot", interpolation="nearest")
plt.title("Predicted Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()
