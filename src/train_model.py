import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp

VERBOSE = True

# Paths
TRAIN_DIR = r"data/train/processed"
VAL_DIR = r"data/val/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 20
WEIGHT_FOREGROUND = 2

# Dataset Class
class NailSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        mask = (mask > 0).float()
        return image, mask

# Data Augmentation and Preprocessing
def augmentation(image, mask):
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-30, 30)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
    if np.random.rand() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    if np.random.rand() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
    if np.random.rand() > 0.5:
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = transforms.functional.adjust_brightness(image, brightness_factor)
    return image, mask

class Transform:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, mask):
        image, mask = augmentation(image, mask)
        image = image.resize((self.img_width, self.img_height))
        mask = mask.resize((self.img_width, self.img_height))
        return self.to_tensor(image), self.to_tensor(mask)

transform = Transform(IMG_HEIGHT, IMG_WIDTH)

train_dataset = NailSegmentationDataset(
    os.path.join(TRAIN_DIR, "images"),
    os.path.join(TRAIN_DIR, "masks"),
    transform=transform,
)
val_dataset = NailSegmentationDataset(
    os.path.join(VAL_DIR, "images"),
    os.path.join(VAL_DIR, "masks"),
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Pre-built U-Net from segmentation_models_pytorch with EfficientNet
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Tversky Loss for improved precision
def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1):
    pred = pred.contiguous()
    target = target.contiguous()
    TP = (pred * target).sum(dim=(2, 3))
    FP = ((1 - target) * pred).sum(dim=(2, 3))
    FN = (target * (1 - pred)).sum(dim=(2, 3))
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky.mean()

def combined_loss(pred, target):
    pred = torch.sigmoid(pred)
    bce = nn.BCEWithLogitsLoss()(pred, target)
    tversky = tversky_loss(pred, target)
    return bce + tversky

# Dice Score
def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection / union).item()

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, scheduler=None):
    model.to(DEVICE)
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(pred_probs, masks)

        val_dice /= len(val_loader)

        # Only print epoch progress if VERBOSE is True
        if VERBOSE:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Dice: {val_dice:.4f}")

        if scheduler:
            scheduler.step(val_loss)

        if VERBOSE:
            visualize_predictions(model, val_dataset)

# Visualization Function
def visualize_predictions(model, dataset, num_samples=3):
    model.eval()
    for i in range(num_samples):
        image, mask = dataset[i]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze().cpu().numpy(), cmap="gray")
        plt.title("Ground Truth Mask")
        plt.subplot(1, 3, 3)
        plt.imshow(pred.squeeze() > 0.5, cmap="gray")
        plt.title("Predicted Mask")
        plt.show()

# Initialize and Train
criterion = combined_loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, scheduler)

# Directory for saving the final model
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save the model
final_model_path = os.path.join(SAVE_DIR, "nail_segmentation_unet.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Model saved at '{final_model_path}'")
