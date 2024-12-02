import os
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Configuration
BASE_DIRS = {
    "train": r"data/train",
    "test": r"data/test",
    "val": r"data/valid"
}
IMG_SIZE = (256, 256)
DEBUG = False
NUM_WORKERS = 4

def create_binary_mask(annotation, image_size):
    """
    Create a binary mask from segmentation polygons.
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in annotation["segmentation"]:
        draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask)

def process_image(image_meta, coco_data, base_dir, output_dir):
    """
    Process a single image and generate its mask.
    """
    image_id = image_meta["id"]
    image_filename = image_meta["file_name"]
    image_path = os.path.join(base_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found, skipping...")
        return

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Generate binary mask
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
    mask = np.zeros((image_meta["height"], image_meta["width"]), dtype=np.uint8)
    for annotation in annotations:
        mask += create_binary_mask(annotation, (image_meta["width"], image_meta["height"]))

    if DEBUG:
        # Visualize the image and its mask
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Generated Mask")
        plt.show()

    # Resize and save processed data
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(IMG_SIZE, Image.Resampling.NEAREST)
    image_resized = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    image_resized.save(os.path.join(output_dir, "images", image_filename))
    mask_resized.save(os.path.join(output_dir, "masks", image_filename))

    print(f"Processed: {image_filename}")

def process_dataset(base_dir, output_dir):
    """
    Process an entire dataset (train, test, or val).
    """
    annotations_file = os.path.join(base_dir, "_annotations.coco.json")
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        return

    with open(annotations_file, "r") as file:
        coco_data = json.load(file)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Parallel processing for large datasets
    print(f"Processing {len(coco_data['images'])} images from {base_dir}...")
    args = [(img, coco_data, base_dir, output_dir) for img in coco_data["images"]]
    with Pool(NUM_WORKERS) as pool:
        pool.starmap(process_image, args)

    print(f"Processed dataset saved to: {output_dir}")

# Protect the entry point
if __name__ == "__main__":
    for dataset_type, base_dir in BASE_DIRS.items():
        output_dir = os.path.join(base_dir, "processed")
        print(f"Processing {dataset_type} dataset...")
        process_dataset(base_dir, output_dir)
