# Nail Segmentation with Deep Learning

This repository provides a complete pipeline for training, testing, and deploying a deep learning model for nail segmentation using **PyTorch** and **ONNX**. The project includes scripts for data preparation, model training, testing, and exporting the trained model for deployment.

---

## Important Note

This repository does not include datasets or pretrained models. To use this repository:

1. **Download a Dataset**: 
   - Obtain any dataset suitable for nail segmentation.
   - Export the dataset in COCO segmentation format.

2. **Organize the Dataset**: 
   - Extract the dataset ZIP file.
   - Create a new folder named `data` in the project root directory if it doesn't already exist.
   - Move the `train`, `valid`, and `test` folders from the extracted dataset into the `data/` directory.

3. **Prepare the Data**:
   - Run the `prepare_data.py` script to preprocess and organize the dataset:
     ```bash
     python src/prepare_data.py
     ```

4. **Train the Model**:
   - Train the segmentation model using the following command:
     ```bash
     python src/train_model.py
     ```

5. **Test the Model**:
   - Run predictions on one of the test images or upload your own image to the `images/` directory:
     ```bash
     python src/test_model.py
     ```

6. **Export the Model**:
   - Convert the trained PyTorch model to ONNX format for deployment:
     ```bash
     python src/convert_to_onnx.py
     ```

7. **Test the ONNX Model**:
   - Verify the ONNX model output:
     ```bash
     python src/test_onnx.py
     ```

---

## Project Structure

- **`src/prepare_data.py`**: Prepares the dataset by resizing images and masks and organizing them for training and validation.
- **`src/train_model.py`**: Trains the segmentation model using the U-Net architecture with an EfficientNet encoder.
- **`src/test_model.py`**: Tests the PyTorch model on sample images and visualizes the predictions.
- **`src/convert_to_onnx.py`**: Converts the trained PyTorch model into ONNX format for deployment.
- **`src/test_onnx.py`**: Tests the exported ONNX model and verifies its output.
- **`models/`**: Directory to store trained models (PyTorch and ONNX).

---

## Features

- **Data Preparation**:
  - Automatically processes and organizes images and masks.
  - Supports debugging visualization with a `DEBUG` flag.

- **Model Training**:
  - U-Net architecture with an EfficientNet-B0 encoder.
  - Supports Tversky loss for improved precision.
  - Progress printing can be toggled on or off with the `VERBOSE` flag.

- **Model Export**:
  - Exports the trained model to ONNX format for cross-platform deployment.

- **Live Deployment**:
  - Compatible with Unity using Barracuda for real-time webcam-based nail segmentation.

---

## Requirements

- **Python Libraries**:
  - `torch`, `torchvision`, `segmentation_models_pytorch`, `numpy`, `matplotlib`, `tqdm`, `Pillow`.

Install dependencies using:
```bash
pip install -r requirements.txt
