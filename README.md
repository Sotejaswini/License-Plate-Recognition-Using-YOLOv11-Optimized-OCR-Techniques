# Optimized License Plate Recognition System (OLPRS)

## Project Overview

The Optimized License Plate Recognition System (OLPRS) is an advanced ALPR framework designed specifically for Indian license plates. Leveraging state-of-the-art deep learning, OLPRS integrates YOLOv11n for real-time license plate detection and a hybrid OCR pipeline (LPRNet, EasyOCR, Tesseract, OCR.space API) enhanced by sophisticated preprocessing (CLAHE, bilateral filtering, dynamic thresholding). The system achieves:

- **mAP@0.5:** 0.9275  
- **Character Recognition Rate (CRR):** 88.30%  
- **Plate Recognition Accuracy (PRA):** 76.67% (on 30 test samples)

Deployed via Streamlit for CPU-based real-time inference, OLPRS is optimized for resource-constrained environments and addresses challenges like diverse plate formats, blur, and poor lighting.

## Key Features

- **YOLOv11n Detection:** High-precision, real-time detection using YOLOv11n fine-tuned on a custom dataset of 10,000 Indian plate images.
- **Hybrid OCR Pipeline:** Combines LPRNet, EasyOCR, Tesseract, and OCR.space API with regex-based postprocessing for robust character recognition.
- **Advanced Preprocessing:** CLAHE, bilateral filtering, adaptive thresholding, and more to handle blur, low light, and occlusions.
- **Real-Time Deployment:** Streamlit-based interface for low-latency, CPU-optimized inference.
- **Region-Specific Optimization:** Tailored for Indian license plates, addressing diverse fonts and regional formats.

## Project Structure

```
OLPRS/
├── app_olpr.py                # Streamlit app for real-time inference
├── evaluation.py              # Script for evaluating model performance
├── inference.py               # Script for running inference on test images
├── EasyOCR_Tessaract.ipynb    # Notebook for OCR pipeline development
├── Enhanced_PaddleOCR.ipynb   # Notebook exploring PaddleOCR (not used due to compatibility issues)
├── evaluation_results/        # Evaluation metrics and plots
├── results/                   # Inference outputs
├── runs/                      # YOLO training logs
├── test_indian_plate_inputs/  # Test images
├── test_indian_plate_results/ # Inference results
├── weights/                   # Trained model weights (e.g., best.pt)
results/
├──test_indian_plate_inputs/           
├──test_indian_plate_results/

```

## Installation

### Prerequisites

- Python 3.8+
- Google Colab with T4 GPU (for training)
- CPU-based system (for inference)
- Required libraries: `torch`, `ultralytics`, `easyocr`, `pytesseract`, `streamlit`, `opencv-python`, `numpy`, `pillow`

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sotejaswini/License-Plate-Recognition-Using-YOLOv11-Optimized-OCR-Techniques.git
   cd License-Plate-Recognition-Using-YOLOv11-Optimized-OCR-Techniques
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pretrained Weights:**
   - Download `yolov8n.pt` and `yolov11n.pt`  from the Ultralytics YOLO repository.
   - Place them in the `weights/` directory.

4. **Prepare Datasets:**
   - Download datasets from Roboflow:
     - License Plate Recognition Dataset
   - Organize datasets in YOLO format (70:20:10 split for train/validation/test) and place them in a suitable directory.

5. **Install Tesseract OCR:**
   - For Ubuntu:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - For Windows: Download and install from Tesseract's GitHub.

## Usage

### Training

- Use the `EasyOCR_Tessaract.ipynb` notebook in Google Colab to fine-tune YOLOv11n on the custom datasets.
- Configure hyperparameters as specified (epochs=50, batch=4, img=640, lr=0.0002, etc.).
- Save the best weights to the `weights/` directory.

### Evaluation

- Run the evaluation script to compute metrics:
  ```bash
  python evaluation.py --model OLPR/weights/best_yolo11n.pt --data data/data.yaml --test-img-dir data/test/images --test-label-dir data/test/labels
  
  ```
- Results and visualizations (e.g., confusion matrices, PR curves) are saved in the `evaluation_results/` directory.

 
### Inference

- Run the Streamlit app for real-time recognition:
  ```bash
  streamlit run OLPR/app_olpr.py
  ```
- Or use the inference script for batch processing:
  ```bash
  python OLPR/inference.py --image test_image.jpg --model OLPR/weights/best_yolo11n.pt --output results/
  ```

## Performance

- **Detection:** YOLOv11n achieves mAP@0.5 of 0.9275, outperforming YOLOv8n (0.9040) on Indian license plate datasets.
- **Recognition:** The hybrid OCR pipeline yields a CRR of 88.30% and PRA of 76.67% on 30 test samples.
- **Challenges Handled:** Robust to diverse fonts, lighting variations, blur, and occlusions due to advanced preprocessing.

## Limitations

- Evaluation limited to 30 test samples due to dataset privacy constraints.
- OCR performance may degrade with severe occlusions or non-standard fonts.
- Lack of transformer-based models limits sequence modeling accuracy.
- GPU interruptions during training constrained scalability.

## Future Work

- Expand the dataset with public contributions for broader edge-case coverage.
- Upgrade to paid GPU resources for extended training.
- Integrate lightweight transformer models (e.g., DistilBERT) for improved recognition.
- Develop a mobile app for on-the-go recognition.
- Add multi-lingual support for regional scripts.
- Deploy OLPRS in live traffic systems for real-world validation.

## License

This project is licensed under the MIT License.
