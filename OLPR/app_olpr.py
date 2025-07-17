import os
import streamlit as st
import logging
import time
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from easyocr import Reader
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WEIGHTS_DIR = "weights"
DEFAULT_UPLOAD_DIR = "uploads"
DEFAULT_RESULT_DIR = "results"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Ensure directories exist
for directory in [DEFAULT_UPLOAD_DIR, DEFAULT_RESULT_DIR, DEFAULT_WEIGHTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Custom LPRNet class (simplified placeholder - replace with your actual implementation)
class LPRNet:
    def __init__(self, model_path, device="cpu"):
        self.model = None
        self.device = device
        self.model_path = model_path

    def load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                # Load your LPRNet model (replace with actual loading logic)
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"LPRNet model loaded from {model_path}")
                return True
            else:
                logger.warning(f"LPRNet model not found at {model_path}, using default initialization")
                return False
        except Exception as e:
            logger.error(f"Failed to load LPRNet model: {str(e)}")
            return False

    def recognize_plate(self, image):
        if self.model is None:
            return "UNREADABLE", 0.0
        # Placeholder for LPRNet inference
        return "Placeholder", 0.9  # Replace with actual inference logic

class AdvancedLPR:
    def __init__(self, yolo_path, lprnet_path, ocr_api_key):
        self.yolo_path = yolo_path
        self.lprnet_path = lprnet_path
        self.ocr_api_key = ocr_api_key
        self.yolo_model = None
        self.lprnet = LPRNet(lprnet_path)
        self.easyocr_reader = None
        self.device = "cpu"  # Force CPU for Streamlit Cloud
        self.init_models()

    def init_models(self):
        try:
            if not os.path.exists(self.yolo_path):
                st.error(f"YOLO model file not found at {self.yolo_path}. Falling back to yolov8n.pt.")
                logger.error(f"YOLO model file not found at {self.yolo_path}")
                self.yolo_model = YOLO("yolov8n.pt")
            else:
                self.yolo_model = YOLO(self.yolo_path)
                logger.info(f"Loaded YOLO model from {self.yolo_path}")
        except Exception as e:
            st.error(f"Failed to load YOLO: {str(e)}")
            logger.error(f"Failed to load YOLO: {str(e)}")
            self.yolo_model = YOLO("yolov8n.pt")

        try:
            if not self.lprnet.load_model(self.lprnet_path):
                st.error(f"LPRNet model file not found at {self.lprnet_path}. Using default initialization.")
                logger.error(f"LPRNet model file not found at {self.lprnet_path}")
        except Exception as e:
            st.error(f"Failed to initialize LPRNet: {str(e)}")
            logger.error(f"Failed to initialize LPRNet: {str(e)}")

        try:
            self.easyocr_reader = Reader(['en'], gpu=False)  # Force CPU
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize EasyOCR: {str(e)}")
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            self.easyocr_reader = None

    def detect_and_recognize(self, image, conf_threshold=0.5):
        if self.yolo_model is None:
            return None, "YOLO model not loaded"
        try:
            results = self.yolo_model(image, conf=conf_threshold, iou=0.5)
            logger.info(f"Detected {len(results[0].boxes)} objects")
            if len(results[0].boxes) == 0:
                return image, "No plates detected"
            plate_boxes = results[0].boxes.xyxy.cpu().numpy()
            plate_image = image[int(plate_boxes[0][1]):int(plate_boxes[0][3]), int(plate_boxes[0][0]):int(plate_boxes[0][2])]
            if self.lprnet.model is not None:
                text, confidence = self.lprnet.recognize_plate(plate_image)
            elif self.easyocr_reader is not None:
                text = self.easyocr_reader.readtext(plate_image, detail=0)[0] if self.easyocr_reader.readtext(plate_image) else "UNREADABLE"
                confidence = 0.9 if text != "UNREADABLE" else 0.0
            else:
                text = "UNREADABLE"
                confidence = 0.0
            return plate_image, {"text": text, "confidence": confidence}
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return None, f"Error during detection: {str(e)}"

def main():
    st.title("🚗 Optimized License Plate Recognition System")
    st.markdown("### Multi-Engine OCR System for Indian License Plates")
    st.markdown("*Developed for Internship Project*")

    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
    show_all_results = st.sidebar.checkbox("Show All Results", value=False)

    # Initialize session state
    if "lpr" not in st.session_state:
        st.session_state.lpr = AdvancedLPR(
            yolo_path=os.path.join(DEFAULT_WEIGHTS_DIR, "best_yolo.pt"),
            lprnet_path=os.path.join(DEFAULT_WEIGHTS_DIR, "best_lprnet.pth"),
            ocr_api_key=""  # Replace with your OCR.Space API key if using
        )

    # File uploader
    uploaded_file = st.file_uploader("Upload an image (.jpg, .jpeg, .png, .bmp)", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None and st.session_state.lpr is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        temp_path = os.path.join(DEFAULT_UPLOAD_DIR, uploaded_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Uploaded file saved to {temp_path}")
        st.write(f"Uploaded file: {uploaded_file.name}, Size: {uploaded_file.size} bytes")

        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            if st.button("🔍 Analyze License Plates"):
                with st.spinner("🔄 Processing with all OCR engines..."):
                    start_time = time.time()
                    cv_image = cv2.imread(temp_path)
                    if cv_image is None:
                        st.error(f"Failed to read image from {temp_path}")
                        logger.error(f"Failed to read image from {temp_path}")
                    else:
                        processed_image, detection_result = st.session_state.lpr.detect_and_recognize(cv_image, confidence_threshold)
                        if processed_image is not None and isinstance(detection_result, dict):
                            processing_time = time.time() - start_time
                            with col2:
                                st.subheader("🎯 Detection Results")
                                processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                                st.image(processed_rgb, use_container_width=True)
                                st.write(f"Plate: {detection_result['text']} ({detection_result['confidence']*100:.1f}%)")
                                st.write(f"Processing Time: {processing_time:.2f} seconds")
                        else:
                            st.error(f"Processing failed: {detection_result}")
                            logger.error(f"Processing failed: {detection_result}")

if __name__ == "__main__":
    main()
