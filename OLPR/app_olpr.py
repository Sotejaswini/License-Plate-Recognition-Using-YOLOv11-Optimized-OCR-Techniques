import streamlit as st
import cv2
import numpy as np
import torch
import os
import json
import requests
import base64
from datetime import datetime
from ultralytics import YOLO
import easyocr
from PIL import Image
import pandas as pd
import re
import logging
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Optimized Car License Plate Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_UPLOAD_DIR = "uploads"
DEFAULT_RESULT_DIR = "results"
DEFAULT_WEIGHTS_DIR = "weights"

# Indian state codes (including TG as alias for Telangana)
STATE_CODES = {
    'TS': 'Telangana', 'TG': 'Telangana', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam',
    'BR': 'Bihar', 'CH': 'Chandigarh', 'CG': 'Chhattisgarh', 'DN': 'Dadra and Nagar Haveli',
    'DD': 'Daman and Diu', 'DL': 'Delhi', 'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana', 'HP': 'Himachal Pradesh',
    'JK': 'Jammu and Kashmir', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala', 'LA': 'Ladakh',
    'LD': 'Lakshadweep', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya',
    'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab', 'PY': 'Puducherry',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura',
    'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal'
}

# Create directories
for directory in [DEFAULT_UPLOAD_DIR, DEFAULT_RESULT_DIR, DEFAULT_WEIGHTS_DIR]:
    os.makedirs(directory, exist_ok=True)

os.makedirs(os.path.join(DEFAULT_RESULT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DEFAULT_RESULT_DIR, "videos"), exist_ok=True)
os.makedirs(os.path.join(DEFAULT_RESULT_DIR, "json"), exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class LPRNetModel:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.max_len = 10

    def load_model(self, model_path):
        try:
            class SimpleLPRNet(torch.nn.Module):
                def __init__(self, class_num, max_len):
                    super(SimpleLPRNet, self).__init__()
                    self.class_num = class_num
                    self.max_len = max_len
                    self.backbone = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2, 2),
                        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2, 2),
                        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(256),
                        torch.nn.ReLU(),
                        torch.nn.AdaptiveAvgPool2d((1, None))
                    )
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(256, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(256, class_num)
                    )

                def forward(self, x):
                    x = self.backbone(x)
                    b, c, h, w = x.size()
                    x = x.view(b, c, w).permute(0, 2, 1)
                    x = self.classifier(x)
                    return x

            self.model = SimpleLPRNet(len(self.chars), self.max_len)
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("LPRNet model loaded successfully")
            else:
                logger.warning(f"LPRNet model not found at {model_path}, using default initialization")
                self.model.apply(self._init_weights)

            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Failed to load LPRNet model: {e}")
            return False

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def predict(self, image):
        if self.model is None:
            return "", 0

        try:
            img = cv2.resize(image, (94, 24))
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                outputs = torch.nn.functional.softmax(outputs, dim=2)

                pred_text = ""
                confidences = []
                for i in range(outputs.shape[1]):
                    pred_idx = torch.argmax(outputs[0, i, :]).item()
                    confidence = torch.max(outputs[0, i, :]).item()
                    if pred_idx < len(self.chars):
                        pred_text += self.chars[pred_idx]
                        confidences.append(confidence)

                final_text = pred_text
                avg_confidence = np.mean(confidences) * 100 if confidences else 0
                return final_text, avg_confidence
        except Exception as e:
            logger.error(f"LPRNet prediction error: {e}")
            return "", 0

class AdvancedLPR:
    def __init__(self, yolo_path, lprnet_path, ocr_api_key):
        self.yolo_model = None
        self.lprnet = LPRNetModel()
        self.easyocr_reader = None
        self.ocr_api_key = ocr_api_key
        self.yolo_path = yolo_path
        self.lprnet_path = lprnet_path
        self.init_models()

        self.plate_patterns = {
            'telangana': r'^(TS|TG)\d{2}[A-Z]{2}\d{4}$',  # e.g., TS03EP6728 or TG03EP6728
            'generic': r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
        }

        self.char_corrections = {
            '0': 'O', '1': 'I', '5': 'S', '2': 'Z', '6': 'G', '8': 'B',
            'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8'
        }

    def init_models(self):
        try:
            self.yolo_model = YOLO(self.yolo_path) if os.path.exists(self.yolo_path) else YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            self.yolo_model = YOLO('yolov8n.pt') if torch.cuda.is_available() else None

        try:
            self.lprnet.load_model(self.lprnet_path)
        except Exception as e:
            logger.error(f"Failed to initialize LPRNet: {e}")

        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None

    def preprocess_plate_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        scale = max(100 / height, 1.0)
        new_height = int(height * scale)
        new_width = int(width * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        variants = {'original': gray}
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            variants['clahe'] = clahe.apply(gray)
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            variants['bilateral'] = bilateral
            _, variants['otsu'] = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception as e:
            logger.error(f"Preprocessin g error: {e}")
        return variants

    def ocr_api_recognition(self, image):
        if not self.ocr_api_key:
            return "", 0

        try:
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            url = "https://api.ocr.space/parse/image"
            payload = {
                'apikey': self.ocr_api_key,
                'base64Image': f"data:image/jpeg;base64,{img_base64}",
                'language': 'eng',
                'isOverlayRequired': False,
                'OCREngine': 2,
                'scale': True,
                'isTable': False
            }

            response = requests.post(url, data=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('OCRExitCode') == 1 and result.get('ParsedResults'):
                    text = result['ParsedResults'][0]['ParsedText'].strip().upper()
                    text = re.sub(r'[^A-Z0-9]', '', text)
                    confidence = 70  # Adjusted to match your example
                    return text, confidence
            return "", 0
        except Exception as e:
            logger.error(f"OCR API error: {e}")
            return "", 0

    def lprnet_recognition(self, image):
        try:
            text, confidence = self.lprnet.predict(image)
            return text, confidence
        except Exception as e:
            logger.error(f"LPRNet recognition error: {e}")
            return "", 0

    def easyocr_recognition(self, image_variants):
        results = []
        if self.easyocr_reader is None:
            return results

        for variant_name, image in image_variants.items():
            try:
                ocr_results = self.easyocr_reader.readtext(
                    image,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    width_ths=0.7,
                    height_ths=0.7,
                    paragraph=False,
                    detail=1
                )
                if ocr_results:
                    texts = [detection[1].strip() for detection in ocr_results if detection[2] > 0.4]
                    confidences = [detection[2] * 100 for detection in ocr_results if detection[2] > 0.4]
                    if texts:
                        combined_text = re.sub(r'[^A-Z0-9]', '', ''.join(texts).upper())
                        avg_confidence = np.mean(confidences) if confidences else 0
                        results.append({
                            'engine': 'easyocr',
                            'variant': variant_name,
                            'text': combined_text,
                            'confidence': avg_confidence
                        })
            except Exception as e:
                logger.error(f"EasyOCR error for {variant_name}: {e}")
        return results

    def clean_and_validate_text(self, text):
        if not text or len(text) < 2:
            return ""
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(text) >= 2 and text[:2] in STATE_CODES:
            if re.match(self.plate_patterns['telangana'], text):
                return text
        return text if 6 <= len(text) <= 10 else ""

    def get_state_name(self, plate_text):
        if not plate_text or len(plate_text) < 2:
            return "Unknown State"
        state_code = plate_text[:2].upper()
        return STATE_CODES.get(state_code, "Unknown State")

    def score_license_plate(self, text, confidence, engine):
        if not text:
            return 0
        score = confidence * 0.7
        if re.match(self.plate_patterns['telangana'], text):
            score += 40
        elif re.match(self.plate_patterns['generic'], text):
            score += 20
        score += 25 if sum(c.isalpha() for c in text) >= 2 and sum(c.isdigit() for c in text) >= 4 else 0
        score += 20 if text[:2].upper() in STATE_CODES else 0
        score += {'lprnet': 10, 'easyocr': 8, 'ocr_api': 12}.get(engine, 0)
        return min(100, max(0, score))

    def select_best_result(self, results):
        if not results:
            return "UNREADABLE", 0, []
        scored_results = []
        seen_texts = set()
        for result in results:
            cleaned_text = self.clean_and_validate_text(result['text'])
            if cleaned_text and cleaned_text not in seen_texts:
                seen_texts.add(cleaned_text)
                score = self.score_license_plate(cleaned_text, result['confidence'], result['engine'])
                scored_results.append({
                    'text': cleaned_text,
                    'score': score,
                    'engine': result['engine'],
                    'variant': result.get('variant', 'original'),
                    'confidence': result['confidence'],
                    'raw_text': result['text'],
                    'state': self.get_state_name(cleaned_text)
                })
        if not scored_results:
            return "UNREADABLE", 0, []
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        best_result = scored_results[0]
        validated_results = [r for r in scored_results if r['score'] > 60]
        if len(validated_results) > 1:
            text_counts = {}
            for r in validated_results:
                text_counts[r['text']] = text_counts.get(r['text'], 0) + 1
            most_common_text = max(text_counts.items(), key=lambda x: x[1])[0]
            for r in validated_results:
                if r['text'] == most_common_text:
                    best_result = r
                    break
        return best_result['text'], best_result['score'], scored_results

    def process_plate_with_all_engines(self, plate_image):
        image_variants = self.preprocess_plate_image(plate_image)
        all_results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_engine = {
                executor.submit(self.easyocr_recognition, image_variants): 'easyocr',
                executor.submit(self.ocr_api_recognition, plate_image): 'ocr_api',
                executor.submit(self.lprnet_recognition, plate_image): 'lprnet'
            }

            for future in as_completed(future_to_engine):
                engine = future_to_engine[future]
                try:
                    result = future.result()
                    if engine == 'easyocr':
                        all_results.extend(result)
                    elif engine in ['ocr_api', 'lprnet']:
                        text, confidence = result
                        if text:
                            all_results.append({
                                'engine': engine,
                                'variant': 'original',
                                'text': text,
                                'confidence': confidence
                            })
                except Exception as e:
                    logger.error(f"Error in {engine}: {e}")
        return self.select_best_result(all_results)

    def detect_and_recognize(self, image, conf_threshold=0.3):
        if self.yolo_model is None:
            return None, "YOLO model not loaded"

        try:
            results = self.yolo_model(image, conf=conf_threshold, iou=0.5)
            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.shape[1], x2 + padding)
                    y2 = min(image.shape[0], y2 + padding)

                    plate_image = image[y1:y2, x1:x2]
                    if plate_image.size == 0:
                        continue

                    plate_text, ocr_score, all_results = self.process_plate_with_all_engines(plate_image)

                    color = (0, 255, 0) if ocr_score > 70 else (0, 255, 255) if ocr_score > 40 else (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    state_name = self.get_state_name(plate_text)
                    text = f"{plate_text} ({ocr_score:.1f}%) - {state_name}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_y = y1 - 10 if y1 > 30 else y2 + 25
                    cv2.rectangle(image, (x1, text_y - text_size[1] - 5), (x1 + text_size[0], text_y + 5), color, -1)
                    cv2.putText(image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    detections.append({
                        'detection_id': i + 1,
                        'license_plate': plate_text,
                        'detection_confidence': float(conf),
                        'ocr_score': float(ocr_score),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'all_ocr_results': all_results,
                        'timestamp': datetime.now().isoformat(),
                        'state': state_name
                    })

            return image, detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None, f"Error during detection"

    def process_video(self, video_path, progress_callback=None, conf_threshold=0.3):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Failed to open video file"

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DEFAULT_RESULT_DIR, "videos", f"processed_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []
        frame_number = 0
        frame_skip = max(1, int(fps // 3))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip == 0:
                processed_frame, detections = self.detect_and_recognize(frame.copy(), conf_threshold)
                if processed_frame is not None:
                    frame = processed_frame
                    for detection in detections:
                        detection['frame_number'] = frame_number
                        detection['timestamp'] = frame_number / fps
                    all_detections.extend(detections)

                if progress_callback and frame_count > 0:
                    progress = min((frame_number + 1) / frame_count, 1.0)
                    progress_callback(progress)

            out.write(frame)
            frame_number += 1

        cap.release()
        out.release()
        return output_path, all_detections

def main():
    st.title("🚗 Optimized License Plate Recognition System")
    st.markdown("### Multi-Engine OCR System for Indian License Plates")
    st.markdown("*Developed for Internship Project*")

    if 'lpr_system' not in st.session_state:
        st.session_state.lpr_system = None
    if 'yolo_path' not in st.session_state:
        st.session_state.yolo_path = os.path.join(DEFAULT_WEIGHTS_DIR, "best_yolo.pt")
    if 'lprnet_path' not in st.session_state:
        st.session_state.lprnet_path = os.path.join(DEFAULT_WEIGHTS_DIR, "best_lprnet.pth")
    if 'ocr_api_key' not in st.session_state:
        st.session_state.ocr_api_key = ""
    if 'results_dir' not in st.session_state:
        st.session_state.results_dir = DEFAULT_RESULT_DIR

    st.sidebar.header("⚙️ System Configuration")
    st.session_state.yolo_path = st.sidebar.text_input("Yolo Model Path", st.session_state.yolo_path)
    st.session_state.lprnet_path = st.sidebar.text_input("LPRNet Model Path", st.session_state.lprnet_path)
    st.session_state.ocr_api_key = st.sidebar.text_input("OCR.Space API Key", st.session_state.ocr_api_key, type="password")
    st.session_state.results_dir = st.sidebar.text_input("Results Directory", st.session_state.results_dir)

    if st.sidebar.button("🔄 Initialize Models"):
        with st.spinner("🔄 Initializing AI models..."):
            try:
                st.session_state.lpr_system = AdvancedLPR(
                    st.session_state.yolo_path,
                    st.session_state.lprnet_path,
                    st.session_state.ocr_api_key
                )
                st.success("Models initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize models: {str(e)}")

    lpr = st.session_state.lpr_system

    st.sidebar.subheader("🔧 Model Status")
    status_items = [
        ("YOLO Detection", "✅" if lpr and lpr.yolo_model else "❌"),
        ("LPRNet OCR", "✅" if lpr and lpr.lprnet.model else "❌"),
        ("EasyOCR", "✅" if lpr and lpr.easyocr_reader else "❌")
    ]
    for item, status in status_items:
        st.sidebar.write(f"{status} {item}")

    st.sidebar.subheader("🎛️ Processing Options")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    show_all_results = st.sidebar.checkbox("Show All OCR Results", False)

    st.sidebar.subheader("📁 Upload Media")
    uploaded_file = st.sidebar.file_uploader(
        "Choose image or video file",
        type=['jpg'],
        help="Upload images containing license plates"
    )

    if uploaded_file is not None and lpr is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        temp_path = os.path.join(DEFAULT_UPLOAD_DIR, uploaded_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

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
                    if cv_image is not None:
                        processed_image, detections = lpr.detect_and_recognize(cv_image, confidence_threshold)
                        processing_time = time.time() - start_time

                        if processed_image is not None:
                            with col2:
                                st.subheader("🎯 Detection Results")
                                processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                                st.image(processed_rgb, use_container_width=True)

                            # Save annotated image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_base_name = os.path.splitext(uploaded_file.name)[0]
                            annotated_image_path = os.path.join(st.session_state.results_dir, "images", f"annotated_{img_base_name}_{timestamp}.jpg")
                            os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
                            cv2.imwrite(annotated_image_path, processed_image)
                            logger.info(f"Annotated image saved to {annotated_image_path}")

                            st.subheader("📊 Analysis Dashboard")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Detections", len(detections))
                            with col2:
                                readable = sum(1 for d in detections if d['license_plate'] != 'UNREADABLE')
                                st.metric("Readable Plates", readable)
                            with col3:
                                avg_score = np.mean([d['ocr_score'] for d in detections]) if detections else 0
                                st.metric("Avg. OCR Score", f"{avg_score:.1f}%")
                            with col4:
                                st.metric("Processing Time", f"{processing_time:.2f}s")

                            if detections:
                                st.subheader("📋 Detection Details")
                                data = [{
                                    'Plate': d['license_plate'],
                                    'State': d['state'],
                                    'OCR Score': f"{d['ocr_score']:.1f}%",
                                    'Det. Confidence': f"{d['detection_confidence']:.2f}",
                                    'Timestamp': d['timestamp']
                                } for d in detections]
                                df = pd.DataFrame(data)
                                st.dataframe(df, use_container_width=True)

                                if show_all_results:
                                    st.subheader("🔍 All OCR Results")
                                    for det in detections:
                                        st.write(f"Plate: {det['license_plate']} ({det['state']})")
                                        for res in det['all_ocr_results']:
                                            st.write(f"- {res['engine']} ({res['variant']}): {res['raw_text']} ({res['confidence']:.1f}%)")

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                json_path = os.path.join(st.session_state.results_dir, "json", f"results_{img_base_name}_{timestamp}.json")
                                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                                with open(json_path, 'w') as f:
                                    json.dump(detections, f, cls=NumpyEncoder)
                                st.success(f"Results saved to {json_path}")
                                st.success(f"Annotated image saved to {annotated_image_path}")

        elif file_extension in ['.mp4', '.avi', '.mov']:
            st.subheader("🎥 Video Processing")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing: {progress*100:.1f}%")

            if st.button("🔍 Process Video"):
                with st.spinner("🔄 Processing video..."):
                    output_path, detections = lpr.process_video(temp_path, update_progress, confidence_threshold)
                    if output_path:
                        st.subheader("📽 Processed Video")
                        st.video(output_path)

                        st.subheader("📊 Analysis Dashboard")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Detections", len(detections))
                        with col2:
                            readable = sum(1 for d in detections if d['license_plate'] != 'UNREADABLE')
                            st.metric("Readable Plates", readable)
                        with col3:
                            unique_plates = len(set(d['license_plate'] for d in detections if d['license_plate'] != 'UNREADABLE'))
                            st.metric("Unique Plates", unique_plates)

                        if detections:
                            st.subheader("📋 Detection Details")
                            data = [{
                                'Plate': d['license_plate'],
                                'State': d['state'],
                                'OCR Score': f"{d['ocr_score']:.1f}%",
                                'Frame': d['frame_number'],
                                'Timestamp': f"{d['timestamp']:.2f}s"
                            } for d in detections]
                            df = pd.DataFrame(data)
                            st.dataframe(df, use_container_width=True)

                            if show_all_results:
                                st.subheader("🔍 All OCR Results")
                                for det in detections:
                                    st.write(f"Plate: {det['license_plate']} ({det['state']})")
                                    for res in det['all_ocr_results']:
                                        st.write(f"- {res['engine']} ({res['variant']}): {res['raw_text']} ({res['confidence']:.1f}%)")

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            json_path = os.path.join(st.session_state.results_dir, "json", f"video_results_{timestamp}.json")
                            os.makedirs(os.path.dirname(json_path), exist_ok=True)
                            with open(json_path, 'w') as f:
                                json.dump(detections, f, cls=NumpyEncoder)
                            st.success(f"Results saved to {json_path}")

if __name__ == "__main__":
    main()
