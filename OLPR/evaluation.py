from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments for paths."""
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 model for license plate detection")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights (best.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--test-img-dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--test-label-dir', type=str, required=True, help='Path to test labels directory')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Output directory for results')
    return parser.parse_args()

def main(model_path, data_yaml, test_img_dir, test_label_dir, output_dir):
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created at {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        return

    # Load model
    try:
        logging.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # YOLO Official Evaluation
    try:
        logging.info("Running YOLO official evaluation on test set...")
        val_results = model.val(data=data_yaml, split='test', conf=0.75, iou=0.5)
        map50 = val_results.box.map50
        map50_95 = val_results.box.map
        precision_yolo = np.mean(val_results.box.p) if val_results.box.p.size > 0 else 0.0
        recall_yolo = np.mean(val_results.box.r) if val_results.box.r.size > 0 else 0.0
        logging.info(f"YOLO Evaluation - mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map50_95:.4f}")
    except Exception as e:
        logging.error(f"YOLO evaluation failed: {e}")
        return

    # Manual Image-Level Evaluation
    y_true = []
    y_pred = []
    y_scores = []

    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Skipping {img_name}: Failed to load image")
            continue

        # Ground truth
        has_gt = False
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    has_gt = len(lines) > 0 and any(line.strip() for line in lines)
            except Exception as e:
                logging.warning(f"Failed to read label file {label_path}: {e}")
        y_true.append(1 if has_gt else 0)

        # Predict using YOLO
        try:
            results = model(img, conf=0.75)
            boxes = results[0].boxes
            has_pred = len(boxes) > 0
            y_pred.append(1 if has_pred else 0)

            if has_pred:
                conf = boxes.conf.cpu().numpy().max()
                y_scores.append(conf)
            else:
                y_scores.append(0)
        except Exception as e:
            logging.warning(f"Prediction failed for {img_name}: {e}")
            y_pred.append(0)
            y_scores.append(0)

    # Compute image-level metrics
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        accuracy = accuracy_score(y_true, y_pred)
        precision_image = precision_score(y_true, y_pred, zero_division=0)
        recall_image = recall_score(y_true, y_pred, zero_division=0)
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_scores)
    except Exception as e:
        logging.error(f"Failed to compute metrics: {e}")
        return

    # Print key metrics
    print(f"\nEvaluation Results:")
    print(f"Accuracy (Image-Level): {accuracy:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    print(f"Precision (YOLO): {precision_yolo:.4f}")
    print(f"Recall (YOLO): {recall_yolo:.4f}")
    print(f"Precision (Image-Level): {precision_image:.4f}")
    print(f"Recall (Image-Level): {recall_image:.4f}")

    # Save all metrics to text file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    try:
        with open(metrics_file, 'w') as f:
            f.write(f"Accuracy (Image-Level): {accuracy:.4f}\n")
            f.write(f"mAP@0.5: {map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {map50_95:.4f}\n")
            f.write(f"Precision (YOLO): {precision_yolo:.4f}\n")
            f.write(f"Recall (YOLO): {recall_yolo:.4f}\n")
            f.write(f"Precision (Image-Level): {precision_image:.4f}\n")
            f.write(f"Recall (Image-Level): {recall_image:.4f}\n")
        logging.info(f"Metrics saved to {metrics_file}")
    except Exception as e:
        logging.error(f"Failed to save metrics to {metrics_file}: {e}")

    # Visualizations
    # 1. Bar plot for metrics
    plt.figure(figsize=(10, 6))
    metrics = [map50, map50_95, precision_yolo, recall_yolo, accuracy, precision_image, recall_image]
    labels = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision (YOLO)', 'Recall (YOLO)', 'Accuracy', 'Precision (Image)', 'Recall (Image)']
    colors = ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9933', '#66CC99']

    bars = plt.bar(labels, metrics, color=colors)
    plt.ylim(0, 1.1)
    plt.title('YOLOv8 Test Metrics for License Plate Detection', fontsize=14)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.03, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    bar_plot_path = os.path.join(output_dir, 'metrics_bar.png')
    try:
        plt.savefig(bar_plot_path, dpi=300)
        logging.info(f"Bar plot saved to {bar_plot_path}")
    except Exception as e:
        logging.error(f"Failed to save bar plot: {e}")
    plt.close()

    # 2. Confusion Matrix (Image-Level)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Plate', 'Plate'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix (Image-Level)')
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix_image_level.png')
    try:
        plt.savefig(cm_plot_path)
        logging.info(f"Confusion matrix saved to {cm_plot_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix: {e}")
    plt.close()

    # 3. Precision-Recall Curve (Image-Level)
    plt.figure(figsize=(6, 6))
    plt.plot(recall_pr, precision_pr, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Image-Level)')
    plt.grid(True)
    pr_plot_path = os.path.join(output_dir, 'precision_recall_image_level.png')
    try:
        plt.savefig(pr_plot_path)
        logging.info(f"Precision-recall curve saved to {pr_plot_path}")
    except Exception as e:
        logging.error(f"Failed to save precision-recall curve: {e}")
    plt.close()

    # 4. YOLO Confusion Matrix and PR Curve from training
    training_results_dir = os.path.dirname(os.path.dirname(model_path))
    cm_path = os.path.join(training_results_dir, 'yolov11_license_plate/confusion_matrix.png')
    pr_path = os.path.join(training_results_dir, 'yolov11_license_plate/PR_curve.png')

    plt.figure(figsize=(12, 5))
    # YOLO Confusion Matrix
    cm_img = cv2.imread(cm_path)
    if cm_img is not None:
        cm_img = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 1)
        plt.imshow(cm_img)
        plt.title('YOLO Confusion Matrix')
        plt.axis('off')
    else:
        logging.warning(f"Could not load {cm_path}")

    # YOLO PR Curve
    pr_img = cv2.imread(pr_path)
    if pr_img is not None:
        pr_img = cv2.cvtColor(pr_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(pr_img)
        plt.title('YOLO Precision-Recall Curve')
        plt.axis('off')
    else:
        logging.warning(f"Could not load {pr_path}")

    plt.tight_layout()
    yolo_plots_path = os.path.join(output_dir, 'yolo_plots.png')
    try:
        plt.savefig(yolo_plots_path)
        logging.info(f"YOLO plots saved to {yolo_plots_path}")
    except Exception as e:
        logging.error(f"Failed to save YOLO plots: {e}")
    plt.close()

    logging.info("Evaluation complete! Results saved in 'evaluation_results/'")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.data, args.test_img_dir, args.test_label_dir, args.output_dir)
