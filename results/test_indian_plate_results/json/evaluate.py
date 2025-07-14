import json
import os
import glob
from tabulate import tabulate

def load_ground_truth(file_path):
    """Load ground truth data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_predicted_plate(file_path):
    """Load predicted plate from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Assume the first entry contains the license plate
        return data[0].get('license_plate', '')

def calculate_metrics(ground_truth, predicted):
    """Calculate metrics for a single image."""
    total_chars = len(ground_truth)
    correct_chars = sum(1 for g, p in zip(ground_truth, predicted) if g == p)
    correct = ground_truth == predicted
    crr = (correct_chars / total_chars * 100) if total_chars > 0 else 0
    return correct, correct_chars, total_chars, crr

def main():
    # Paths
    ground_truth_path = '../../ground_truth.json'
    json_dir = '.'

    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)

    # Get all input JSON files
    input_files = sorted(glob.glob(os.path.join(json_dir, 'input_*.json')))

    # Initialize results
    results = []
    total_correct = 0
    total_images = 0
    total_crr = 0

    # Process each input file
    for input_file in input_files:
        # Extract image number from filename (e.g., input_1.json -> image1.jpg)
        file_num = os.path.basename(input_file).split('_')[1].split('.')[0]
        image_name = f'input_{file_num}.jpg'

        # Get ground truth and predicted plate
        gt_plate = ground_truth.get(image_name, '')
        if not gt_plate:
            print(f"Warning: No ground truth found for {image_name}")
            continue

        pred_plate = load_predicted_plate(input_file)

        # Calculate metrics
        correct, correct_chars, total_chars, crr = calculate_metrics(gt_plate, pred_plate)

        # Update totals
        total_images += 1
        if correct:
            total_correct += 1
        total_crr += crr

        # Store results
        results.append([
            image_name,
            gt_plate,
            pred_plate,
            'Yes' if correct else 'No',
            correct_chars,
            total_chars,
            f'{crr:.2f}'
        ])

    # Calculate final metrics
    accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    avg_crr = (total_crr / total_images) if total_images > 0 else 0

    # Print results table
    headers = ['Image', 'Ground Truth', 'Predicted Plate', 'Correct', '# Correct Chars', 'Total Chars', 'CRR (%)']
    print("\nEvaluation Results:")
    print(tabulate(results, headers=headers, tablefmt='grid'))
     # Print results table
    headers = ['Image', 'Ground Truth', 'Predicted Plate', 'Correct', '# Correct Chars', 'Total Chars', 'CRR (%)']
    # Filter only correct predictions
    correct_results = [row for row in results if row[3] == False or str(row[3]).lower() == "no"]
    print("\nCorrect Predictions:")
    print(tabulate(correct_results, headers=headers, tablefmt='grid'))
    #print("\nEvaluation Results:")
    #print(tabulate(results, headers=headers, tablefmt='grid'))

    # Print final metrics
    print(f"\nFinal Evaluation Metrics:")
    print(f"Total Images Processed: {total_images}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average CRR: {avg_crr:.2f}%")

if __name__ == '__main__':
    main()
