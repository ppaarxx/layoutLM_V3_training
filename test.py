# ------------------------------------------------------------------------------------------------------
# Parth
# ------------------------------------------------------------------------------------------------------

import os
import torch
import pandas as pd
import logging
import json
from tqdm.auto import tqdm
from transformers import (
    LayoutLMv3ImageProcessor, LayoutLMv3Tokenizer, LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification
)
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# ------------------------------------------------------------------------------------------------------
# Configuration and Setup
# ------------------------------------------------------------------------------------------------------

# Configuration variables
MODEL_PATH = "./saved_model/"
TEST_DATASET_PATH = "Test"
RESULTS_DIR = "./test_results/"
BATCH_SIZE = 8  # Process images in batches for better memory management

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------------------------------
# Model and Components Initialization
# ------------------------------------------------------------------------------------------------------

def initialize_model_components():
    """Initialize model components with error handling."""
    logger.info("Initializing model components...")
    
    try:
        # Set device
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        # Load processor components
        feature_extractor = LayoutLMv3ImageProcessor()
        tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
        processor = LayoutLMv3Processor(feature_extractor, tokenizer)
        
        # Validate model path exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model path '{MODEL_PATH}' does not exist")
        
        # Load model
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = LayoutLMv3ForSequenceClassification.from_pretrained(MODEL_PATH)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model components initialized successfully")
        return device, processor, model
        
    except Exception as e:
        logger.error(f"Failed to initialize model components: {e}")
        raise

# ------------------------------------------------------------------------------------------------------
# Label Configuration and Validation
# ------------------------------------------------------------------------------------------------------

def load_and_validate_labels():
    """Load labels and validate against test dataset structure."""
    logger.info("Loading and validating labels...")
    
    # Load labels from saved model config if available
    config_path = os.path.join(MODEL_PATH, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'id2label' in config:
                idx2label = {int(k): v for k, v in config['id2label'].items()}
                label2idx = {v: k for k, v in idx2label.items()}
                logger.info("Labels loaded from model config")
            else:
                raise KeyError("No id2label found in config")
        except Exception as e:
            logger.warning(f"Could not load labels from config: {e}. Using default labels.")
            # Fallback to default labels
            label2idx = {'pan': 0, 'all_graphs': 1, 'normal_photo': 2, 'aadhar': 3, 'urine_report': 4, 
                        'path_investigation_reports': 5, 'mer': 6}
            idx2label = {v: k for k, v in label2idx.items()}
    else:
        logger.warning("No config file found. Using default labels.")
        label2idx = {'pan': 0, 'all_graphs': 1, 'normal_photo': 2, 'aadhar': 3, 'urine_report': 4, 
                    'path_investigation_reports': 5, 'mer': 6}
        idx2label = {v: k for k, v in label2idx.items()}
    
    logger.info(f"Label mapping: {label2idx}")
    return label2idx, idx2label

# ------------------------------------------------------------------------------------------------------
# Test Dataset Validation
# ------------------------------------------------------------------------------------------------------

def validate_test_dataset(label2idx):
    """Validate test dataset structure and return valid test classes."""
    logger.info("Validating test dataset...")
    
    if not os.path.exists(TEST_DATASET_PATH):
        raise FileNotFoundError(f"Test dataset path '{TEST_DATASET_PATH}' does not exist")
    
    test_classes = [d for d in os.listdir(TEST_DATASET_PATH) 
                   if os.path.isdir(os.path.join(TEST_DATASET_PATH, d))]
    
    if not test_classes:
        raise ValueError(f"No test class directories found in '{TEST_DATASET_PATH}'")
    
    # Check if test classes match training labels
    unknown_classes = [cls for cls in test_classes if cls not in label2idx]
    if unknown_classes:
        logger.warning(f"Unknown classes found in test set (will be skipped): {unknown_classes}")
    
    valid_classes = [cls for cls in test_classes if cls in label2idx]
    logger.info(f"Valid test classes: {valid_classes}")
    
    # Count samples per class
    for class_name in valid_classes:
        class_folder = os.path.join(TEST_DATASET_PATH, class_name)
        images = [f for f in os.listdir(class_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Class '{class_name}': {len(images)} images")
    
    return valid_classes

# ------------------------------------------------------------------------------------------------------
# Prediction Functions
# ------------------------------------------------------------------------------------------------------

def predict_single_image(image_path, processor, model, device):
    """Predict a single image with error handling."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
        
        return predicted_label, confidence
        
    except Exception as e:
        logger.error(f"Failed to predict image {image_path}: {e}")
        return None, None

def predict_batch_images(image_paths, processor, model, device):
    """Predict a batch of images for better efficiency."""
    try:
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
        
        if not images:
            return [], []
        
        inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).tolist()
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
        
        return predictions, confidences
        
    except Exception as e:
        logger.error(f"Failed to predict batch: {e}")
        return [], []

# ------------------------------------------------------------------------------------------------------
# Testing and Results Generation
# ------------------------------------------------------------------------------------------------------

def test_class(class_name, label2idx, idx2label, processor, model, device):
    """Test all images in a class and return results."""
    logger.info(f"Testing class: {class_name}")
    
    class_folder = os.path.join(TEST_DATASET_PATH, class_name)
    image_files = [f for f in os.listdir(class_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        logger.warning(f"No valid images found in class '{class_name}'")
        return []
    
    results = []
    failed_predictions = 0
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc=f"Processing {class_name}"):
        batch_files = image_files[i:i+BATCH_SIZE]
        batch_paths = [os.path.join(class_folder, f) for f in batch_files]
        
        predictions, confidences = predict_batch_images(batch_paths, processor, model, device)
        
        for j, img_name in enumerate(batch_files):
            true_label = label2idx[class_name]
            true_label_name = idx2label[true_label]
            
            if j < len(predictions) and predictions[j] is not None:
                predicted_label = predictions[j]
                predicted_label_name = idx2label[predicted_label]
                confidence = confidences[j] if j < len(confidences) else 0.0
                
                results.append({
                    "filename": img_name,
                    "true_label": true_label_name,
                    "prediction": predicted_label_name,
                    "confidence": round(confidence, 4),
                    "correct": true_label_name == predicted_label_name
                })
            else:
                failed_predictions += 1
                logger.warning(f"Failed to predict {img_name}")
    
    if failed_predictions > 0:
        logger.warning(f"Failed to predict {failed_predictions} images in class '{class_name}'")
    
    return results

# ------------------------------------------------------------------------------------------------------
# Evaluation Metrics and Reporting
# ------------------------------------------------------------------------------------------------------

def generate_overall_report(all_results):
    """Generate overall evaluation metrics and save detailed report."""
    logger.info("Generating overall evaluation report...")
    
    if not all_results:
        logger.error("No results to generate report from")
        return
    
    # Combine all results
    df_all = pd.DataFrame(all_results)
    
    # Calculate overall metrics
    y_true = df_all['true_label'].tolist()
    y_pred = df_all['prediction'].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Save detailed classification report
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), 'w') as f:
        f.write("Overall Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_true, y_pred))
    
    # Save confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Confusion matrix saved")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")
    
    # Per-class accuracy
    class_accuracies = {}
    for class_name in set(y_true):
        class_mask = df_all['true_label'] == class_name
        class_correct = df_all[class_mask]['correct'].sum()
        class_total = class_mask.sum()
        class_accuracies[class_name] = class_correct / class_total if class_total > 0 else 0
    
    # Save summary report
    summary = {
        "overall_accuracy": accuracy,
        "total_samples": len(df_all),
        "class_accuracies": class_accuracies,
        "classification_report": report
    }
    
    with open(os.path.join(RESULTS_DIR, "summary_report.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save all results
    df_all.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)
    
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info("Detailed reports saved to results directory")

# ------------------------------------------------------------------------------------------------------
# Main Testing Pipeline
# ------------------------------------------------------------------------------------------------------

def main():
    """Main testing pipeline."""
    try:
        # Initialize components
        device, processor, model = initialize_model_components()
        label2idx, idx2label = load_and_validate_labels()
        valid_classes = validate_test_dataset(label2idx)
        
        if not valid_classes:
            logger.error("No valid test classes found")
            return
        
        # Test each class
        all_results = []
        class_results = {}
        
        for class_name in valid_classes:
            try:
                results = test_class(class_name, label2idx, idx2label, processor, model, device)
                if results:
                    class_results[class_name] = results
                    all_results.extend(results)
                    
                    # Save individual class results
                    df_class = pd.DataFrame(results)
                    class_accuracy = df_class['correct'].mean()
                    
                    output_csv = os.path.join(RESULTS_DIR, f"{class_name}_results.csv")
                    df_class.to_csv(output_csv, index=False)
                    
                    logger.info(f"Class '{class_name}' - Accuracy: {class_accuracy:.4f} "
                              f"({df_class['correct'].sum()}/{len(df_class)})")
                    logger.info(f"Results saved to {output_csv}")
                else:
                    logger.warning(f"No results generated for class '{class_name}'")
                    
            except Exception as e:
                logger.error(f"Failed to test class '{class_name}': {e}")
        
        # Generate overall evaluation report
        if all_results:
            generate_overall_report(all_results)
            logger.info("Testing completed successfully!")
        else:
            logger.error("No results generated from any class")
            
    except Exception as e:
        logger.error(f"Testing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------------
# Parth
# ------------------------------------------------------------------------------------------------------