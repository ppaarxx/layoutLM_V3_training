# ------------------------------------------------------------------------------------------------------
# Parth
# ------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import torch
import logging
import shutil
from tqdm.auto import tqdm
from transformers import (
    LayoutLMv3ImageProcessor, LayoutLMv3Tokenizer, LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification, TrainingArguments, Trainer
)
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import warnings
import gc

# ------------------------------------------------------------------------------------------------------
# Configuration and Setup
# ------------------------------------------------------------------------------------------------------

# Configuration variables
TRAIN_DATASET_PATH = "Train"
OUTPUT_MODEL_DIR = "./saved_model/"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignore warnings
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuda.*")

# ------------------------------------------------------------------------------------------------------
# Model Components Initialization
# ------------------------------------------------------------------------------------------------------

logger.info("Initializing model components...")
try:
    feature_extractor = LayoutLMv3ImageProcessor()
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    device = torch.device("cpu")  # Use CPU instead of CUDA
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.error(f"Failed to initialize model components: {e}")
    raise

# ------------------------------------------------------------------------------------------------------
# Data Validation and Loading
# ------------------------------------------------------------------------------------------------------

def validate_dataset_structure():
    """Validate the dataset directory structure and contents."""
    if not os.path.exists(TRAIN_DATASET_PATH):
        raise FileNotFoundError(f"Training dataset path '{TRAIN_DATASET_PATH}' does not exist")
    
    labels = [label for label in os.listdir(TRAIN_DATASET_PATH) if os.path.isdir(os.path.join(TRAIN_DATASET_PATH, label))]
    
    if len(labels) == 0:
        raise ValueError(f"No label directories found in '{TRAIN_DATASET_PATH}'")
    
    logger.info(f"Found {len(labels)} classes: {labels}")
    
    # Validate minimum samples per class
    for label in labels:
        label_path = os.path.join(TRAIN_DATASET_PATH, label)
        samples = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(samples) < MIN_SAMPLES_PER_CLASS:
            logger.warning(f"Class '{label}' has only {len(samples)} samples (minimum recommended: {MIN_SAMPLES_PER_CLASS})")
        else:
            logger.info(f"Class '{label}': {len(samples)} samples")
    
    return labels

def load_and_validate_images():
    """Load image paths and labels with validation."""
    logger.info("Loading and validating dataset...")
    
    train_images = []
    train_labels = []
    corrupted_images = []
    
    for label in os.listdir(TRAIN_DATASET_PATH):
        label_path = os.path.join(TRAIN_DATASET_PATH, label)
        
        if not os.path.isdir(label_path):
            continue
            
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(image_files, desc=f"Validating {label}"):
            img_path = os.path.join(label_path, img_name)
            
            # Validate image can be loaded
            try:
                with Image.open(img_path) as img:
                    img.convert("RGB")  # Test conversion
                train_images.append(img_path)
                train_labels.append(label)
            except Exception as e:
                logger.warning(f"Corrupted image {img_path}: {e}")
                corrupted_images.append(img_path)
    
    if corrupted_images:
        logger.warning(f"Found {len(corrupted_images)} corrupted images. They will be skipped.")
    
    if len(train_images) == 0:
        raise ValueError("No valid images found in the dataset")
    
    logger.info(f"Successfully loaded {len(train_images)} valid images")
    return train_images, train_labels

# Validate and load data
labels = validate_dataset_structure()
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}
logger.info(f"Label mapping: {label2idx}")

train_images, train_labels = load_and_validate_images()
train_data = pd.DataFrame({'image_path': train_images, 'label': train_labels})

# ------------------------------------------------------------------------------------------------------
# Data Splitting and Preprocessing
# ------------------------------------------------------------------------------------------------------

logger.info("Splitting data into train and validation sets...")

# Create validation set
validation_data = train_data.sample(frac=VALIDATION_SPLIT, random_state=42)
train_data = train_data.drop(validation_data.index)

# Reset indices
train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

# Validate splits are not empty
if len(train_data) == 0 or len(validation_data) == 0:
    raise ValueError("Train or validation split is empty. Check your data and validation split ratio.")

logger.info(f"Training examples: {len(train_data)}, Validation examples: {len(validation_data)}")

def encode_training_example(examples):
    """Encode examples with error handling."""
    images = []
    valid_indices = []
    
    for idx, path in enumerate(examples['image_path']):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to process image {path}: {e}")
    
    if not images:
        raise ValueError("No valid images in batch")
    
    # Filter examples to only include valid images
    valid_labels = [examples["label"][i] for i in valid_indices]
    
    encoded_inputs = processor(images, padding="max_length", truncation=True, return_token_type_ids=True)
    encoded_inputs["labels"] = [label2idx[label] for label in valid_labels]
    
    return encoded_inputs

# ------------------------------------------------------------------------------------------------------
# Dataset Creation and Encoding
# ------------------------------------------------------------------------------------------------------

logger.info("Creating and encoding datasets...")

# Feature definitions
training_features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
})

try:
    # Convert DataFrame to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_data)
    valid_dataset = Dataset.from_pandas(validation_data)
    
    # Encode datasets
    logger.info("Encoding training dataset...")
    encoded_train_dataset = train_dataset.map(
        encode_training_example, 
        remove_columns=train_dataset.column_names, 
        features=training_features, 
        batched=True, 
        batch_size=16,
        desc="Encoding train data"
    )
    
    logger.info("Encoding validation dataset...")
    encoded_valid_dataset = valid_dataset.map(
        encode_training_example, 
        remove_columns=valid_dataset.column_names, 
        features=training_features, 
        batched=True, 
        batch_size=16,
        desc="Encoding valid data"
    )
    
    # Format for PyTorch
    encoded_train_dataset.set_format(type="torch", device=device)
    encoded_valid_dataset.set_format(type="torch", device=device)
    
    logger.info("Dataset encoding completed successfully")
    
except Exception as e:
    logger.error(f"Failed to encode datasets: {e}")
    raise

# ------------------------------------------------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------------------------------------------------

logger.info("Initializing model...")
try:
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=len(label2idx)
    )
    model = model.to(device)
    logger.info(f"Model initialized with {len(label2idx)} classes")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

# ------------------------------------------------------------------------------------------------------
# Training Configuration and Execution
# ------------------------------------------------------------------------------------------------------

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,        
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir=LOGS_DIR,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_total_limit=1,
    lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=None,
    processing_class=processor,
)

logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
try:
    trainer.train()
    logger.info("Training completed successfully")
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# ------------------------------------------------------------------------------------------------------
# Memory Cleanup and Model Saving
# ------------------------------------------------------------------------------------------------------

logger.info("Cleaning up memory...")
# Clear datasets from memory
del encoded_train_dataset, encoded_valid_dataset, train_dataset, valid_dataset
gc.collect()

# Checkpoint safety
logger.info("Saving model...")
try:
    if os.path.exists(OUTPUT_MODEL_DIR):
        # Create backup of existing model
        backup_dir = f"{OUTPUT_MODEL_DIR}_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(OUTPUT_MODEL_DIR, backup_dir)
        logger.info(f"Existing model backed up to {backup_dir}")
    
    # Create output directory
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    # Save the model
    trainer.save_model(OUTPUT_MODEL_DIR)
    logger.info(f"Model saved successfully to {OUTPUT_MODEL_DIR}")
    
    # Clean up old checkpoints in results directory
    if os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            if item.startswith("checkpoint"):
                checkpoint_path = os.path.join(RESULTS_DIR, item)
                if os.path.isdir(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Cleaned up checkpoint: {checkpoint_path}")
    
except Exception as e:
    logger.error(f"Failed to save model: {e}")
    raise

logger.info("Script completed successfully!")

# ------------------------------------------------------------------------------------------------------
# Parth
# ------------------------------------------------------------------------------------------------------