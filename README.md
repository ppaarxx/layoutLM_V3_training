# LayoutLMv3 Document Classification

A robust implementation for fine-tuning Microsoft's LayoutLMv3 model on custom document classification tasks. This repository provides production-ready code for training and testing document image classification models with comprehensive error handling, logging, and evaluation metrics.

## Overview

LayoutLMv3 is a multimodal pre-trained model that understands both text and layout information in document images. This implementation makes it easy to fine-tune the model on your own document classification datasets with minimal setup and maximum reliability.

### Key Features

- **Robust Error Handling**: Comprehensive validation and graceful failure handling
- **Production Ready**: Proper logging, checkpointing, and memory management
- **Comprehensive Evaluation**: Detailed metrics including confusion matrices and per-class accuracy
- **Easy Configuration**: All parameters configurable from the top of each script
- **Memory Efficient**: Batch processing and automatic cleanup
- **Backup Safety**: Automatic model backups before saving new versions

## Quick Start

### Prerequisites

```bash
pip install torch transformers datasets pandas scikit-learn pillow tqdm
```

Optional for visualization:
```bash
pip install matplotlib seaborn
```

### Dataset Structure

Organize your data in the following directory structure:

```
Train/
├── class1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image3.jpg
│   └── ...
└── ...

Test/
├── class1/
│   ├── test1.jpg
│   └── ...
├── class2/
│   ├── test2.jpg
│   └── ...
└── ...
```

### Training

1. **Configure your settings** in `train.py`:
   ```python
   TRAIN_DATASET_PATH = "Train"  # Path to training data
   BATCH_SIZE = 4               # Adjust based on your GPU memory
   NUM_EPOCHS = 10              # Number of training epochs
   ```

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Monitor progress**: Check the logs in `training.log` or console output

### Testing

1. **Configure test settings** in `test.py`:
   ```python
   MODEL_PATH = "./saved_model/"     # Path to your trained model
   TEST_DATASET_PATH = "Test"        # Path to test data
   BATCH_SIZE = 8                    # Batch size for testing
   ```

2. **Run evaluation**:
   ```bash
   python test.py
   ```

3. **View results**: Check the `test_results/` directory for detailed reports

## Configuration Options

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TRAIN_DATASET_PATH` | Path to training dataset | `"Train"` |
| `BATCH_SIZE` | Training batch size | `4` |
| `NUM_EPOCHS` | Number of training epochs | `10` |
| `LEARNING_RATE` | Learning rate | `2e-5` |
| `VALIDATION_SPLIT` | Fraction for validation | `0.2` |
| `MIN_SAMPLES_PER_CLASS` | Minimum samples warning threshold | `5` |

### Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_PATH` | Path to saved model | `"./saved_model/"` |
| `TEST_DATASET_PATH` | Path to test dataset | `"Test"` |
| `BATCH_SIZE` | Testing batch size | `8` |
| `RESULTS_DIR` | Output directory for results | `"./test_results/"` |

## Output Files

### Training Outputs
- `saved_model/`: Fine-tuned model and tokenizer
- `results/`: Training checkpoints and logs
- `training.log`: Detailed training logs

### Testing Outputs
- `test_results/[class]_results.csv`: Per-class prediction results
- `test_results/all_results.csv`: Combined results with confidence scores
- `test_results/classification_report.txt`: Detailed classification metrics
- `test_results/confusion_matrix.png`: Confusion matrix visualization
- `test_results/summary_report.json`: JSON summary with key metrics
- `testing.log`: Detailed testing logs

## Features in Detail

### Error Handling
- Validates dataset structure before training
- Handles corrupted or unreadable images gracefully
- Comprehensive try-catch blocks with informative error messages
- Automatic recovery from common issues

### Memory Management
- Batch processing for large datasets
- Automatic memory cleanup after training
- Configurable batch sizes based on available memory
- Efficient data loading with progress tracking

### Evaluation Metrics
- Overall accuracy and per-class accuracy
- Precision, recall, and F1-score for each class
- Confusion matrix with visualization
- Confidence scores for predictions
- Detailed classification report

### Safety Features
- Automatic backup of existing models
- Validation of minimum samples per class
- Checkpoint cleanup to save disk space
- Comprehensive logging for debugging

## Troubleshooting

### Common Issues

**Out of Memory Error**
- Reduce `BATCH_SIZE` in the configuration
- Use CPU instead of GPU by keeping `device = torch.device("cpu")`

**No Valid Images Found**
- Check image file extensions (supported: .png, .jpg, .jpeg)
- Verify directory structure matches the expected format
- Check file permissions

**Model Loading Error**
- Ensure the model path exists and contains valid model files
- Check if the model was saved completely during training

**Class Mismatch Error**
- Verify test classes match training classes
- Check spelling and case sensitivity of folder names

### Getting Help

1. Check the log files (`training.log` or `testing.log`) for detailed error messages
2. Verify your dataset structure matches the expected format
3. Ensure all dependencies are installed with correct versions

## Requirements

- Python
- PyTorch
- Transformers
- Datasets
- Pandas
- Scikit-learn
- PIL/Pillow
- tqdm

Optional:
- matplotlib 3.5+ (for confusion matrix plots)
- seaborn 0.11+ (for better visualizations)

## Contributing

Feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft Research for the LayoutLMv3 model
- Hugging Face for the transformers library
- The open-source community for various supporting libraries

---

**Note**: This implementation is designed for educational and research purposes. For production use, consider additional optimizations and security measures based on your specific requirements.