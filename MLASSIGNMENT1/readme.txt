# SMS Spam Classification Project

A machine learning project that implements SMS spam detection using k-Nearest Neighbors (k-NN) and Naive Bayes classifiers. The project is built from scratch without using scikit-learn or other third-party ML libraries, implementing all algorithms manually.

## 📋 Project Overview

This project classifies SMS messages as either "ham" (legitimate) or "spam" using two different machine learning approaches:

- **k-Nearest Neighbors (k-NN)**: Instance-based learning algorithm
- **Multinomial Naive Bayes**: Probabilistic classifier based on Bayes' theorem

### Key Features

- ✅ **No Third-Party ML Libraries**: All algorithms implemented from scratch
- ✅ **Progress Tracking**: Real-time progress bars and time estimation
- ✅ **Comprehensive Visualizations**: Confusion matrices, metrics comparison, and class distribution
- ✅ **Reproducible Results**: Fixed random seeds ensure consistent outputs
- ✅ **Performance Metrics**: Accuracy, Precision, Recall, F1-Score analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd MLASSIGNMENT1
   
   # Or simply download and extract the project folder
   ```

2. **Install required dependencies**
   ```bash
   pip install tqdm matplotlib numpy
   ```

3. **Verify the dataset is present**
   ```bash
   ls -la SMSSpamCollection
   ```
   The `SMSSpamCollection` file should be present in the project directory.

### Running the Project

Simply execute the main script:

```bash
python Assignment1.py
```

## 📁 Project Structure

```
MLASSIGNMENT1/
├── Assignment1.py              # Main execution file
├── SMSSpamCollection          # SMS dataset
├── README.md                  # This file
├── utils/
│   ├── data_loader.py         # Data loading utilities
│   └── text_processing_clean.py # Text tokenization (sklearn-free)
├── models/
│   ├── knn_classifier.py      # k-NN implementation
│   └── naive_bayes_classifier.py # Naive Bayes implementation
├── evaluation/
│   ├── metrics.py             # Performance metrics calculation
│   └── visualization.py       # Plotting utilities
├── data_preprocessing.py      # Train-test splitting
├── feature_engineering.py     # TF-IDF vectorization
└── Generated Outputs/
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    └── class_distribution.png
```

## �� Expected Output

The program will display:

1. **Data Loading**: Dataset statistics and class distribution
2. **Text Processing**: Tokenization progress with vocabulary size
3. **Data Splitting**: Train-test split analysis
4. **Feature Engineering**: TF-IDF vector creation
5. **Model Training**: Progress bars for both classifiers
6. **Model Evaluation**: Performance metrics for each model
7. **Model Comparison**: Side-by-side performance comparison
8. **Visualizations**: Three generated plots
9. **Timing Summary**: Execution time breakdown

### Sample Output
```
================================================================================
SMS SPAM CLASSIFICATION PROJECT
================================================================================

1. LOADING AND PREPROCESSING DATA
--------------------------------------------------
✓ Loaded 5574 messages
✓ Ham: 4827 (86.6%)
✓ Spam: 747 (13.4%)
⏱️  Data loading completed in 0.0s

2. TOKENIZING TEXT
--------------------------------------------------
Tokenizing: 100%|█████████████████████████████████████| 5574/5574 [00:00<00:00, 209713.32msg/s]
✓ Tokenized 5574 messages
✓ Vocabulary size: 8753
⏱️  Tokenization completed in 0.0s

[... continues with detailed progress ...]

⏱️  TOTAL EXECUTION TIME: 37.5s
================================================================================
```

## 🔧 Dependencies

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `tqdm` | Latest | Progress bars and time estimation |
| `matplotlib` | Latest | Data visualization |
| `numpy` | Latest | Numerical computations |

### Installation Commands

```bash
# Install all dependencies at once
pip install tqdm matplotlib numpy

# Or install individually
pip install tqdm
pip install matplotlib
pip install numpy
```

### Python Standard Library

The project uses only Python standard library modules for core functionality:
- `time` - Timing and performance measurement
- `random` - Random number generation (with fixed seeds)
- `collections` - Data structures (Counter, defaultdict)
- `math` - Mathematical operations
- `re` - Regular expressions for text processing
- `os` - File system operations

## 📈 Performance Results

### Model Performance (Typical Results)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| k-NN | 0.9093 | 0.9615 | 0.3356 | 0.4975 |
| Naive Bayes | 0.9847 | 0.9853 | 0.8993 | 0.9404 |

### Execution Time Breakdown

- **Data Loading**: ~0.0s
- **Tokenization**: ~0.0s (200K+ messages/second)
- **Data Splitting**: ~0.0s
- **Feature Engineering**: ~0.1s
- **k-NN Training & Prediction**: ~27.6s
- **Naive Bayes Training & Prediction**: ~0.0s
- **Visualization**: ~9.7s
- **Total Time**: ~37.5s

## 🎯 Algorithm Details

### k-Nearest Neighbors (k-NN)
- **k value**: 5 neighbors
- **Distance metric**: Euclidean distance
- **Implementation**: Custom from-scratch implementation
- **Training**: Lazy learning (stores all training data)
- **Prediction**: Majority vote among k nearest neighbors

### Multinomial Naive Bayes
- **Smoothing**: Laplace smoothing (α=1.0)
- **Implementation**: Custom from-scratch implementation
- **Features**: TF-IDF weighted word frequencies
- **Training**: Calculates class priors and word likelihoods
- **Prediction**: Maximum a posteriori (MAP) estimation

### TF-IDF Vectorization
- **Term Frequency**: Normalized by document length
- **Inverse Document Frequency**: log(N/df) where N is total documents
- **Implementation**: Custom from-scratch implementation
- **Vocabulary**: Built from training data only

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'tqdm'**
   ```bash
   pip install tqdm
   ```

2. **FileNotFoundError: SMSSpamCollection**
   - Ensure the dataset file is in the project root directory
   - Check file permissions

3. **ImportError: No module named 'sklearn'**
   - The project is designed to work without sklearn
   - If you see this error, ensure you're using `utils.text_processing_clean.py`

4. **Memory Issues with Large Datasets**
   - The current implementation loads all data into memory
   - For very large datasets, consider implementing batch processing

### Performance Optimization

- **k-NN Speed**: The k-NN implementation is not optimized for speed. For production use, consider:
  - Using approximate nearest neighbor algorithms
  - Implementing vectorized distance calculations
  - Using specialized libraries like FAISS

- **Memory Usage**: For large vocabularies, consider:
  - Implementing sparse matrix representations
  - Using feature selection to reduce vocabulary size

## 📚 Technical Implementation

### Data Processing Pipeline

1. **Data Loading**: Reads tab-separated SMS data
2. **Text Cleaning**: Lowercase conversion, punctuation removal
3. **Tokenization**: Space-separated word splitting
4. **Train-Test Split**: Stratified 80/20 split with fixed random seed
5. **Feature Engineering**: TF-IDF vectorization
6. **Model Training**: Both algorithms trained on same features
7. **Evaluation**: Comprehensive metrics calculation
8. **Visualization**: Three different plot types

### Reproducibility

- **Random Seed**: Fixed at 42 for consistent results
- **Deterministic Algorithms**: Both k-NN and Naive Bayes are deterministic
- **Consistent Splits**: Stratified splitting maintains class distribution

## 📄 License

This project is created for educational purposes. The SMS dataset is publicly available and commonly used in machine learning research.

## 🤝 Contributing

This is an academic project, but suggestions for improvements are welcome:

- Algorithm optimizations
- Additional evaluation metrics
- Enhanced visualizations
- Code documentation improvements

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure the dataset file is present and accessible
4. Check Python version compatibility (3.7+)

---
