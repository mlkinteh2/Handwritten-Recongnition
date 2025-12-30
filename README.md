
# Handwritten Writer Identification Using Deep Learning (CNN)
<img width="1984" height="5171" alt="image" src="https://github.com/user-attachments/assets/54c672b8-add7-4dba-ad02-0bfdd9e29ed1" />


## Project Overview

This project implements a **writer identification system** using **deep learning only**, specifically a **Convolutional Neural Network (CNN)**.
The goal is to correctly identify the writer of handwritten characters by learning distinctive handwriting patterns that generalize well to **unseen writing styles**.

The system is designed to fully comply with the course requirements:

* CPU-only execution
* No internet access required
* Deep learning models only
* Runnable on a Windows system

---

## Objectives

* Extract informative handwriting regions using patch-based processing
* Train a CNN model to learn writer-specific handwriting features
* Evaluate the model using accuracy, confusion matrix, classification report, and AUC
* Separate training (`train.py`) and testing (`run.py`) workflows clearly

---

## Dataset

* **Training set**: 70 handwritten document images
* **Test set**: 140 handwritten document images
* Writer IDs are encoded in the filenames
* Each image contains multiple handwritten regions

---

## Pre-processing Steps

The following pre-processing steps are applied to prepare the data for deep learning:

### 1. Grayscale Conversion

* Removes unnecessary color information
* Reduces computational complexity
* Focuses learning on handwriting structure and texture

### 2. Normalization

* Pixel values are scaled to the range **[0, 1]**
* Improves numerical stability during training

### 3. Patch Extraction

* Images are divided into **112 × 112** pixel patches
* A sliding window approach is used
* Increases training samples from limited data

### 4. Ink-Based Filtering

* Patches with less than **2% ink coverage** are discarded
* Prevents training on blank or non-informative regions

---

## Model Architecture

A **Convolutional Neural Network (CNN)** is used to automatically learn hierarchical handwriting features.

### Network Structure

* **Input**: 112 × 112 × 1 (grayscale patch)
* **Convolution Block 1**: 32 filters + ReLU + MaxPooling
* **Convolution Block 2**: 64 filters + ReLU + MaxPooling
* **Convolution Block 3**: 128 filters + ReLU + MaxPooling
* **Global Average Pooling**
* **Dense Layer**: 128 units + ReLU
* **Dropout**: 0.3 (regularization)
* **Output Layer**: Softmax activation (70 classes)

**Total Parameters**: ~1.37 million

---

## Training Configuration

* **Optimizer**: Adam
* **Loss Function**: Sparse Categorical Cross-Entropy
* **Batch Size**: 64
* **Epochs**: 30
* **Validation Split**: 20%

These choices ensure stable convergence, efficient memory usage, and suitability for CPU-only environments.

---

## Evaluation Results

### Validation Performance

* **Validation Accuracy**: **70.57%**
* **Multiclass AUC (OvR)**: **0.9912**

### Test Performance

* **Average Test Accuracy**: **81.43%**
* **Weighted F1-Score**: **0.77**

Additional evaluation outputs include:

* Confusion Matrix
* Classification Report
* Per-class performance metrics

---

## Project Structure

```
Handwriting-Project/
│
├── train.py          # Training and validation script
├── run.py            # Testing and inference script
├── model.keras       # Trained CNN model
├── classes.npy       # Label mapping file
├── train/            # Training images
├── test/             # Test images
├── result.csv        # Prediction results
└── README.md         # Project documentation
```

---

## How to Run

### Train the Model

```bash
python train.py
```

This script:

* Preprocesses training data
* Trains the CNN model
* Evaluates on validation data
* Saves the trained model

### Run Testing / Inference

```bash
python run.py
```

This script:

* Loads the trained model
* Predicts writer IDs for test images
* Saves predictions to `result.csv`
* Displays average accuracy

---

## System Requirements

* **Operating System**: Windows 11
* **CPU**: Intel Core i5 or equivalent
* **RAM**: 16 GB
* **GPU**: Not required
* **Internet**: Not required

---

## Compliance with Project Rules

* ✔ Deep learning models only (CNN)
* ✔ No classical machine learning models
* ✔ Python scripts runnable locally
* ✔ CPU-only execution
* ✔ Generative AI used only for coding assistance

---

## Conclusion

This project demonstrates that **patch-based extraction combined with CNN feature learning** provides a robust and scalable solution for writer identification.
The model achieves strong discriminative performance, high AUC, and reliable test accuracy while remaining computationally efficient and compliant with all project constraints.

---

## Author

**Modou Lamin Kinteh**
Bachelor of Data Science
Deep Learning Course Project


