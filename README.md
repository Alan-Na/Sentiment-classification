# Sentiment & Survey Response Classification System

### End-to-End | Ensemble Learning | Production-Ready Inference

This repository demonstrates a complete MLE workflow designed to transform a standard classification task into a deployable, reproducible product.

Unlike typical data science notebooks, this project focuses on **production readiness**:

* **Hybrid Ensemble Architecture:** Combines a statistical baseline (`Softmax(TF-IDF + LR)`) with a Neural Network (`MLP(Structured + TF-IDF → SVD)`).
* **Lightweight Inference Engine:** A standalone `pred.py` script that performs inference using **Pure NumPy**, removing heavy production dependencies like PyTorch or Scikit-Learn.
* **Drift Prevention:** Shared feature engineering logic (`features.py`) ensures strict alignment between training and inference pipelines.
* **Automated Reporting:** Automatically generates Confusion Matrices and Per-Class F1 charts upon training.

---

## Repository Structure

```text
.
├── src/
│   ├── train.py               # Main pipeline: Train -> Eval -> Export -> Report
│   ├── features.py            # Shared feature engineering (prevents training-serving skew)
│   ├── export_artifacts.py    # Serializes weights/vocab for the numpy inference engine
│   ├── reporting.py           # Metrics calculation and visualization (Matplotlib)
│   └── config.py              # Centralized configuration (hyperparams, column definitions)
├── baselines/
│   └── knn_baseline.py        # KNN baseline for performance benchmarking
├── pred.py                    # Zero-dependency inference script (loads artifacts/)
├── artifacts/                 # Serialized model assets (generated during training)
├── reports/                   # Performance graphs and metrics (generated during training)
└── data/
    └── training_data_clean.csv  # (Not committed) Place your dataset here

```

---

## Quick Start

### 1. Installation

Set up a clean virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Data Setup

Place your cleaned dataset at `data/training_data_clean.csv`.

* **Required Column:** `label` (Target class)
* **Optional Column:** `student_id` (Used for **Group-Aware Splitting** to prevent data leakage from the same user appearing in both train and test sets).

### 3. Train & Export

Run the full pipeline. This command handles training, evaluation, artifact export, and report generation:

```bash
python -m src.train --data data/training_data_clean.csv --seed 42

```

**Outputs:**

* `artifacts/*`: JSON and NPY files containing vocabularies, IDF vectors, and model weights.
* `reports/metrics.json`: Detailed evaluation metrics.
* `reports/confusion_matrix.png` & `per_class_f1.png`: Visualization of model performance.

### 4. Inference (Production Mode)

Use the lightweight `pred.py` script to predict on new data. This script mimics a production environment by loading the exported artifacts and running inference without the training stack.

```bash
python pred.py path/to/unlabeled.csv > preds.txt

```

> **Note:** `pred.py` exposes a simple API: `predict_all(csv_path) -> List[str]`.

---

## Performance & Benchmarking

The system automatically compares the ensemble model against baselines. You can fill in your specific results below:

| Model | Accuracy | Macro-F1 | Architecture Notes |
| --- | --- | --- | --- |
| **KNN Baseline** | *(Run script)* | *(Run script)* | Features: Likert Scale + Keyword Indicators |
| **Softmax Branch** | *(See reports)* | *(See reports)* | TF-IDF (1-2 ngrams) + Logistic Regression |
| **MLP Branch** | *(See reports)* | *(See reports)* | Structured Data (One-Hot) + Text (SVD) → 3-Layer MLP |
| **Ensemble (Final)** | **(See reports)** | **(See reports)** | Weighted Probability Averaging |

To run the KNN baseline for comparison:

```bash
python -m baselines.knn_baseline --data data/training_data_clean.csv

```

---

## 🛠 Engineering Highlights

### 1. Pure NumPy Inference (`pred.py`)

To simulate a low-latency, lightweight deployment environment, the inference engine was rebuilt from scratch using only NumPy.

* **Custom TF-IDF & SVD:** Implemented vectorization logic that mirrors Scikit-Learn but relies solely on the exported `vocab.json` and `idf.npy`.
* **Matrix Operations:** The MLP forward pass and Logistic Regression probability calculations are performed via raw matrix multiplication.
* **Benefit:** Drastically reduces the size of the inference docker image and cold-start times.

### 2. Feature Consistency

A common pitfall in ML systems is **Training-Serving Skew**. This project solves it by isolating feature logic in `src/features.py`:

* **Text Construction:** Consistent concatenation of Likert scales, multi-select headers, and raw text.
* **Tokenization:** Regex-based tokenization used in `train.py` is exactly replicated in `pred.py`.

### 3. Automated Artifact Export

The `src.export_artifacts.py` module handles the complex task of serializing:

* Scikit-Learn pipelines (Imputers, Scalers).
* PyTorch model weights (transposed for NumPy compatibility).
* Ensemble weights.

---

## Configuration

Hyperparameters and column definitions are managed centrally in `src/config.py`.

* **Hardware:** Automatically detects CUDA/CPU.
* **Ensemble Weights:** Adjustable in `src/config.py` (Default: 0.5/0.5).
