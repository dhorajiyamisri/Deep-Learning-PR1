# 🧠 Deep Learning Practical — PR 1
## Breast Cancer Classification Using Artificial Neural Networks

<p align="center">
  <strong>RED & WHITE SKILL EDUCATION</strong><br/>
  <em>Deep Learning Practical Assignment · PR 1</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PROJECT-DEEP%20LEARNING-6C63FF?style=for-the-badge" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/PYTHON-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/NEURAL%20NETWORKS-ANN-00A896?style=for-the-badge" alt="ANN"/>
  <img src="https://img.shields.io/badge/STATUS-ACADEMIC%20PROJECT-2E8B57?style=for-the-badge" alt="Academic Project"/>
</p>

<p align="center">
  <a href="DL_PR1.ipynb">📓 Notebook</a> •
  <a href="DL_PR1.html">🌐 HTML Report</a> •
  <a href="plots/">📊 Visualizations</a>
</p>

---

## 📌 01 · Project Overview

This practical project explores how **Artificial Neural Networks (ANNs)** can learn patterns from numerical data and perform binary classification.

Using the Breast Cancer Wisconsin Diagnostic dataset, the project progresses from exploratory data analysis to neural network construction, training, regularization, and final model evaluation.

The focus is not only on building a model, but also on understanding **why different architectures and training techniques produce different results**.

### 🎯 Problem Statement

Develop and evaluate neural network models that classify breast cancer observations into the dataset's two target classes: malignant and benign.

### Objectives

- Understand the dataset and its feature relationships.
- Build a Single Layer Perceptron and a Multi-Layer Perceptron.
- Compare activation functions.
- Investigate Early Stopping and Dropout.
- Evaluate L1, L2, and combined regularization.
- Compare models using classification metrics and confusion matrices.

> **Important:** This is an educational machine learning project, not a clinically validated diagnostic system.

---

## 🏫 02 · Academic Information

| Detail | Information |
|---|---|
| Institute | Red & White Skill Education |
| Practical | Deep Learning — PR 1 |
| Project Title | Breast Cancer Classification Using Deep Learning |
| Student | Misari Dhorajiya |
| Domain | Artificial Intelligence & Deep Learning |
| Programming Language | Python |

---

## 🗂️ 03 · Dataset & Preprocessing

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through `scikit-learn`.

| Dataset property | Value |
|---|---:|
| Total observations | 569 |
| Input features | 30 |
| Malignant observations (class 0) | 212 |
| Benign observations (class 1) | 357 |
| Training split | 80% |
| Testing split | 20% |

### Data preparation workflow

1. Load the dataset and inspect its structure.
2. Explore the target distribution and feature correlations.
3. Split data using stratified train-test splitting.
4. Fit `StandardScaler` on training features only.
5. Transform the training and testing features using the fitted scaler.

Using the training data to fit the scaler avoids leaking information from the test set into preprocessing.

### 📊 Target Class Distribution

![Target class distribution](plots/class_distribution.png)

**Why this plot matters:** It shows the number of observations in each target class. Understanding class distribution helps interpret accuracy and identify whether additional metrics are needed.

### 🔥 Feature Correlation Heatmap

![Feature correlation heatmap](plots/correlation_heatmap.png)

**Why this plot matters:** The heatmap helps reveal relationships among numerical features. Highly correlated features may contain overlapping information, which is useful to understand during exploratory analysis.

---

## 🧪 04 · Practical Tasks: Seven Experiments

### Task 1 — Exploratory Data Analysis

Dataset structure, target distribution, feature correlations, stratified splitting, and feature scaling.

**Learning outcome:** Understand the data before training a neural network.

### Task 2 — Single Layer Perceptron (SLP)

A baseline neural network with one sigmoid output neuron for binary classification.

![SLP training curves](plots/slp_training_curves.png)

**Interpretation:** Training and validation curves help track learning over epochs and identify whether the model's loss decreases and its accuracy improves.

![SLP confusion matrix](plots/slp_confusion_matrix.png)

**Interpretation:** The confusion matrix shows correct predictions and misclassifications for the SLP model.

### Task 3 — Multi-Layer Perceptron (MLP)

A deeper network with hidden layers and nonlinear activation functions.

The practical compares ReLU, Tanh, and Sigmoid activation variants.

![Activation comparison](plots/activation_comparison.png)

**Why this plot matters:** It compares the observed performance of activation-function experiments and helps identify which variant performed better under the chosen training setup.

![MLP confusion matrix](plots/mlp_confusion_matrix.png)

**Interpretation:** This matrix provides a class-by-class view of the MLP's predictions on the evaluated data.

### Task 4 — Early Stopping

Early Stopping monitors validation loss and stops training when improvement stalls for the configured patience period.

![Early stopping curve](plots/early_stopping_curve.png)

**Why this plot matters:** It helps explain when validation loss stops improving and why training for more epochs is not always beneficial.

![Early stopping versus normal training](plots/early_stopping_vs_normal.png)

**Interpretation:** This comparison helps assess the effect of Early Stopping relative to baseline training.

### Task 5 — Dropout Regularization

Dropout randomly disables a proportion of hidden-layer activations during training, which can help reduce overfitting.

![Dropout comparison](plots/dropout_comparison.png)

**Why this plot matters:** It compares the tested dropout rates and their validation performance. The best rate should be selected using actual validation results, not assumed in advance.

![Dropout confusion matrix](plots/dropout_confusion_matrix.png)

**Interpretation:** This matrix shows the classification errors made by the evaluated dropout model.

### Task 6 — L1 and L2 Regularization

The practical investigates regularization strategies that penalize model weights to control complexity.

- **L1:** Penalizes the absolute values of weights.
- **L2:** Penalizes the squared values of weights.
- **L1 + L2:** Combines both penalties.

![Regularization accuracy](plots/regularization_accuracy.png)

**Why this plot matters:** It compares test accuracy across the regularization experiments. Accuracy should be interpreted alongside other classification metrics.

![Regularization comparison](plots/regularization_comparison.png)

**Interpretation:** This visualization helps compare the observed regularization experiments. Use the plot labels and notebook outputs to determine precisely which metric or loss is being compared.

### Task 7 — Final Combined Model

The final experiment combines the selected regularization and training strategies according to the practical requirements.

![Final model confusion matrix](plots/final_model_confusion_matrix.png)

**Interpretation:** The confusion matrix summarizes the final model's predictions. The notebook's classification report should be used alongside this plot to discuss precision, recall, and F1-score.

---

## 📈 05 · Model Evaluation

The project uses multiple metrics because a single score cannot fully describe classification performance.

| Metric | What it measures |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision | Fraction of predicted positive cases that are correct |
| Recall | Fraction of actual positive cases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion matrix | Correct and incorrect predictions by class |

For a medical classification dataset, false negatives and false positives have different implications. Interpretation should consider the positive-class definition and the type of error being measured.

**Results policy:** Model scores should be copied from the notebook's actual evaluation output. No accuracy or metric is assumed in this README.

---

## 🧰 06 · Technologies Used

- **Python** — programming language
- **NumPy** — numerical operations
- **Pandas** — data handling and analysis
- **Matplotlib** — plotting
- **Seaborn** — statistical visualization
- **Scikit-learn** — dataset, preprocessing, splitting, and evaluation utilities
- **TensorFlow / Keras** — neural network modeling, if used in the notebook
- **Jupyter Notebook / VS Code** — development environment

---

## 🚀 07 · Run This Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/dhorajiyamisri/Deep-Learning-PR1.git
cd Deep-Learning-PR1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Open the notebook

Open `DL_PR1.ipynb` in VS Code or Jupyter Notebook and execute the cells in order.

The HTML report can be opened in a web browser for convenient reading.

---

## 📁 08 · Repository Structure

```text
Deep-Learning-PR1/
├── DL_PR1.ipynb
├── DL_PR1.html
├── requirements.txt
├── README.md
└── plots/
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── activation_comparison.png
    ├── dropout_comparison.png
    ├── dropout_confusion_matrix.png
    ├── early_stopping_curve.png
    ├── early_stopping_vs_normal.png
    ├── final_model_confusion_matrix.png
    ├── mlp_confusion_matrix.png
    ├── regularization_accuracy.png
    ├── regularization_comparison.png
    ├── slp_confusion_matrix.png
    └── slp_training_curves.png
```

---

## 🧠 09 · Key Learnings

Through these experiments, the project demonstrates how to:

- Prepare data for neural network training.
- Understand the role of hidden layers and activation functions.
- Monitor training and validation behavior.
- Apply Early Stopping and Dropout.
- Compare L1, L2, and combined regularization.
- Evaluate classification results using multiple metrics.
- Interpret confusion matrices instead of relying on accuracy alone.

---

## 👩‍💻 Author

**Misari Dhorajiya**

GitHub: [@dhorajiyamisri](https://github.com/dhorajiyamisri)

Project repository: [Deep-Learning-PR1](https://github.com/dhorajiyamisri/Deep-Learning-PR1)

---

<p align="center">
  <strong>RED & WHITE SKILL EDUCATION</strong><br/>
  <em>Deep Learning Practical — PR 1</em><br/><br/>
  Built as an academic learning project.
</p>
