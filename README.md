# 🧬 Breast Cancer Classification Using Deep Learning

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=00D9FF&center=true&vCenter=true&width=800&lines=Breast+Cancer+Classification;Artificial+Neural+Network+(ANN);Deep+Learning+%7C+Python+%7C+TensorFlow" alt="Animated project title">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-orange" alt="Deep Learning">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Project-PR1-8A2BE2" alt="Project PR1">
</p>

<p align="center">
  <b>An end-to-end Deep Learning project exploring breast cancer classification using Artificial Neural Networks.</b>
</p>

<p align="center">
  <a href="https://github.com/dhorajiyamisri/Deep-Learning-PR1">📂 View Repository</a> •
  <a href="https://github.com/dhorajiyamisri">👩‍💻 GitHub Profile</a>
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset Description](#-dataset-description)
- [Data Dictionary](#-data-dictionary)
- [Technologies Used](#-technologies-used)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Deep Learning Concepts](#-deep-learning-concepts)
- [Model Experiments](#-model-experiments)
- [Model Evaluation](#-model-evaluation)
- [Visualization Gallery](#-visualization-gallery)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [How to Run](#-how-to-run)
- [Key Learnings](#-key-learnings)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🔍 Project Overview

**Breast Cancer Classification Using Deep Learning** is an educational project that explores how Artificial Neural Networks (ANNs) can learn patterns from numerical diagnostic data.

The project uses the Breast Cancer Wisconsin Diagnostic dataset provided by Scikit-learn. It covers data exploration, preprocessing, neural network experiments, model evaluation, visualization, and saving trained model artifacts.

The classification task predicts one of two target classes:

- **Malignant — Class 0**
- **Benign — Class 1**

The project also explores important deep learning concepts, including perceptrons, multilayer perceptrons, activation functions, loss functions, gradient descent, early stopping, dropout, and regularization.

> ⚠️ **Disclaimer:** This project is for educational purposes only. Its predictions must not be used as a substitute for professional medical diagnosis.

## 🎯 Problem Statement

Diagnostic datasets contain numerical measurements that may reveal patterns associated with different diagnostic classes.

The objective of this project is to train a neural network that learns from these measurements and predicts whether a sample belongs to the malignant or benign class.

This is a **supervised binary classification problem**.

## 🎯 Objectives

- Understand the structure of a real-world diagnostic dataset.
- Perform exploratory data analysis (EDA).
- Analyze feature distributions and correlations.
- Prepare features for neural network training.
- Understand perceptrons and multilayer perceptrons.
- Experiment with different neural network configurations.
- Explore training optimization and regularization techniques.
- Evaluate the model using appropriate classification metrics.
- Visualize model performance using plots.
- Save the trained model and preprocessing scaler.

## 📊 Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset  
**Source:** `sklearn.datasets.load_breast_cancer`

| Property | Description |
|---|---|
| Total observations | 569 |
| Input features | 30 |
| Target classes | 2 |
| Feature type | Numerical |
| Learning type | Supervised learning |
| Problem type | Binary classification |

The dataset contains measurements derived from cell nuclei. Features represent characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

### Target Class Mapping

| Target value | Class | Description |
|---|---|---|
| 0 | Malignant | Malignant diagnostic class |
| 1 | Benign | Benign diagnostic class |

This mapping follows the original Scikit-learn dataset.

## 📚 Data Dictionary

The dataset includes 30 numerical features grouped into three measurement categories: mean, standard error, and worst values.

| Feature Group | Description |
|---|---|
| Radius | Nucleus radius measurements |
| Texture | Variation in grayscale intensity |
| Perimeter | Boundary measurements |
| Area | Area-related measurements |
| Smoothness | Variation in local radius lengths |
| Compactness | Compactness-related measurements |
| Concavity | Severity of concave portions |
| Concave points | Measurements of concave boundary points |
| Symmetry | Symmetry-related measurements |
| Fractal dimension | Complexity of the boundary |

These feature groups help describe the numerical characteristics used by the classifier.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | Data handling and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Dataset loading, preprocessing, splitting, and evaluation |
| TensorFlow / Keras | Deep learning model development |
| Joblib | Saving and loading preprocessing objects |
| Jupyter Notebook | Experimentation and documentation |
| Git and GitHub | Version control and project hosting |

## 🔄 Project Workflow

```text
Dataset Loading
      ↓
Data Inspection
      ↓
Exploratory Data Analysis
      ↓
Feature and Target Separation
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Neural Network Experiments
      ↓
Model Training and Validation
      ↓
Performance Evaluation
      ↓
Visualization and Comparison
      ↓
Save Final Model and Scaler
      ↓
Results and Documentation
```

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis helps understand the dataset before model training.

The analysis includes:

- Inspecting the dataset dimensions.
- Reviewing feature names and data types.
- Checking missing values.
- Examining class distribution.
- Exploring relationships between numerical features.
- Visualizing feature correlations.

### 1. Class Distribution

<p align="center">
  <img src="plots/class_distribution.png" alt="Class Distribution" width="360">
</p>

<p align="center"><sub>Figure 1 — Distribution of malignant and benign classes</sub></p>

### 2. Feature Correlation Heatmap

<p align="center">
  <img src="plots/correlation_heatmap.png" alt="Feature Correlation Heatmap" width="360">
</p>

<p align="center"><sub>Figure 2 — Correlation among numerical features</sub></p>

## 🧹 Data Preprocessing

Preprocessing prepares the dataset for neural network training.

The workflow includes:

1. Loading the dataset.
2. Separating input features and target labels.
3. Splitting the dataset into training and testing subsets.
4. Applying `StandardScaler` to numerical features.
5. Using the same fitted scaler to transform test data.

### Why Standardization?

Neural networks can be affected when input features have substantially different scales. Standardization makes numerical features more comparable and can help model optimization.

The scaler should be fitted only on the training data to avoid data leakage.

## 🧠 Deep Learning Concepts

### 1. Perceptron

A perceptron is a basic artificial neuron. It combines input values with weights and a bias, then applies an activation function to produce an output.

### 2. Multilayer Perceptron (MLP)

An MLP consists of an input layer, one or more hidden layers, and an output layer. Hidden layers learn patterns from the input features.

### 3. Activation Functions

Activation functions introduce nonlinearity into neural networks.

Examples include:

- ReLU
- Sigmoid

The activation functions used in each experiment should be checked in the notebook.

### 4. Loss Function

A loss function measures the difference between predicted outputs and actual labels. The training process aims to reduce this loss.

### 5. Gradient Descent

Gradient descent updates model parameters to reduce the loss. Common approaches include batch, stochastic, and mini-batch gradient descent.

### 6. Regularization

Regularization techniques can help reduce overfitting.

Examples include:

- Dropout
- L1 regularization
- L2 regularization
- Early stopping

Their impact depends on the network architecture and training configuration.

## 🧪 Model Experiments

The project notebook explores different neural network concepts and techniques.

| Experiment | Purpose |
|---|---|
| Single-Layer Perceptron (SLP) | Understand basic neural classification |
| Multilayer Perceptron (MLP) | Learn more complex patterns |
| Activation Comparison | Explore activation function behavior |
| Early Stopping | Monitor validation performance during training |
| Dropout | Study regularization |
| L1/L2 Regularization | Compare weight-penalty approaches |
| Final Combined Model | Evaluate the selected model configuration |

The experiment plots are presented in the visualization gallery below. Exact conclusions should be based on the corresponding notebook outputs.

## 📈 Model Evaluation

A classification model should be evaluated using more than accuracy alone.

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Proportion of positive predictions that are correct |
| Recall | Proportion of actual positive cases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Summary of correct and incorrect predictions by class |

The previously recorded test accuracy was approximately **95.61%**. Confirm it against the final model's notebook output before using it as a verified result.

| Metric | Result |
|---|---|
| Test accuracy | Approximately 95.61% — verify against notebook |
| Precision | Refer to the final classification report |
| Recall | Refer to the final classification report |
| F1-score | Refer to the final classification report |

For medical classification, recall and the types of classification errors are particularly important. Accuracy alone is not sufficient to establish clinical reliability.

## 🖼️ Visualization Gallery

All images below use relative paths from the repository root. The specified width keeps each plot compact on the GitHub README page.

### A. Exploratory Data Analysis

<table>
  <tr>
    <td align="center" width="50%">
      <img src="plots/class_distribution.png" alt="Class Distribution" width="300">
      <br><b>Class Distribution</b>
      <br><sub>Target class balance</sub>
    </td>
    <td align="center" width="50%">
      <img src="plots/correlation_heatmap.png" alt="Correlation Heatmap" width="300">
      <br><b>Correlation Heatmap</b>
      <br><sub>Feature relationships</sub>
    </td>
  </tr>
</table>

### B. Training and Activation Experiments

<table>
  <tr>
    <td align="center" width="50%">
      <img src="plots/slp_training_curves.png" alt="SLP Training Curves" width="300">
      <br><b>SLP Training Curves</b>
      <br><sub>Single-layer perceptron training</sub>
    </td>
    <td align="center" width="50%">
      <img src="plots/activation_comparison.png" alt="Activation Comparison" width="300">
      <br><b>Activation Comparison</b>
      <br><sub>Comparison of activation experiments</sub>
    </td>
  </tr>
</table>

### C. Early Stopping Experiments

<table>
  <tr>
    <td align="center" width="50%">
      <img src="plots/early_stopping_curve.png" alt="Early Stopping Curve" width="300">
      <br><b>Early Stopping Curve</b>
      <br><sub>Training and validation behavior</sub>
    </td>
    <td align="center" width="50%">
      <img src="plots/early_stopping_vs_normal.png" alt="Early Stopping vs Normal Training" width="300">
      <br><b>Early Stopping vs Normal</b>
      <br><sub>Comparison of training approaches</sub>
    </td>
  </tr>
</table>

### D. Dropout Experiments

<table>
  <tr>
    <td align="center" width="50%">
      <img src="plots/dropout_comparison.png" alt="Dropout Comparison" width="300">
      <br><b>Dropout Comparison</b>
      <br><sub>Comparison of dropout configurations</sub>
    </td>
    <td align="center" width="50%">
      <img src="plots/dropout_confusion_matrix.png" alt="Dropout Confusion Matrix" width="300">
      <br><b>Dropout Confusion Matrix</b>
      <br><sub>Classification results for the dropout experiment</sub>
    </td>
  </tr>
</table>

### E. Model Confusion Matrices

<table>
  <tr>
    <td align="center" width="50%">
      <img src="plots/slp_confusion_matrix.png" alt="SLP Confusion Matrix" width="300">
      <br><b>SLP Confusion Matrix</b>
      <br><sub>Single-layer model predictions</sub>
    </td>
    <td align="center" width="50%">
      <img src="plots/mlp_confusion_matrix.png" alt="MLP Confusion Matrix" width="300">
      <br><b>MLP Confusion Matrix</b>
      <br><sub>Multilayer model predictions</sub>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="plots/final_model_confusion_matrix.png" alt="Final Model Confusion Matrix" width="320">
      <br><b>Final Model Confusion Matrix</b>
      <br><sub>Actual versus predicted classes for the final model</sub>
    </td>
  </tr>
</table>

### F. Regularization Experiments

<table>
  <tr>
    <td align="center" width="50%">
      <img src="plots/regularization_accuracy.png" alt="Regularization Accuracy" width="300">
      <br><b>Regularization Accuracy</b>
      <br><sub>Accuracy across regularization experiments</sub>
    </td>
    <td align="center" width="50%">
      <img src="plots/regularization_comparison.png" alt="Regularization Comparison" width="300">
      <br><b>Regularization Comparison</b>
      <br><sub>Comparison of regularization approaches</sub>
    </td>
  </tr>
</table>

> **Image path note:** These paths are relative to the repository root. The files must exist in `plots/` and be committed to GitHub with the exact filenames shown above. If any filename differs in the actual folder, update its `src` value to match.

## 📁 Project Structure

```text
Deep-Learning-PR1/
│
├── DL_PR1.ipynb
├── DL_PR1.html
├── README.md
├── requirements.txt
│
├── models/
│   ├── final_model.keras
│   └── scaler.pkl
│
└── plots/
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── slp_training_curves.png
    ├── activation_comparison.png
    ├── early_stopping_curve.png
    ├── early_stopping_vs_normal.png
    ├── dropout_comparison.png
    ├── dropout_confusion_matrix.png
    ├── slp_confusion_matrix.png
    ├── mlp_confusion_matrix.png
    ├── final_model_confusion_matrix.png
    ├── regularization_accuracy.png
    └── regularization_comparison.png
```

### Main Files

| File or Folder | Description |
|---|---|
| `DL_PR1.ipynb` | Main notebook containing experiments and analysis |
| `DL_PR1.html` | Exported notebook, if included |
| `requirements.txt` | Project dependencies |
| `models/final_model.keras` | Saved final model, if present |
| `models/scaler.pkl` | Saved preprocessing scaler, if present |
| `plots/` | EDA and model evaluation visualizations |

## ⚙️ Installation and Setup

### Prerequisites

- Python version compatible with the project's dependencies
- Git
- Visual Studio Code or Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/dhorajiyamisri/Deep-Learning-PR1.git
```

### Step 2: Enter the Project Directory

```bash
cd Deep-Learning-PR1
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Open the Notebook

```bash
jupyter notebook DL_PR1.ipynb
```

Run the notebook cells in order to reproduce the data analysis and experiments.

## ▶️ How to Run

1. Clone or download the repository.
2. Install the dependencies.
3. Open `DL_PR1.ipynb` in Jupyter Notebook or VS Code.
4. Execute the notebook cells in sequence.
5. Review the data visualizations, model experiments, and evaluation outputs.
6. Check the `models/` folder for saved model artifacts, if generated.

## 💡 Key Learnings

Through this project, I explored:

- The difference between machine learning and deep learning.
- Perceptrons and multilayer neural networks.
- Feature scaling and train-test splitting.
- Activation functions and loss functions.
- Gradient descent and neural network optimization.
- Early stopping, dropout, and regularization.
- Classification metrics and confusion matrices.
- Saving trained models and preprocessing objects.
- Documenting machine learning experiments using GitHub.

## 🚀 Limitations and Future Improvements

### Current Limitations

- The dataset is small compared with the variety of real-world clinical data.
- Performance on this dataset does not guarantee performance on unseen populations.
- Results depend on the train-test split and training configuration.
- The project is educational and has not established clinical validity.

### Future Improvements

- Add cross-validation for more robust evaluation.
- Compare additional neural network architectures.
- Include ROC curves and precision-recall curves.
- Document all experiments with verified metrics.
- Add automated tests and reproducible training configurations.
- Validate generalization using independent, appropriately sourced data.

## ✅ Conclusion

This project demonstrates an educational workflow for breast cancer classification using deep learning. It combines dataset exploration, preprocessing, neural network experiments, performance evaluation, and visualization.

Working through these steps helps develop practical understanding of Artificial Neural Networks and the process of evaluating a supervised classification model.

## 👩‍💻 Author

**Misari Dhorajiya**

Aspiring Data Science and AI/ML professional, currently looking for a Data Science / AI Internship.

- **GitHub Profile:** [dhorajiyamisri](https://github.com/dhorajiyamisri)
- **Project Repository:** [Deep-Learning-PR1](https://github.com/dhorajiyamisri/Deep-Learning-PR1)

---

<p align="center">
  <i>Learning Deep Learning through hands-on experimentation, evaluation, and visualization.</i>
</p>
