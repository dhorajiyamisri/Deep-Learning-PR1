# 🧬 Breast Cancer Classification Using Deep Learning

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=25&pause=1000&color=00D9FF&center=true&vCenter=true&width=800&lines=Breast+Cancer+Classification;Artificial+Neural+Network+(ANN);Deep+Learning+%7C+Python+%7C+TensorFlow" alt="Animated project title" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-orange" alt="Deep Learning">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
</p>

<p align="center">
  <b>An end-to-end Deep Learning project for classifying breast cancer diagnostic data using an Artificial Neural Network (ANN).</b>
</p>

<p align="center">
  <a href="https://github.com/dhorajiyamisri/Deep-Learning-PR1">GitHub Repository</a> •
  <a href="https://deep-learning-pr1-cozrkcexkgrurrcc3zrjeg.streamlit.app/">Live Demo</a>
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset Description](#-dataset-description)
- [Data Dictionary](#-data-dictionary)
- [Technologies Used](#-technologies-used)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Deep Learning Model](#-deep-learning-model)
- [Model Experiments](#-model-experiments)
- [Model Evaluation](#-model-evaluation)
- [Evaluation Visualizations](#-evaluation-visualizations)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Run the Application](#-run-the-application)
- [Key Learnings](#-key-learnings)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🔍 Project Overview

Breast Cancer Classification Using Deep Learning is an academic project designed to explore how Artificial Neural Networks (ANNs) can learn patterns from numerical diagnostic data.

The project uses the **Breast Cancer Wisconsin Diagnostic dataset** available through Scikit-learn. It follows a machine learning workflow that includes data exploration, preprocessing, neural network experiments, model evaluation, model saving, and application development.

The objective is to classify diagnostic observations into two categories:

- **Malignant:** Class 0
- **Benign:** Class 1

The project also explores fundamental deep learning concepts such as perceptrons, multilayer neural networks, activation functions, loss functions, regularization, and model evaluation.

> **Important:** This project is developed for educational purposes. Its predictions are not a substitute for professional medical diagnosis.

## 🎯 Problem Statement

Breast cancer diagnostic datasets contain multiple numerical measurements that can be difficult to evaluate through manual pattern inspection alone.

This project explores whether a neural network can learn useful patterns from these measurements and classify observations into malignant and benign categories.

The task is a **binary classification problem**.

## 🎯 Project Objectives

- Understand the structure of a diagnostic dataset.
- Perform exploratory data analysis (EDA).
- Identify patterns and relationships between numerical features.
- Prepare features for neural network training.
- Understand perceptrons and multilayer perceptrons.
- Experiment with neural network training and regularization.
- Evaluate classification performance using suitable metrics.
- Save the trained model and preprocessing scaler.
- Explore model predictions through a Streamlit interface.

## 📊 Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:** Scikit-learn built-in dataset.

| Property | Description |
|---|---|
| Total observations | 569 |
| Input features | 30 |
| Target classes | 2 |
| Feature type | Numerical |
| Problem type | Binary classification |
| Dataset loading | `sklearn.datasets.load_breast_cancer` |

The dataset contains numerical measurements derived from cell nuclei in breast mass samples.

Examples of feature groups include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension

### Target Classes

| Label | Class | Meaning |
|---|---|---|
| 0 | Malignant | Malignant class in the original Scikit-learn dataset |
| 1 | Benign | Benign class in the original Scikit-learn dataset |

The class encoding is based on the original Scikit-learn dataset.

## 📚 Data Dictionary

The dataset contains 30 numerical input features. The following table describes representative feature groups.

| Feature Group | Description |
|---|---|
| `radius` | Measurements related to nucleus radius |
| `texture` | Variation in grayscale intensity |
| `perimeter` | Boundary measurement |
| `area` | Area-related measurement |
| `smoothness` | Local variation in radius lengths |
| `compactness` | Compactness-related measurement |
| `concavity` | Severity of concave portions |
| `concave points` | Number or extent of concave portions |
| `symmetry` | Symmetry-related measurement |
| `fractal dimension` | Complexity of the boundary |

The dataset includes mean, standard error, and worst-value measurements for the feature groups. The target variable represents the diagnostic class.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | Data handling and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Dataset loading, preprocessing, splitting, and evaluation |
| TensorFlow / Keras | Neural network development |
| Joblib | Saving and loading preprocessing objects |
| Jupyter Notebook | Experimentation and documentation |
| Streamlit | Interactive web interface |
| Git and GitHub | Version control and project hosting |

## 🔄 Project Workflow

```text
Dataset Loading
      ↓
Data Inspection and Exploration
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
Save Final Model and Scaler
      ↓
Streamlit Application
      ↓
Prediction Output
```

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis helps understand the dataset before model training.

The analysis can include:

- Checking the number of rows and columns.
- Inspecting data types and summary statistics.
- Checking for missing values.
- Studying the distribution of target classes.
- Exploring feature distributions.
- Examining relationships between selected features.

### EDA Visualizations

The following section can display small charts generated from the notebook.

<p align="center">
  <img src="plots/class_distribution.png" alt="Class Distribution" width="360">
</p>

<p align="center"><sub>Figure 1 — Distribution of diagnostic classes</sub></p>

<p align="center">
  <img src="plots/correlation_heatmap.png" alt="Feature Correlation Heatmap" width="360">
</p>

<p align="center"><sub>Figure 2 — Correlation between numerical features</sub></p>

**Note:** Save the corresponding charts in the `plots/` folder using these filenames before publishing the README. If a chart has not been generated, remove its image section until the file is available.

## 🧹 Data Preprocessing

Data preprocessing prepares numerical features for neural network training.

The workflow includes:

1. Loading the dataset.
2. Separating the input features from the target labels.
3. Splitting observations into training and testing sets.
4. Standardizing numerical features using `StandardScaler`.
5. Using the fitted scaler to transform the test data consistently.

### Why Standardization?

Neural networks can be sensitive to differences in feature scales. Standardization transforms numerical features to a comparable scale.

The scaler must be fitted on the training data only. The same fitted scaler is then used to transform test data and future prediction inputs.

## 🧠 Deep Learning Model

The project explores Artificial Neural Networks and their role in classification.

### 1. Perceptron

A perceptron is a basic artificial neuron that combines input values with weights and a bias before applying an activation function.

### 2. Multilayer Perceptron (MLP)

An MLP contains an input layer, one or more hidden layers, and an output layer.

The hidden layers learn relationships between the input features. The output layer produces a value used to classify an observation.

### 3. Activation Functions

Activation functions help neural networks learn nonlinear patterns.

Examples studied in deep learning include:

- ReLU
- Sigmoid
- Other activation functions explored in the notebook

The exact activation functions used in the final model should be confirmed from the model architecture in `DL_PR1.ipynb`.

### 4. Loss Function and Optimization

A loss function measures the difference between the model's predictions and the actual labels.

During training, an optimizer updates the model weights to reduce the loss.

### 5. Regularization

Techniques such as dropout, early stopping, and L1/L2 regularization can help control overfitting. Their effectiveness depends on the model configuration and dataset.

## 🧪 Model Experiments

The practical notebook explores fundamental neural network concepts and techniques for improving model performance.

| Experiment | Purpose |
|---|---|
| Single-Layer Perceptron | Understand basic classification |
| Multilayer Perceptron | Learn more complex patterns |
| Activation Functions | Explore activation choices |
| Early Stopping | Stop training when validation performance no longer improves |
| Dropout | Study regularization by temporarily deactivating units |
| L1/L2 Regularization | Explore weight penalties |
| Final Combined Model | Evaluate the selected model configuration |

The final architecture, hyperparameters, and experiment results should be interpreted from the notebook's actual outputs.

## 📈 Model Evaluation

Accuracy alone does not provide a complete picture of classification performance. Precision, recall, F1-score, and a confusion matrix can provide additional information about classification errors.

| Metric | What It Measures |
|---|---|
| Accuracy | Fraction of predictions that are correct |
| Precision | Fraction of positive predictions that are correct |
| Recall | Fraction of actual positive cases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Counts of correct and incorrect predictions by class |

The project's previously recorded test accuracy was approximately **95.61%**. Verify this figure against the final evaluation output in `DL_PR1.ipynb` before using it in a formal report.

| Metric | Result |
|---|---|
| Test accuracy | Approximately 95.61% — verify notebook |
| Precision | To be confirmed from the classification report |
| Recall | To be confirmed from the classification report |
| F1-score | To be confirmed from the classification report |

## 📊 Evaluation Visualizations

### Confusion Matrix

<p align="center">
  <img src="plots/confusion_matrix.png" alt="Model Confusion Matrix" width="360">
</p>

<p align="center"><sub>Figure 3 — Actual versus predicted diagnostic classes</sub></p>

### Training History

<p align="center">
  <img src="plots/training_history.png" alt="Training and Validation Accuracy and Loss" width="360">
</p>

<p align="center"><sub>Figure 4 — Training and validation performance across epochs</sub></p>

These image paths are placeholders until the charts are generated and committed to the repository. Keep only the charts that actually exist. A chart should represent the real notebook results rather than illustrative or fabricated values.

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
├── plots/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── training_history.png
│
└── app.py
```

**Important:** This is the intended structure. Keep filenames in the README aligned with the actual files in your repository. The `app.py` file and each plot must exist before they are listed as available project files.

### Main Project Files

| File or Folder | Description |
|---|---|
| `DL_PR1.ipynb` | Notebook containing data analysis and deep learning experiments |
| `DL_PR1.html` | Exported notebook, if included |
| `models/` | Saved model and scaler |
| `plots/` | Evaluation and EDA charts |
| `requirements.txt` | Python package dependencies |
| `app.py` | Streamlit application entry point, if included |

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12, or a version supported by all installed dependencies
- Git
- Visual Studio Code or another Python editor

### Step 1: Clone the Repository

```bash
git clone https://github.com/dhorajiyamisri/Deep-Learning-PR1.git
```

### Step 2: Open the Project

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

If TensorFlow installation fails, check Python-version compatibility and the relevant TensorFlow installation instructions.

### Step 5: Open the Notebook

```bash
jupyter notebook DL_PR1.ipynb
```

Run the notebook cells in order to reproduce the experiments, subject to the installed dependencies and available saved files.

## 🌐 Run the Application

If `app.py` is present and configured to load the saved model and scaler, start the Streamlit application with:

```bash
streamlit run app.py
```

The application should open in your browser.

The deployed demo link is:

**[Open Breast Cancer Classification Demo](https://deep-learning-pr1-cozrkcexkgrurrcc3zrjeg.streamlit.app/)**

If the demo is unavailable or access is restricted, use the notebook or run the application locally.

## 💡 Key Learnings

Through this project, I explored:

- The difference between machine learning and deep learning.
- How perceptrons and multilayer neural networks work.
- Feature scaling and train-test splitting.
- Neural network training and loss minimization.
- Regularization and overfitting control.
- Classification metrics and confusion matrices.
- Saving a trained model for reuse.
- Connecting a machine learning model to a Streamlit interface.
- Maintaining and documenting a project using GitHub.

## 🚀 Limitations and Future Improvements

### Current Limitations

- The dataset is relatively small compared with real-world clinical datasets.
- Performance on this dataset does not guarantee performance on new populations or clinical settings.
- Model evaluation depends on the selected split, preprocessing, and training configuration.
- A prediction from this project must not be treated as a medical diagnosis.

### Future Improvements

- Add cross-validation for more robust evaluation.
- Compare multiple neural network architectures.
- Include a detailed classification report and ROC curve.
- Improve the application interface and input validation.
- Add automated tests and a reproducible training pipeline.
- Evaluate generalization on independent, appropriately validated data.

## ✅ Conclusion

This project demonstrates an educational end-to-end workflow for breast cancer classification using deep learning. It connects numerical data exploration and preprocessing with neural network experiments, model evaluation, saved artifacts, and a potential web application.

The project helped strengthen my understanding of Artificial Neural Networks, binary classification, model evaluation, and practical machine learning workflows.

## 👩‍💻 Author

**Misari Dhorajiya**

Aspiring Data Science and AI/ML professional, currently looking for a Data Science / AI Internship.

- **GitHub:** [@dhorajiyamisri](https://github.com/dhorajiyamisri)
- **Project Repository:** [Deep-Learning-PR1](https://github.com/dhorajiyamisri/Deep-Learning-PR1)

---

<p align="center">
  <i>Built as a learning project to explore Deep Learning, neural networks, and practical model development.</i>
</p>
