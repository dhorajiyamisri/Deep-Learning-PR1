# 🧬 Breast Cancer Classification Using Deep Learning

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=25&pause=1000&color=00D9FF&center=true&vCenter=true&width=800&lines=Breast+Cancer+Classification;Artificial+Neural+Network+(ANN);Deep+Learning+%7C+Python+%7C+TensorFlow" alt="Animated project title" />
</p>

<p align="center">
  An end-to-end Deep Learning project that explores Artificial Neural Networks for classifying breast cancer diagnostic data.
</p>

<p align="center">
  <a href="https://deep-learning-pr1-cozrkcexkgrurrcc3zrjeg.streamlit.app/">
    <img src="https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
  </a>
  <a href="https://github.com/dhorajiyamisri/Deep-Learning-PR1">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Academic_Project-blue?style=flat-square" alt="Project Status"/>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Deep Learning Workflow](#-deep-learning-workflow)
- [Model Experiments](#-model-experiments)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Run the Streamlit Application](#-run-the-streamlit-application)
- [Saved Model](#-saved-model)
- [Future Improvements](#-future-improvements)
- [Limitations and Disclaimer](#-limitations-and-disclaimer)
- [Author](#-author)

---

## 🔍 Overview

Breast Cancer Classification Using Deep Learning is an academic machine learning project developed to understand how Artificial Neural Networks (ANNs) can learn patterns from numerical diagnostic features.

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through Scikit-learn. It explores data analysis, preprocessing, neural network training, regularization, model evaluation, and saving a trained model for use in a Streamlit application.

The goal is to build a complete learning workflow that connects **data preprocessing, deep learning experiments, model evaluation, and web application development**.

## 🎯 Project Objectives

- Understand and explore a real-world diagnostic dataset.
- Perform exploratory data analysis (EDA).
- Prepare numerical features for neural network training.
- Understand single-layer and multilayer neural network concepts.
- Compare activation functions and training approaches.
- Explore early stopping, dropout, and regularization.
- Evaluate classification performance using appropriate metrics.
- Save the trained model and feature scaler.
- Develop a web interface using Streamlit.

## 📊 Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

**Source:** Scikit-learn built-in dataset.

| Property | Description |
|---|---|
| Total observations | 569 |
| Input features | 30 |
| Target classes | 2 |
| Feature type | Numerical |
| Learning task | Binary classification |

### Target classes

- **Malignant (0):** The malignant class in the original Scikit-learn dataset.
- **Benign (1):** The benign class in the original Scikit-learn dataset.

The dataset contains numerical measurements derived from cell nuclei, including properties related to radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

### Data preprocessing

The project workflow includes:

- Inspecting the dataset and its structure.
- Exploring feature distributions and relationships.
- Separating input features and target labels.
- Splitting the data into training and testing sets.
- Applying `StandardScaler` to standardize input features.

The scaler fitted during training is saved and reused for prediction inputs.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data handling and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Dataset, preprocessing, splitting, and evaluation |
| TensorFlow / Keras | Neural network development and training |
| Joblib | Saving and loading the scaler |
| Streamlit | Interactive web application |
| Jupyter Notebook | Experiments and documentation |
| Git and GitHub | Version control and project hosting |

## 🧠 Deep Learning Workflow

```text
Breast Cancer Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Feature and Target Separation
        |
        v
Train-Test Split
        |
        v
StandardScaler Preprocessing
        |
        v
Neural Network Experiments
        |
        v
Model Evaluation
        |
        v
Save Trained Model and Scaler
        |
        v
Streamlit Application
        |
        v
Prediction Output
```

## 🧪 Model Experiments

The practical notebook explores fundamental neural network concepts and model improvement techniques.

| Experiment | Purpose |
|---|---|
| Single-Layer Perceptron | Understand basic classification |
| Multilayer Perceptron (MLP) | Learn nonlinear patterns |
| Activation Function Comparison | Study neural network activation choices |
| Early Stopping | Reduce unnecessary training and help control overfitting |
| Dropout | Explore regularization through randomly deactivated units |
| L1 and L2 Regularization | Study penalties on model weights |
| Final Combined Model | Evaluate the selected neural network configuration |

The exact configuration and results for each experiment should be taken from the corresponding notebook cells.

## 📈 Model Performance

The final model evaluation previously recorded in the project was approximately **95.61% accuracy**.

Accuracy alone does not fully describe a classifier's performance. The notebook should also be used to review the confusion matrix, precision, recall, and F1-score.

| Metric | Result |
|---|---|
| Test accuracy | Approximately 95.61% |
| Precision | Refer to notebook classification report |
| Recall | Refer to notebook classification report |
| F1-score | Refer to notebook classification report |

*Recheck the final evaluation output in `DL_PR1.ipynb` before using these figures in a formal submission or presentation.*

## 📁 Project Structure

```text
Deep-Learning-PR1/
│
├── DL_PR1.ipynb
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── final_model.keras
│   └── scaler.pkl
│
├── plots/
│   └── Add saved evaluation charts here
│
└── .gitignore
```

*The structure above describes the intended project organization. Keep only the files and folders that actually exist in your repository.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/dhorajiyamisri/Deep-Learning-PR1.git
cd Deep-Learning-PR1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure your installed TensorFlow version is compatible with your Python version and operating system.

## 🚀 Run the Streamlit Application

From the project root directory, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that address in your browser to access the application.

### Live Demo

Try the deployed application:

**[Open Breast Cancer Classification App](https://deep-learning-pr1-cozrkcexkgrurrcc3zrjeg.streamlit.app/)**

If the live application requests access or displays an error, check the deployment and account permissions before sharing the link publicly.

## 💾 Saved Model

The project uses two saved artifacts:

- `models/final_model.keras` — the trained TensorFlow/Keras model.
- `models/scaler.pkl` — the fitted Scikit-learn feature scaler.

The application should apply the same preprocessing used during training before passing inputs to the model.

**Important:** The feature names, input order, scaling method, and output interpretation in the application must match the training notebook.

## 🔮 Future Improvements

- Add clear input validation and more helpful error messages.
- Improve the user interface and accessibility.
- Display a confusion matrix and evaluation charts.
- Add a model comparison summary based on measured results.
- Improve automated testing and deployment checks.
- Document model limitations and dataset provenance more thoroughly.

## ⚠️ Limitations and Disclaimer

This project is intended for **educational and demonstration purposes only**. It is not a clinically validated diagnostic system and must not be used to diagnose, rule out, or make treatment decisions about breast cancer.

Model predictions depend on the dataset, preprocessing, and training procedure. Real medical decisions require qualified healthcare professionals and appropriate clinical testing.

## 👩‍💻 Author

**Misari Dhorajiya**

Deep Learning | Artificial Intelligence | Machine Learning | Data Science

- **GitHub:** [@dhorajiyamisri](https://github.com/dhorajiyamisri)
- **Project Repository:** [Deep-Learning-PR1](https://github.com/dhorajiyamisri/Deep-Learning-PR1)
- **Live Application:** [Streamlit Demo](https://deep-learning-pr1-cozrkcexkgrurrcc3zrjeg.streamlit.app/)

---

<p align="center">
  <b>Built with Python, TensorFlow, and curiosity for Deep Learning.</b>
</p>

<p align="center">
  ⭐ If you find this project useful for learning, consider starring the repository.
</p>
