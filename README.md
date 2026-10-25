<p align="center">
  <img src="project_banner.png" alt="Deep Learning PR1 — Breast Cancer Classification | Red & White Skill Education" width="100%">
</p>

<h1 align="center">Deep Learning Practical — PR 1</h1>
<h3 align="center">Breast Cancer Classification Using Artificial Neural Networks</h3>

<p align="center">
  <strong>Red & White Skill Education</strong><br>
  Academic Practical Project · Student: <strong>Misari Dhorajiya</strong>
</p>

<p align="center">
  <a href="DL_PR1.ipynb">📓 Open Notebook</a> &nbsp;·&nbsp;
  <a href="DL_PR1.html">🌐 Read HTML Report</a> &nbsp;·&nbsp;
  <a href="plots/">📊 Explore Plots</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep%20Learning-ANN-6C63FF?style=flat-square" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-0F9D8A?style=flat-square" alt="Binary Classification">
  <img src="https://img.shields.io/badge/Institute-Red%20%26%20White-E63946?style=flat-square" alt="Red and White Skill Education">
</p>

01 · Project at a Glance
This project explores how Artificial Neural Networks (ANNs) can learn patterns from numerical diagnostic features and classify observations into the two target categories provided by the Breast Cancer Wisconsin Diagnostic dataset.

Rather than evaluating only one network, this practical builds a learning journey from exploratory data analysis to baseline neural networks, activation-function experiments, and techniques designed to help control overfitting.
Educational disclaimer: This project is for learning and experimentation only. It is not a clinically validated tool and must not be used to diagnose or make medical decisions.

Project objectives
- Explore the dataset, target distribution, and feature correlations.
- Prepare data with a stratified train/test split and feature scaling.
- Build and evaluate a Single Layer Perceptron (SLP) and Multi-Layer Perceptron (MLP).
- Compare activation functions and training strategies.
- Investigate Early Stopping, Dropout, and L1/L2 regularization.
- Compare models using suitable classification metrics and confusion matrices.
02 · Dataset & Preparation
The dataset is loaded from sklearn.datasets.load_breast_cancer(as_frame=True).
Dataset detail	Value
Observations	569
Input features	30
Malignant observations (target 0)	212
Benign observations (target 1)	357
Training / test split	80% / 20%
Split strategy	Stratified
Scaling	StandardScaler


Preprocessing workflow: Load and inspect data → explore target and correlations → split the data → fit the scaler on training features only → transform train and test features. Fitting preprocessing only on training data helps prevent test-data leakage.
Dataset visualizations
<table>
  <tr>
    <td width="50%" align="center">
      <strong>Target class distribution</strong><br>
      <img src="plots/class_distribution.png" alt="Target class distribution chart" width="100%"><br>
      <sub>Shows the number of observations in each target class and helps interpret the class balance.</sub>
    </td>
    <td width="50%" align="center">
      <strong>Feature correlation heatmap</strong><br>
      <img src="plots/correlation_heatmap.png" alt="Feature correlation heatmap" width="100%"><br>
      <sub>Highlights relationships among input features and helps identify strongly correlated measurements.</sub>
    </td>
  </tr>
</table>

03 · Seven Practical Tasks
Task 1 — Exploratory Data Analysis & Preprocessing
Inspect the dataset, understand the target labels, visualize class distribution and feature correlation, then perform a stratified split and standardize features. This establishes a consistent foundation for the experiments.
Task 2 — Single Layer Perceptron (SLP)
Create a baseline network with a single sigmoid output neuron for binary classification. Training curves help reveal how loss and accuracy change over epochs; the confusion matrix shows which classes were correctly or incorrectly predicted.
<table>
  <tr>
    <td width="50%" align="center">
      <strong>SLP training curves</strong><br>
      <img src="plots/slp_training_curves.png" alt="SLP training curves" width="100%"><br>
      <sub>Training and validation behavior over epochs.</sub>
    </td>
    <td width="50%" align="center">
      <strong>SLP confusion matrix</strong><br>
      <img src="plots/slp_confusion_matrix.png" alt="SLP confusion matrix" width="100%"><br>
      <sub>Breakdown of correct predictions and classification errors.</sub>
    </td>
  </tr>
</table>

Task 3 — Multi-Layer Perceptron (MLP) & Activation Functions
Use hidden layers to learn nonlinear patterns. Compare ReLU, Tanh, and Sigmoid variants using the experiment's recorded results. The model's confusion matrix provides a class-level view of its predictions.
<table>
  <tr>
    <td width="50%" align="center">
      <strong>Activation function comparison</strong><br>
      <img src="plots/activation_comparison.png" alt="Activation function comparison" width="100%"><br>
      <sub>Compares the observed performance of the tested activation variants.</sub>
    </td>
    <td width="50%" align="center">
      <strong>MLP confusion matrix</strong><br>
      <img src="plots/mlp_confusion_matrix.png" alt="MLP confusion matrix" width="100%"><br>
      <sub>Shows the MLP's correct and incorrect predictions by class.</sub>
    </td>
  </tr>
</table>

Task 4 — Early Stopping
Early Stopping monitors validation loss and halts training when validation performance no longer improves within the configured patience. It can reduce unnecessary training and help limit overfitting.
<table>
  <tr>
    <td width="50%" align="center">
      <strong>Early stopping loss curve</strong><br>
      <img src="plots/early_stopping_curve.png" alt="Early stopping curve" width="100%"><br>
      <sub>Tracks validation loss and the stopping point during training.</sub>
    </td>
    <td width="50%" align="center">
      <strong>Baseline vs. Early Stopping</strong><br>
      <img src="plots/early_stopping_vs_normal.png" alt="Early stopping versus baseline training" width="100%"><br>
      <sub>Compares the recorded training behavior of the baseline and callback-based runs.</sub>
    </td>
  </tr>
</table>

Task 5 — Dropout Regularization
Dropout randomly disables a fraction of hidden-layer activations during training. Compare the tested dropout rates using validation results, then inspect the selected model's classification errors.
<table>
  <tr>
    <td width="50%" align="center">
      <strong>Dropout-rate comparison</strong><br>
      <img src="plots/dropout_comparison.png" alt="Dropout rate comparison" width="100%"><br>
      <sub>Compares validation performance for the tested dropout rates.</sub>
    </td>
    <td width="50%" align="center">
      <strong>Dropout confusion matrix</strong><br>
      <img src="plots/dropout_confusion_matrix.png" alt="Dropout confusion matrix" width="100%"><br>
      <sub>Summarizes correct and incorrect predictions for the evaluated dropout model.</sub>
    </td>
  </tr>
</table>

Task 6 — L1, L2 & Combined Regularization
Compare weight-penalty strategies intended to constrain model complexity:
- L1: penalizes the absolute values of weights and can encourage sparse weights.
- L2: penalizes squared weights and discourages excessively large weights.
- L1 + L2: combines both penalties.
<table>
  <tr>
    <td width="50%" align="center">
      <strong>Regularization accuracy</strong><br>
      <img src="plots/regularization_accuracy.png" alt="Regularization accuracy comparison" width="100%"><br>
      <sub>Compares test accuracy across the evaluated regularization settings.</sub>
    </td>
    <td width="50%" align="center">
      <strong>Regularization comparison</strong><br>
      <img src="plots/regularization_comparison.png" alt="Regularization comparison" width="100%"><br>
      <sub>Visual comparison of the regularization experiments; interpret it using the plot labels and notebook outputs.</sub>
    </td>
  </tr>
</table>

Task 7 — Final Combined Model
Bring together the selected techniques required by the practical, then evaluate the final model. The final confusion matrix should be interpreted alongside the notebook's classification report and other metrics.
<p align="center">
  <strong>Final model confusion matrix</strong><br>
  <img src="plots/final_model_confusion_matrix.png" alt="Final model confusion matrix" width="68%">
</p>

04 · Evaluation: More Than Accuracy
The notebook's actual outputs should be used when reporting final scores. No performance values are hard-coded here because results must reflect the model that was successfully trained and evaluated.
Metric	What it tells us
Accuracy	Overall fraction of correct predictions
Precision	How often positive predictions are correct
Recall	How many actual positive cases are identified
F1-score	Balance between precision and recall
Confusion matrix	The counts of correct and incorrect predictions by class


For a medical dataset, false negatives and false positives have different implications. Always state which class is treated as positive and interpret recall, precision, and the confusion matrix together.
05 · Tools & Technologies
- Python — core programming language
- Pandas & NumPy — data handling and numerical computation
- Matplotlib & Seaborn — visual analysis
- Scikit-learn — dataset access, splitting, scaling, and evaluation utilities
- TensorFlow / Keras — neural-network layers, training, and callbacks (as used in the notebook)
- Jupyter Notebook & VS Code — development environment
06 · Run Locally
1. Clone the repository
git clone https://github.com/dhorajiyamisri/Deep-Learning-PR1.git
cd Deep-Learning-PR1
2. Install dependencies
python -m pip install -r requirements.txt
3. Run the notebook
Open DL_PR1.ipynb in VS Code or Jupyter Notebook, select the Python environment where the dependencies are installed, then run the cells in order. The exported HTML report can be opened in a browser.
07 · Repository Structure
Deep-Learning-PR1/
├── README.md
├── project_banner.png
├── DL_PR1.ipynb
├── DL_PR1.html
├── requirements.txt
└── plots/
    ├── activation_comparison.png
    ├── class_distribution.png
    ├── correlation_heatmap.png
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
08 · Key Learning Outcomes
- Why stratified splitting and training-only scaling matter.
- How a single-neuron baseline differs from a multilayer network.
- How activation functions affect nonlinear learning.
- How validation monitoring supports Early Stopping decisions.
- How Dropout and weight regularization can help control overfitting.
- Why model selection should consider multiple evaluation metrics.
👩‍💻 Author
Misari Dhorajiya
GitHub: @dhorajiyamisri
Repository: Deep-Learning-PR1
<p align="center">
  <strong>RED & WHITE SKILL EDUCATION</strong><br>
  <em>Deep Learning Practical — PR 1</em><br>
  Built for academic learning and experimentation.
</p>
