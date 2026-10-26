import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# PAGE CONFIG

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)

# PATHS

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

st.write("BASE_DIR:", BASE_DIR)
st.write("MODEL_PATH:", MODEL_PATH)
st.write("MODEL EXISTS:", MODEL_PATH.exists())
st.write("SCALER EXISTS:", SCALER_PATH.exists())