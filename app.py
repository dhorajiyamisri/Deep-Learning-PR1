import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# =========================
# HEADER
# =========================

st.title("🧬 Breast Cancer Classification")

st.write(
    "Deep Learning project built using an Artificial Neural Network."
)

st.info(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)

# =========================
# CHECK FILES
# =========================

if not MODEL_PATH.exists():
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

if not SCALER_PATH.exists():
    st.error(f"Scaler file not found:\n{SCALER_PATH}")
    st.stop()

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


try:
    model, scaler = load_artifacts()
    st.success("✅ Trained model and scaler loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================
# MODEL INFO
# =========================

st.subheader("Model Information")

st.write("Input Features:", model.input_shape[-1])
st.write("Output Units:", model.output_shape[-1])

# =========================
# FEATURE NAMES
# =========================

feature_names = [

    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

# =========================
# DEFAULT VALUES
# =========================

dataset = load_breast_cancer(as_frame=True)
default_values = dataset.data.mean()

# =========================
# INPUT SECTION
# =========================

st.subheader("Enter Patient Features")

col1, col2 = st.columns(2)

inputs = []

for i, feature in enumerate(feature_names):

    default = float(default_values.iloc[i])

    if i % 2 == 0:

        value = col1.number_input(
            feature,
            value=default,
            format="%.6f"
        )

    else:

        value = col2.number_input(
            feature,
            value=default,
            format="%.6f"
        )

    inputs.append(value)

# =========================
# PREDICT BUTTON
# =========================

st.markdown("###")

predict = st.button(
    "🔍 Predict Cancer Type",
    use_container_width=True
)

# =========================
# PREDICTION
# =========================

if predict:

    try:

        input_array = np.array(inputs).reshape(1, -1)

        scaled_input = scaler.transform(input_array)

        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        probability = float(prediction[0][0])

        st.subheader("Prediction Result")

        st.metric(
            "Benign Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        if probability >= 0.5:

            st.success(
                f"✅ Benign Tumor Detected "
                f"({probability*100:.2f}%)"
            )

        else:

            st.error(
                f"⚠️ Malignant Tumor Detected "
                f"({(1-probability)*100:.2f}%)"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Developed using TensorFlow, Scikit-Learn and Streamlit."
)