import streamlit as st
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"

st.set_page_config(page_title="Transmission Line Fault Detection",
                   page_icon="⚡",
                   layout="centered")

st.title("Three-Phase Transmission Line Fault Detection")
st.write("Enter the six electrical measurements to predict whether a fault is present.")

@st.cache_resource
def load_models():
    logistic_data = np.load(RESULTS_DIR / "logistic_models.npz")
    rbf_package = joblib.load(MODELS_DIR / "best_rbf_svm.joblib")
    return logistic_data, rbf_package

try:
    logistic_data, rbf_package = load_models()
except Exception as error:
    st.error(f"Could not load saved models: {error}")
    st.stop()

with st.form("fault_form"):
    st.subheader("Electrical Measurements")

    col1, col2 = st.columns(2)

    with col1:
        Ia = st.number_input("Ia", value=0.0, format="%.6f")
        Ib = st.number_input("Ib", value=0.0, format="%.6f")
        Ic = st.number_input("Ic", value=0.0, format="%.6f")

    with col2:
        Va = st.number_input("Va", value=0.0, format="%.6f")
        Vb = st.number_input("Vb", value=0.0, format="%.6f")
        Vc = st.number_input("Vc", value=0.0, format="%.6f")

    submitted = st.form_submit_button("Detect Fault")

if submitted:
    X_raw = np.array([[Ia, Ib, Ic, Va, Vb, Vc]], dtype=float)

    # Logistic Regression Set Q
    X_q_raw = np.column_stack((X_raw, X_raw ** 2))

    mean_q = logistic_data["mean_Q"]
    std_q = logistic_data["std_Q"]
    theta_q = logistic_data["theta_Q"]

    std_q = np.where(std_q == 0, 1, std_q)
    X_q = (X_q_raw - mean_q) / std_q
    X_q = np.column_stack((np.ones(X_q.shape[0]), X_q))

    probability = 1 / (1 + np.exp(-np.clip(X_q @ theta_q, -500, 500)))
    logistic_prediction = int(probability[0] >= 0.5)

    # RBF SVM
    mean_svm = rbf_package["mean"]
    std_svm = np.where(rbf_package["std"] == 0, 1,
                       rbf_package["std"])

    X_svm = (X_raw - mean_svm) / std_svm
    svm_prediction = int(rbf_package["model"].predict(X_svm)[0])

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Logistic Set Q")
        if logistic_prediction == 1:
            st.error("Fault Detected")
        else:
            st.success("No Fault Detected")
        st.write(f"Predicted fault probability: {probability[0]:.2%}")

    with col2:
        st.markdown("### RBF SVM")
        if svm_prediction == 1:
            st.error("Fault Detected")
        else:
            st.success("No Fault Detected")

    st.caption("Predictions are based on the saved models and their training preprocessing.")
