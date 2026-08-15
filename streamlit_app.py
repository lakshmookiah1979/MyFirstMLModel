"""
streamlit_app.py
-----------------
Streamlit UI for the Breast Cancer classification project.

Features:
    a. Dataset upload option (CSV)
    b. Model selection dropdown (6 trained models)
    c. Display of evaluation metrics
    d. Confusion matrix / classification report
    Bonus: results table comparing all 6 models on the uploaded test data

Run with:
    streamlit run streamlit_app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Model Explorer", layout="wide")
st.title("🩺 Breast Cancer Classification — Model Explorer")
st.write(
    "Upload a test CSV (must include the 30 feature columns and a `target` "
    "column), pick a model, and see its evaluation metrics — or compare "
    "all 6 trained models at once."
)

MODEL_DIR = "model"

# Map of display name -> saved filename
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbour": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Random Forest (Ensemble 1)": "random_forest.pkl",
    "Gradient Boosting (Ensemble 2)": "gradient_boosting.pkl",
}

# ------------------------------------------------------------------
# Load scaler + available models
# ------------------------------------------------------------------
@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)


@st.cache_resource
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


scaler = load_scaler()

available_models = {
    name: fname
    for name, fname in MODEL_FILES.items()
    if os.path.exists(os.path.join(MODEL_DIR, fname))
}

if scaler is None or not available_models:
    st.error(
        "No trained models / scaler found in the `model/` folder. "
        "Run `train_models.py` first, then push the generated `.pkl` files "
        "to GitHub so this app can load them."
    )
    st.stop()

# ------------------------------------------------------------------
# a. Dataset upload option (CSV)
# ------------------------------------------------------------------
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file (30 feature columns + a 'target' column)", type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started, e.g. the project's own `test_data.csv`.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write(f"Loaded data: **{df.shape[0]} rows, {df.shape[1]} columns**")
st.dataframe(df.head())

if "target" not in df.columns:
    st.error(
        "This CSV has no `target` column, so evaluation metrics can't be "
        "computed. Please upload a labeled test set."
    )
    st.stop()

X_uploaded = df.drop(columns=["target"])
y_uploaded = df["target"]

try:
    X_scaled = scaler.transform(X_uploaded)
except Exception as e:
    st.error(
        f"Could not scale the uploaded data — check that it has the same "
        f"30 feature columns the models were trained on. Error: {e}"
    )
    st.stop()

# ------------------------------------------------------------------
# Helper: evaluate one model on the uploaded data
# ------------------------------------------------------------------
def evaluate_model(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }
    return y_pred, metrics


# ------------------------------------------------------------------
# b. Model selection dropdown
# ------------------------------------------------------------------
st.header("2. Select a Model")
selected_model_name = st.selectbox(
    "Choose a model to inspect in detail", list(available_models.keys())
)
model = load_model(available_models[selected_model_name])

y_pred, metrics = evaluate_model(model, X_scaled, y_uploaded)

# ------------------------------------------------------------------
# c. Display of evaluation metrics
# ------------------------------------------------------------------
st.header("3. Evaluation Metrics")
cols = st.columns(len(metrics))
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, f"{value:.4f}")

# ------------------------------------------------------------------
# d. Confusion matrix / classification report
# ------------------------------------------------------------------
st.header("4. Confusion Matrix & Classification Report")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_uploaded, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_uploaded, y_pred,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(report_df)

# ------------------------------------------------------------------
# Bonus: Compare ALL models on this same uploaded test data
# ------------------------------------------------------------------
st.header("5. Compare All Models on This Test Data")

if st.button("Run comparison across all 6 models"):
    comparison_rows = []
    with st.spinner("Evaluating all models..."):
        for name, fname in available_models.items():
            m = load_model(fname)
            _, m_metrics = evaluate_model(m, X_scaled, y_uploaded)
            m_metrics["Model"] = name
            comparison_rows.append(m_metrics)

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    comparison_df = comparison_df[
        ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
    ]
    comparison_df = comparison_df.sort_values(by="Accuracy", ascending=False)

    st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

    st.bar_chart(comparison_df["Accuracy"])

st.markdown("---")
st.caption(
    "Breast Cancer Wisconsin (Diagnostic) dataset — for educational purposes only, "
    "not a medical diagnostic tool."
)
