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
st.set_page_config(
    page_title="Breast Cancer Model Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Breast Cancer Classification — Model Explorer")
st.caption(
    "Upload a labeled test CSV, pick a model, and explore its evaluation "
    "metrics — or compare all 6 trained models side by side."
)
st.divider()

MODEL_DIR = "model"

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
# Sidebar — all controls live here
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("1. Upload Test Data")
    uploaded_file = st.file_uploader(
        "CSV with 30 feature columns + 'target'", type=["csv"]
    )

    st.subheader("2. Select a Model")
    selected_model_name = st.selectbox(
        "Model to inspect in detail", list(available_models.keys())
    )

    st.divider()
    st.caption(
        "Breast Cancer Wisconsin (Diagnostic) dataset — "
        "for educational purposes only, not a medical diagnostic tool."
    )

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started, e.g. this project's own `test_data.csv`.")
    st.stop()

with st.spinner("Reading uploaded data..."):
    df = pd.read_csv(uploaded_file)

with st.expander(f"📄 Preview uploaded data — {df.shape[0]} rows, {df.shape[1]} columns", expanded=False):
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


with st.spinner(f"Running {selected_model_name}..."):
    model = load_model(available_models[selected_model_name])
    y_pred, metrics = evaluate_model(model, X_scaled, y_uploaded)

# ------------------------------------------------------------------
# Quick summary badges
# ------------------------------------------------------------------
n_malignant_pred = int((y_pred == 0).sum())
n_benign_pred = int((y_pred == 1).sum())

badge_col1, badge_col2, badge_col3 = st.columns(3)
badge_col1.metric("Selected Model", selected_model_name.split("(")[0].strip())
badge_col2.metric("🔴 Predicted Malignant", n_malignant_pred)
badge_col3.metric("🟢 Predicted Benign", n_benign_pred)

st.divider()

# ------------------------------------------------------------------
# Tabs: Metrics | Confusion Matrix & Report | Compare All Models
# ------------------------------------------------------------------
tab_metrics, tab_matrix, tab_compare = st.tabs(
    ["📊 Evaluation Metrics", "🧩 Confusion Matrix & Report", "⚖️ Compare All Models"]
)

with tab_metrics:
    st.subheader(f"Metrics — {selected_model_name}")
    cols = st.columns(len(metrics))
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

with tab_matrix:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_uploaded, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="rocket_r",
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            ax=ax, cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Classification Report")
        report_dict = classification_report(
            y_uploaded, y_pred,
            target_names=["Malignant (0)", "Benign (1)"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

with tab_compare:
    st.subheader("All 6 Models on This Test Data")
    run_compare = st.button("▶️ Run comparison across all 6 models", use_container_width=True)

    if run_compare:
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

        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color="#c6f6d5"),
            use_container_width=True,
        )

        st.bar_chart(comparison_df["Accuracy"])
    else:
        st.caption("Click the button above to evaluate and rank all 6 models on your uploaded data.")
