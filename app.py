from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "is_spam"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Spam Email Classifier",  layout="wide")


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.joblib")


@st.cache_data
def load_results():
    path = MODEL_DIR / "results.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


# ---------- Sidebar ----------
st.sidebar.title("Panel")
uploaded = st.sidebar.file_uploader("Upload test CSV file", type="csv")
model_name = st.sidebar.selectbox("Choose a model to check its result", list(MODEL_FILES))
st.sidebar.caption("The CSV needs all 57 feature columns plus the `is_spam` column.")

# ---------- Main ----------
st.title("Spam Email Classifier")
st.write("Compare five trained classifiers on the Spambase test data. "
         "Models are loaded from disk; nothing is retrained.")

results = load_results()
if results is not None:
    with st.expander("Test-set results for all models (from training)", expanded=False):
        st.dataframe(results, use_container_width=True)
        st.bar_chart(results[["Accuracy", "AUC", "F1", "MCC"]])

if uploaded is None:
    st.info("Upload a CSV in the sidebar to begin (try `test_data.csv`).")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the file as CSV: {e}")
    st.stop()

scaler = load_scaler()
expected = list(scaler.feature_names_in_)

if TARGET not in data.columns:
    st.error(f"The file must contain the target column `{TARGET}`.")
    st.stop()

missing = [c for c in expected if c not in data.columns]
if missing:
    st.error(f"Missing {len(missing)} feature column(s), e.g. {missing[:5]}")
    st.stop()

X = data[expected]          
y = data[TARGET]

if y.nunique() < 2:
    st.warning("The target has only one class in this file, so AUC cannot be computed.")

st.subheader(f"Results: {model_name}")
st.caption(f"{len(data)} rows uploaded")

model = load_model(MODEL_FILES[model_name])
X_s = scaler.transform(X)
y_pred = model.predict(X_s)
y_prob = model.predict_proba(X_s)[:, 1]

cols = st.columns(6)
cols[0].metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
if y.nunique() > 1:
    cols[1].metric("AUC", f"{roc_auc_score(y, y_prob):.4f}")
else:
    cols[1].metric("AUC", "n/a")
cols[2].metric("Precision", f"{precision_score(y, y_pred, zero_division=0):.4f}")
cols[3].metric("Recall", f"{recall_score(y, y_pred, zero_division=0):.4f}")
cols[4].metric("F1", f"{f1_score(y, y_pred, zero_division=0):.4f}")
cols[5].metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=["Not spam", "Spam"], ax=ax, colorbar=False
    )
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, y_pred, target_names=["Not spam", "Spam"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)