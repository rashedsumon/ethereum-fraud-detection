# app.py
import streamlit as st
import pandas as pd
import os
from src.data import load_data, prepare_labels
from src.features import build_features
from src.model import load_model, predict_proba, train_xgb, save_model

# -----------------------
# Config
# -----------------------
MODEL_PATH = "models/xgb_model.joblib"
MODEL_DIR = "models"
DEFAULT_DATA_PATH = "/kaggle/input/ethereum-frauddetection-dataset/transaction_dataset.csv"
LOCAL_DATA_PATH = "data/raw/transaction_dataset.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Ethereum Fraud Detector", layout="wide")
st.title("Ethereum Fraud Detection — Demo")
st.markdown("Upload a CSV of transactions or use the packaged dataset.")

# -----------------------
# Load Dataset
# -----------------------
uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
use_default = st.checkbox("Use default dataset path (Kaggle)", value=False)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_default:
    df = load_data(DEFAULT_DATA_PATH)
elif os.path.exists(LOCAL_DATA_PATH):
    df = load_data(LOCAL_DATA_PATH)
else:
    st.warning(
        "No dataset loaded. Upload a CSV, place it at `data/raw/transaction_dataset.csv`, or enable default Kaggle path."
    )
    st.stop()

st.write("Dataset preview:", df.head())

# -----------------------
# Prepare Features
# -----------------------
df = prepare_labels(df)
X, y, feature_names = build_features(df)

# -----------------------
# Load or Train Model
# -----------------------
st.sidebar.header("Model")

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.sidebar.success(f"Loaded model from `{MODEL_PATH}`")
else:
    st.sidebar.warning("Model not found. Training a new model...")
    with st.spinner("Training model, please wait..."):
        model = train_xgb(X, y)
        save_model(model, MODEL_PATH)
    st.sidebar.success(f"Model trained and saved to `{MODEL_PATH}`")

# -----------------------
# Inference
# -----------------------
st.sidebar.subheader("Inference")
threshold = st.sidebar.slider("Flag threshold (probability)", 0.0, 1.0, 0.5)

if st.sidebar.button("Run inference"):
    probs = predict_proba(model, X)
    df_results = df.copy()
    df_results["fraud_prob"] = probs
    df_results["flagged"] = df_results["fraud_prob"] >= threshold

    st.write("Top flagged addresses:")
    flagged = df_results[df_results["flagged"]].sort_values("fraud_prob", ascending=False)
    st.dataframe(flagged.head(200))

    st.download_button(
        "Download flagged CSV",
        flagged.to_csv(index=False),
        file_name="flagged.csv"
    )

    st.write("Feature importance (top 20):")
    try:
        importances = model.get_booster().get_score(importance_type="gain")
        importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
        st.table(importances)
    except Exception as e:
        st.write("Can't show feature importance:", e)

# -----------------------
# Info
# -----------------------
st.info(
    "To serve a production API, start uvicorn with "
    "`uvicorn src.api:app --host 0.0.0.0 --port 8000`"
)
