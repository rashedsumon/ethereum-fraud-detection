# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.data import load_data, prepare_labels
from src.features import build_features
from src.model import load_model, predict_proba
import os

MODEL_PATH = "models/xgb_model.joblib"
DEFAULT_DATA_PATH = "/kaggle/input/ethereum-frauddetection-dataset/transaction_dataset.csv"
LOCAL_DATA_PATH = "data/raw/transaction_dataset.csv"

st.set_page_config(page_title="Ethereum Fraud Detector", layout="wide")

st.title("Ethereum Fraud Detection — Demo")
st.markdown("Upload a CSV of transactions or use the packaged dataset.")

uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
use_default = st.checkbox("Use default dataset path (Kaggle)", value=False)

if uploaded:
    df = pd.read_csv(uploaded)
elif use_default:
    df = load_data(DEFAULT_DATA_PATH)
elif os.path.exists(LOCAL_DATA_PATH):
    df = load_data(LOCAL_DATA_PATH)
else:
    st.warning("No dataset loaded. Upload or place dataset at data/raw/transaction_dataset.csv or enable default Kaggle path.")
    st.stop()

st.write("Dataset preview:", df.head())

# Prepare data & features
df = prepare_labels(df)  # safe — will not overwrite if labels exist
X, y, feature_names = build_features(df)

st.sidebar.header("Model")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.sidebar.success("Loaded model from models/xgb_model.joblib")
else:
    st.sidebar.warning("Model not found. Please run `python train.py` to train and save a model.")
    model = None

if model is not None:
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
        st.download_button("Download flagged CSV", flagged.to_csv(index=False), file_name="flagged.csv")
        st.write("Feature importance (top 20):")
        try:
            importances = model.get_booster().get_score(importance_type="gain")
            importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
            st.table(importances)
        except Exception as e:
            st.write("Can't show feature importance:", e)

st.info("To serve a production API, start uvicorn with `uvicorn src.api:app --host 0.0.0.0 --port 8000`")
