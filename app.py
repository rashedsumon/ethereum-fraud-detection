# app.py
import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.features import build_features

MODEL_PATH = "models/model.pkl"
DEFAULT_DATA_PATH = "/kaggle/input/ethereum-frauddetection-dataset/transaction_dataset.csv"
LOCAL_DATA_PATH = "data/raw/transaction_dataset.csv"

st.set_page_config(page_title="Ethereum Fraud Detector", layout="wide")
st.title("Ethereum Fraud Detection — Demo")
st.markdown("Upload a CSV of transactions or use the packaged dataset.")

# --- Load dataset ---
uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
use_default = st.checkbox("Use default Kaggle dataset", value=False)

if uploaded:
    df = pd.read_csv(uploaded)
elif use_default:
    df = pd.read_csv(DEFAULT_DATA_PATH)
elif os.path.exists(LOCAL_DATA_PATH):
    df = pd.read_csv(LOCAL_DATA_PATH)
else:
    st.warning(
        "No dataset loaded. Upload a CSV or place it at data/raw/transaction_dataset.csv "
        "or enable default Kaggle path."
    )
    st.stop()

st.write("Dataset preview:", df.head())

# --- Prepare features ---
X, y, feature_names = build_features(df)

# --- Load or train model ---
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.sidebar.success(f"Loaded model from {MODEL_PATH}")
else:
    st.sidebar.info("Model not found. Training a new model...")
    if y is None:
        st.error("No label column ('FLAG') found in dataset. Cannot train model.")
        st.stop()
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Only create directory if it's not empty
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    st.sidebar.success(f"Model trained and saved at {MODEL_PATH}")

# --- Run inference ---
st.sidebar.header("Inference")
threshold = st.sidebar.slider("Flag threshold (probability)", 0.0, 1.0, 0.5)

if st.sidebar.button("Run inference"):
    probs = model.predict_proba(X)[:, 1]  # probability of fraud
    df_results = df.copy()
    df_results["fraud_prob"] = probs
    df_results["flagged"] = df_results["fraud_prob"] >= threshold

    flagged = df_results[df_results["flagged"]].sort_values("fraud_prob", ascending=False)
    st.write("Top flagged addresses:")
    st.dataframe(flagged.head(200))

    st.download_button(
        "Download flagged CSV", flagged.to_csv(index=False), file_name="flagged.csv"
    )

    # Feature importance
    st.write("Feature importance (top 20):")
    try:
        importances = model.feature_importances_
        feat_imp = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )[:20]
        st.table(pd.DataFrame(feat_imp, columns=["Feature", "Importance"]))
    except Exception as e:
        st.write("Cannot display feature importance:", e)

st.info(
    "To serve as a production API, run: `uvicorn src.api:app --host 0.0.0.0 --port 8000`"
)
