# src/data.py
import pandas as pd
import numpy as np

def load_data(path):
    """
    Load CSV from given path. Returns DataFrame.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load data at {path}: {e}")
    return df

def prepare_labels(df, label_cols=("isFraud", "label", "fraud")):
    """
    Ensure there is a 'label' column (0/1). If none of label_cols exist,
    create a heuristic label for experimentation:
      - mark transactions with extremely high value or unusually high frequency as suspicious.
    NOTE: heuristic labeling is experimental only — replace with ground truth.
    """
    for col in label_cols:
        if col in df.columns:
            df["label"] = df[col].astype(int)
            return df

    # Heuristic: mark top 0.5% by value OR addresses with >N tx in short time as suspicious
    df = df.copy()
    if "value" in df.columns:
        high_value_cut = df["value"].quantile(0.995)
        df["label"] = (df["value"] >= high_value_cut).astype(int)
    else:
        # fallback: flag top 0.5% by gas or by tx_count if exists
        if "gas" in df.columns:
            df["label"] = (df["gas"] >= df["gas"].quantile(0.995)).astype(int)
        else:
            df["label"] = 0
    return df
