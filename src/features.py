# src/features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def basic_address_features(df):
    """
    Extract numeric features from an address-level aggregated dataset.
    """
    df = df.copy()
    
    # Drop non-feature columns
    drop_cols = ["Unnamed: 0", "Index", "Address"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    return X

def build_features(df):
    """
    Main feature builder: accepts raw df, returns scaled features X (numpy array),
    labels y (if present), and feature_names.
    """
    df = df.copy()
    
    # Extract numeric features
    X = basic_address_features(df)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names = X.columns.tolist()
    
    # Extract labels if present
    y = None
    if "FLAG" in df.columns:
        y = df["FLAG"].astype(int).values
    
    return X_scaled, y, feature_names
