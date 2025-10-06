# src/model.py
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_xgb(X, y, params=None):
    """
    Train a simple XGBoost classifier. Expects X as numpy array and y as 0/1.
    """
    if y is None:
        raise ValueError("Labels `y` required for training.")
    params = params or {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "n_jobs": -1
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y if len(set(y))>1 else None, random_state=42)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
    # Print eval
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:,1]
    print(classification_report(y_val, preds))
    try:
        print("Val ROC AUC:", roc_auc_score(y_val, proba))
    except Exception:
        pass
    return model

def predict_proba(model, X):
    """
    Returns probability for the positive class.
    """
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        # fallback for models without predict_proba
        preds = model.predict(X)
        probs = preds
    return probs

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
