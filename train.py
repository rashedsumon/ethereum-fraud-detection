# train.py
import os
import pandas as pd
from src.model import train_xgb, save_model

# Paths
DATA_PATH = "data/raw/transaction_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")

# Create models directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Example: assuming 'label' is the target column
TARGET_COL = "label"

# Features and target
X = df.drop(columns=[TARGET_COL]).values
y = df[TARGET_COL].values

# Train model
model = train_xgb(X, y)

# Save model
save_model(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
