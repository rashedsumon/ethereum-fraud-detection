# train.py
import pandas as pd
import numpy as np
from src.model import train_xgb, save_model

# Load dataset
data_path = "/kaggle/input/ethereum-frauddetection-dataset/transaction_dataset.csv"
df = pd.read_csv(data_path)

# Example: Assuming 'label' is the target column
TARGET_COL = "label"

# Select features and target
X = df.drop(columns=[TARGET_COL]).values
y = df[TARGET_COL].values

# Train model
model = train_xgb(X, y)

# Save trained model
save_model(model, "xgb_model.pkl")
print("Model saved to xgb_model.pkl")
