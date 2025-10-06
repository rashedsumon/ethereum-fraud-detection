# Ethereum Fraud Detection — Streamlit + API + Training

## Overview
This repo demonstrates a full pipeline for building ML systems to detect suspicious activity on blockchain (Ethereum-style) transaction data, using on-chain transaction CSV data. It includes:
- Data ingestion & cleaning
- Feature engineering (transactional + graph-based features)
- Model training (XGBoost) with joblib saving
- Streamlit app (`app.py`) for interactive inference
- FastAPI (`src/api.py`) to expose prediction endpoint
- Optional GNN stub for experimentation

## Project layout
(See file tree in repository root.)

## Dataset
This project expects your dataset at:
`/kaggle/input/ethereum-frauddetection-dataset/transaction_dataset.csv`
or you can place a copy at `data/raw/transaction_dataset.csv`.

## Quick start (local)
1. Create virtual environment with Python 3.11.0:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
