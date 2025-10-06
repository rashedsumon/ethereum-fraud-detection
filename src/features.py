# src/features.py
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler

def basic_transaction_features(df):
    """
    Derive basic features per transaction: value, gas, time-based, etc.
    Return DataFrame X_basic
    """
    X = pd.DataFrame()
    if "value" in df.columns:
        X["value"] = df["value"].fillna(0).astype(float)
    else:
        X["value"] = 0.0

    if "gas" in df.columns:
        X["gas"] = df["gas"].fillna(0).astype(float)
    else:
        X["gas"] = 0.0

    # Timestamp conversion
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            X["hour"] = ts.dt.hour.fillna(0).astype(int)
            X["day_of_week"] = ts.dt.weekday.fillna(0).astype(int)
        except Exception:
            # If timestamp already in ISO format
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            X["hour"] = ts.dt.hour.fillna(0).astype(int)
            X["day_of_week"] = ts.dt.weekday.fillna(0).astype(int)
    else:
        X["hour"] = 0
        X["day_of_week"] = 0

    return X

def graph_features(df, src_col="from_address", dst_col="to_address"):
    """
    Build a directed graph from transactions and compute per-transaction graph features.
    Return DataFrame with columns like src_out_degree, dst_in_degree, src_pagerank, etc.
    """
    # Build graph
    G = nx.DiGraph()
    # Ensure strings
    df = df.copy()
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    edges = list(zip(df[src_col], df[dst_col]))
    G.add_edges_from(edges)

    # Precompute centralities (approx)
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())

    # Map back to df
    src_pr = df[src_col].map(pagerank).fillna(0.0).astype(float)
    dst_pr = df[dst_col].map(pagerank).fillna(0.0).astype(float)

    src_outdeg = df[src_col].map(out_deg).fillna(0).astype(int)
    src_indeg = df[src_col].map(in_deg).fillna(0).astype(int)
    dst_outdeg = df[dst_col].map(out_deg).fillna(0).astype(int)
    dst_indeg = df[dst_col].map(in_deg).fillna(0).astype(int)

    gf = pd.DataFrame({
        "src_pagerank": src_pr,
        "dst_pagerank": dst_pr,
        "src_outdeg": src_outdeg,
        "src_indeg": src_indeg,
        "dst_outdeg": dst_outdeg,
        "dst_indeg": dst_indeg
    })

    # Aggregate path feature: shortest path length from src to dst if small graph
    try:
        spl = []
        for s, d in zip(df[src_col], df[dst_col]):
            try:
                l = nx.shortest_path_length(G, source=s, target=d)
            except Exception:
                l = -1
            spl.append(l)
        gf["shortest_path_len"] = spl
    except Exception:
        gf["shortest_path_len"] = -1

    return gf

def build_features(df):
    """
    Main feature builder: accepts raw df, returns features X (numpy), labels y (if present), and feature_names.
    """
    df = df.copy()
    X_basic = basic_transaction_features(df)
    X_graph = graph_features(df, src_col="from_address" if "from_address" in df.columns else "src",
                              dst_col="to_address" if "to_address" in df.columns else "dst")
    X = pd.concat([X_basic.reset_index(drop=True), X_graph.reset_index(drop=True)], axis=1)

    # Fill NaN and scale numeric columns
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = list(X.columns)

    y = None
    if "label" in df.columns:
        y = df["label"].astype(int).values

    return X_scaled, y, feature_names
