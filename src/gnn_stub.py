# src/gnn_stub.py
"""
GNN STUB / NOTES

Graph Neural Network experimentation requires heavy libraries (PyTorch + DGL or PyTorch Geometric).
We provide a stub and guidance here. DO NOT pip install these in the main requirements unless you have GPU/CUDA or need CPU builds.

Recommended approach:
1) Install PyTorch for your platform: https://pytorch.org/get-started/locally/
2) Install DGL (recommended) or PyG:
   - CPU-only DGL (example): pip install dgl -f https://data.dgl.ai/wheels/repo.html
   - OR CPU-only PyG: follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Example structure:
- Build a node-level graph (addresses as nodes), edges are transactions.
- Create node features (aggregated transactional features).
- Label nodes by whether they've been flagged in your label set.
- Use GraphSAGE / GCN for node classification.

This file would contain dataset -> DGLGraph conversion and a training loop.
"""

# Placeholder functions
def build_dgl_graph(df):
    raise NotImplementedError("Install DGL/PyG and implement this function for GNN experiments.")
