# DiskSparseAdam
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21296366.svg)(https://doi.org/10.5281/zenodo.21296366)]

An out-of-core Sparse Adam optimizer for PyTorch, designed to train large-scale Graph Neural Networks (GNNs) and Knowledge Graph Embeddings (KGE) on consumer-grade GPUs.

## Problem

Training graphs with tens of millions of nodes often requires specialized infrastructure or distributed frameworks. The primary bottleneck is not the model weights themselves, but the optimizer states (momentum and variance), which typically scale as $\mathcal{O}(N)$ with the total graph size and exhaust available VRAM/RAM.

## Approach

**DiskSparseAdam** stores optimizer states on NVMe/SSD storage via `np.memmap` and streams only the required node states into GPU memory using an asynchronous double-buffering queue.

As a result, **optimizer memory usage scales with the number of accessed nodes in a batch $\mathcal{O}(B)$ rather than the total graph size $\mathcal{O}(N)$.**

## Key Features

- **Near-zero optimizer VRAM overhead.** Only the states for active batch indices are loaded into GPU memory. The rest remains on disk.
- **Async I/O.** Background threads write updated weights and states back to disk, preventing the GPU from stalling during `step()`.
- **Safe duplicate aggregation.** Correctly aggregates gradients for highly connected hub nodes appearing multiple times in the same batch.
- **Hyperbolic workflow support.** Includes optional conformal gradient scaling and retraction bounds for Poincaré embeddings, compatible with non-Euclidean representation learning.
- **Minimal integration effort.** Integrates into existing training loops by explicitly passing sparse gradients.

## Best Suited For

- Knowledge Graph Embeddings (TransE, RotatE, Hyperbolic KGE)
- Graph Neural Networks (GNNs)
- Hyperbolic Representation Learning
- Large-scale Retrieval Systems

## Installation

```bash
pip install disk-sparse-adam
```

## Usage Example

```python
import torch
import numpy as np
from dsa import DiskSparseRiemannianAdam

# 1. Initialize embeddings directly on disk
num_nodes, dim = 10_000_000, 1024
weights_disk = np.memmap(
    "./cache/emb.npy", dtype="float32", mode="w+", shape=(num_nodes, dim)
)

# 2. Initialize optimizer
optimizer = DiskSparseRiemannianAdam(
    cache_dir="./cache",
    params_dict={"graph_nodes": weights_disk},
    lr=0.001,
    k_value=1.0,  # optional, for conformal gradient scaling in Poincaré Ball
)

# 3. Training loop
# F_batch, R_batch = ...
loss.backward()

# Pass sparse gradients directly to the optimizer
optimizer.step({"graph_nodes": (batch_indices, node_embeddings.grad)})

# 4. Graceful shutdown (flushes OS buffers to SSD)
optimizer.shutdown()
```

## Enterprise

`disk-sparse-adam` provides a low-level memory management engine for sparse structures. If your organization requires a comprehensive end-to-end framework for **Topological Data Analysis**, **Neuro-Symbolic Reasoning**, or **Explainable AI (XAI)** in Drug Discovery and FinTech, please reach out regarding the **Topological Knowledge Kernel (TKK)** Enterprise solution.

