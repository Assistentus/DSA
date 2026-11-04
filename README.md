
# DiskSparseAdam (DSA)

[![Tests](https://github.com/Assistentus/DSA/actions/workflows/test.yml/badge.svg)](https://github.com/Assistentus/DSA/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://doi.org/10.5281/zenodo.21296366"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21296366.svg" alt="DOI"></a>

<p align="center">
  <img src="banner.png" alt="DiskSparseAdam Banner" width="100%">
</p>

**DiskSparseAdam (DSA)** is a drop-in Out-of-Core Sparse Adam optimizer for PyTorch. It reduces optimizer memory footprint from $\mathcal{O}(N)$ to $\mathcal{O}(B)$, enabling the training of very large graph embeddings and Knowledge Graph representations on **consumer-grade GPUs and workstations**, as well as cloud environments like **Kaggle** and **Google Colab** — at the price of disk-bound optimizer I/O (see [Performance Notes](#-performance-notes)).

---

## 🚀 Key Features

* **Near-Zero VRAM Overhead:** Optimizer momentum ($m$), variance ($v$), weights ($w$) and per-row step counters ($t$) reside on NVMe/SSD storage via `np.memmap`. Only active batch parameters $\mathcal{O}(B)$ are loaded into GPU memory.
* **Lossless Async I/O:** A background worker thread with double-buffering (`AsyncDiskWriter`) flushes state updates to disk asynchronously without stalling GPU computations. A full write queue never drops batches: `put_update` waits up to `put_timeout` seconds, then writes the batch synchronously.
* **Read Consistency:** The writer tracks in-flight rows; `step()` drains the queue before reading rows the writer has not finished writing, so no step ever computes from stale state.
* **Per-Row Adam Bias Correction:** Each row carries its own step counter (`<name>_t.dat`), so the first update of a freshly touched row is a correct Adam step regardless of the global step count.
* **Explicit State Lifecycle:** State directories are guarded by `<name>_meta.json` (shape, dtype, format version). `resume=True` continues existing state; `resume=False` (default) fully reinitializes with a warning; shape/dtype mismatches, partial or truncated state files raise explicit errors instead of being silently reused.
* **Safe Duplicate Aggregation:** Automatically aggregates duplicate node/entity indices (hub nodes) appearing within the same batch via `torch.unique` and `index_add_`.
* **Sorted Sequential I/O:** Batch indices are sorted after deduplication, so reads and writes hit the memmaps sequentially.
* **Riemannian & Euclidean Support:** Supports Riemannian gradient scaling and Poincaré Ball retraction ($k > 0$) for Hyperbolic embeddings, as well as standard Euclidean Adam ($k = 0.0$). Rows read from disk are projected into the ball, so corrupted files cannot break the geometry.
* **Honest Shutdown & Checkpoints:** `shutdown()` drains the queue and raises `RuntimeError` if data could not be fully written; `save_checkpoint`/`load_checkpoint` provide a consistent on-disk checkpoint API (`state_dict()` intentionally raises `NotImplementedError`).
* **Cross-Platform:** Runs seamlessly on local Linux/Windows GPU machines, workstations, and cloud notebooks.

---

## 📊 Performance Notes

There is no fixed throughput number: DSA trades compute for memory, and its
speed is bounded by your disk — expect optimizer-state I/O to be the
bottleneck on large tables. The correctness guarantees added in 0.2.0
(blocking writes instead of dropped batches, in-flight row tracking, a fourth
per-row counter file, sorted indices) add roughly ~10% step-time overhead
compared to 0.1.0. Measure on your own hardware before drawing conclusions.

What stays true at any scale:

| Property | Since |
| :--- | :--- |
| Optimizer states reside on disk; VRAM footprint scales with batch size $\mathcal{O}(B)$ | 0.1.0 |
| Duplicate node aggregation within a batch | 0.1.0 |
| No silently dropped update batches | 0.2.0 |
| Step reads never race the background writer | 0.2.0 |
| Correct bias correction for sparsely updated rows | 0.2.0 |

---

## 📦 Installation

### Online Installation
```bash
pip install git+https://github.com/Assistentus/DSA.git
```

### Editable Local Mode
```bash
git clone https://github.com/Assistentus/DSA.git
cd DSA
pip install -e .
```

---

## 💻 1. Local Machine / Workstation Example

This example demonstrates the exact PyTorch Autograd workflow for local GPU/CPU execution:

```python
import os
import shutil
import torch
from dsa.optimizer import DiskSparseRiemannianAdam

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "./dsa_local_cache"

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# 2. Define Dataset & Parameters (e.g., 100,000 entities)
num_entities = 100_000
dim = 64
batch_size = 1024

# Initial weights in RAM
initial_weights = torch.randn(num_entities, dim) * 0.1
params = {"entity_emb": initial_weights}

# 3. Initialize Optimizer (k=0.0 for standard Euclidean Adam)
optimizer = DiskSparseRiemannianAdam(
    params=params,
    lr=0.01,
    k=0.0,
    disk_dir=cache_dir
)

# Target vectors to converge towards
target_embeddings = torch.randn(num_entities, dim) * 0.1

print("🚀 Starting local training loop...")

# 4. Training Loop
for epoch in range(1, 101):
    indices = torch.randint(0, num_entities, (batch_size,))
    idx_np = indices.numpy()
    
    # Read current weights directly from DSA disk memmap
    # (drain the write queue first so the read cannot race a pending write)
    optimizer.writer.wait_all_flushed()
    vec_np = optimizer.state_files["entity_emb"]["w"][idx_np].copy()
    vec = torch.from_numpy(vec_np).to(device).requires_grad_(True)
    
    # Compute Loss (MSE against target)
    target_vec = target_embeddings[indices].to(device)
    loss = torch.mean((vec - target_vec) ** 2)
    
    # Compute gradients via PyTorch autograd
    loss.backward()
    
    # Pass explicit indices and gradients to DSA
    optimizer.step(updates={"entity_emb": (indices, vec.grad.cpu())})

# 5. Flush background writer thread to disk
optimizer.shutdown()
print(f"✅ Training complete. Final Loss: {loss.item():.4f}")

# Clean up local cache
shutil.rmtree(cache_dir)
```

---

## 🏆 2. Kaggle & Cloud Notebooks Example
[![Kaggle Dataset Banner](https://img.shields.io/badge/DiskSparseAdam-Kaggle_Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/assistentus/disk-sparse-adam)

In a **Kaggle Notebook**, VRAM and RAM are strictly capped (~16 GB VRAM, ~30 GB RAM). `DiskSparseAdam` offloads optimizer states onto Kaggle's fast `/kaggle/working` NVMe disk, completely bypassing VRAM limitations.

### Installation in Kaggle Notebooks
```python
# Cell 1: Install directly from GitHub
!pip install -q git+https://github.com/Assistentus/DSA.git
```

### Full Kaggle Notebook Integration Code

👉 **Interactive Kaggle Demo:** [DiskSparseAdam Kaggle Notebook](https://www.kaggle.com/code/assistentus/disksparseadam-dsa)

```python
import os
import shutil
import torch
from dsa.optimizer import DiskSparseRiemannianAdam

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure Kaggle Working Disk Directory
KAGGLE_CACHE_DIR = "/kaggle/working/dsa_optimizer_cache"
os.makedirs(KAGGLE_CACHE_DIR, exist_ok=True)

num_entities = 1_000_000
dim = 128
batch_size = 2048

# Initialize embeddings
embeddings = torch.randn(num_entities, dim) * 0.01

# 3. Initialize DiskSparseRiemannianAdam
optimizer = DiskSparseRiemannianAdam(
    params={"entity_emb": embeddings},
    lr=0.001,
    k=0.0,
    disk_dir=KAGGLE_CACHE_DIR,
    max_queue_size=300
)

print("🚀 Starting training on Kaggle...")

# 4. Training Loop
for epoch in range(1, 11):
    indices = torch.randint(0, num_entities, (batch_size,))
    idx_np = indices.numpy()
    
    # Fetch weights from DSA disk cache
    # (drain the write queue first so the read cannot race a pending write)
    optimizer.writer.wait_all_flushed()
    vec_np = optimizer.state_files["entity_emb"]["w"][idx_np].copy()
    vec = torch.from_numpy(vec_np).to(device).requires_grad_(True)
    
    # Compute Loss
    loss = (vec ** 2).sum()
    loss.backward()
    
    # Apply step on disk
    optimizer.step(updates={"entity_emb": (indices, vec.grad.cpu())})

# 5. Shutdown & Clean disk
optimizer.shutdown()

if os.path.exists(KAGGLE_CACHE_DIR):
    shutil.rmtree(KAGGLE_CACHE_DIR)
    print("🧹 Disk cache cleaned.")
```

---

## 🌌 3. Hyperbolic Poincaré Ball Workflow ($k > 0$)

For Hyperbolic Knowledge Graph Embeddings (e.g., Poincaré embeddings, RotatE in hyperbolic space), set the curvature parameter `k > 0.0`:

```python
# Set k=1.0 for Poincaré Ball Conformal Gradient Scaling & Retraction
optimizer = DiskSparseRiemannianAdam(
    params={"hyperbolic_nodes": initial_weights},
    lr=0.001,
    k=1.0,  # Hyperbolic mode
    disk_dir="./hyperbolic_cache"
)

# The optimizer automatically applies Riemannian scaling and retraction during step()
optimizer.step(updates={"hyperbolic_nodes": (indices, gradients)})
```

---

## 📖 API Reference

### `DiskSparseRiemannianAdam`

```python
from dsa.optimizer import DiskSparseRiemannianAdam

optimizer = DiskSparseRiemannianAdam(
    params,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    k=1.0,
    disk_dir="./disk_cache",
    max_queue_size=150,
    resume=False,
    put_timeout=5.0
)
```

#### Parameters:
* **`params`** (*dict* or *list*): Dictionary of `{"param_name": torch.Tensor}` or a list of tensors to optimize.
* **`lr`** (*float*, optional): Learning rate (default: `1e-3`).
* **`beta1`, `beta2`** (*float*, optional): Coefficients for computing running averages of the gradient and its square (default: `0.9`, `0.999`).
* **`eps`** (*float*, optional): Term added to the denominator for numerical stability (default: `1e-8`).
* **`weight_decay`** (*float*, optional): Weight decay coefficient (default: `0.0`).
* **`k`** (*float*, optional): Poincaré Ball curvature parameter ($k > 0$ for Hyperbolic, $k = 0$ for Euclidean) (default: `1.0`).
* **`disk_dir`** (*str*, optional): Directory path where `np.memmap` files (`_w.dat`, `_m.dat`, `_v.dat`, `_t.dat`) and the `<name>_meta.json` descriptors will be created (default: `"./disk_cache"`).
* **`max_queue_size`** (*int*, optional): Maximum capacity of the background I/O queue (default: `150`).
* **`resume`** (*bool*, optional): `False` (default) — if state files already exist in `disk_dir`, they are fully reinitialized (weights from `params`, zeroed moments and counters) with a warning; `True` — continue from the existing state as is. Shape/dtype mismatch raises `RuntimeError` either way.
* **`put_timeout`** (*float*, optional): Seconds to wait for queue space before a write batch falls back to a synchronous write (no data is ever dropped; default: `5.0`).

#### Methods:
* **`step(updates, current_k=None)`**: Performs a single optimization step. 
  * `updates` (*dict*): Dictionary mapping `param_name` to a tuple `(indices_tensor, grads_tensor)`.
  * `current_k` (*float* or *Tensor*, optional): Dynamic curvature $k$ override.
* **`shutdown()`**: Drains pending writes, flushes all state files and terminates the background writer thread. Raises `RuntimeError` if the data could not be fully written. Must be called at the end of the training process.
* **`save_checkpoint(path)`**: Writes a consistent checkpoint: drains the queue, flushes `w/m/v/t` and atomically stores a manifest with file paths and shapes.
* **`load_checkpoint(path)`**: Verifies the manifest (shapes, paths, file sizes) and reopens the memmaps. Checkpoint files must live in the optimizer's `disk_dir`.
* **`state_dict()` / `load_state_dict()`**: Raise `NotImplementedError` — DSA state lives on disk and cannot be serialized through these methods.

---

## ⬆️ Upgrading from 0.1.0

0.2.0 keeps the public API (constructor, `step(updates, current_k)`, `shutdown()`, `state_files`) compatible; the new constructor arguments (`resume`, `put_timeout`) are optional. Behavior changes to be aware of:

* **`resume=False` is the default.** Re-running with default arguments fully reinitializes the state in `disk_dir` (weights from `params`, zeroed moments and per-row counters) and logs a warning. In 0.1.0, existing `_w/_m/_v.dat` files were silently reused. Pass `resume=True` to continue a previous run.
* **Four state files per parameter** instead of three: a per-row step counter `<name>_t.dat` (int32) was added, plus a `<name>_meta.json` descriptor. Old 0.1.0 state directories are not compatible and must be removed (or reopened with the default `resume=False`, which reinitializes them).
* **No dropped batches.** A full write queue blocks (up to `put_timeout` seconds) and then writes synchronously instead of discarding the batch.
* **`shutdown()` can raise.** If pending data cannot be fully written, you get a `RuntimeError` with details instead of a false success message.
* **`state_dict()`/`load_state_dict()` raise `NotImplementedError`.** Use `save_checkpoint()`/`load_checkpoint()` or reopen the directory with `resume=True`.
* **Bias correction is per-row.** Rows that are updated rarely (or for the first time late in training) now get correct Adam bias correction; early-training dynamics differ from 0.1.0, which scaled by the global step counter.

---

## 🛠️ Best Practices & Troubleshooting

1. **Always call `optimizer.shutdown()`**: `shutdown()` drains the write queue and flushes all memory-mapped buffers. Since 0.2.0 it raises `RuntimeError` instead of silently pretending everything was saved.
2. **Disk Space Limit**: Ensure your NVMe drive has enough space for 4 arrays per parameter (`_w.dat`, `_m.dat`, `_v.dat` float32 and `_t.dat` int32) plus a small `<name>_meta.json`. For 1M entities with 128 dimensions, this takes ~1.5 GB + 4 MB.
3. **Queue Overflow Warning**: A full queue no longer drops batches: the write waits `put_timeout` seconds and then completes synchronously (a warning and the `writer._sync_fallbacks` counter are recorded). If this happens regularly, your disk is the bottleneck: increase `max_queue_size` or reduce the step rate.
4. **Reopening a state directory** is controlled by `resume`: leftover files are never silently reused — you either continue explicitly (`resume=True`), reinitialize with a warning (`resume=False`, the default), or get a clear error (shape mismatch, partial or truncated files, missing meta).

---

## 📄 License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2024 Maksim Khotinsky

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Citations

If you use this software in your research, please cite it using the following DOI:

**APA:**
> Khotinsky, M. (2026). DiskSparseAdam (DSA): Out-of-Core Sparse Adam Optimizer. Zenodo. https://doi.org/10.5281/zenodo.21296366

**BibTeX:**
```bibtex
@software{khotinsky_dsa,
  author       = {Maksim Khotinsky},
  title        = {DiskSparseAdam (DSA): Out-of-Core Sparse Adam Optimizer},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21296366},
  url          = {https://doi.org/10.5281/zenodo.21296366}
}
```
