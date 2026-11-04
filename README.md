
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

* **Near-Zero VRAM Overhead:** Optimizer momentum ($m$), variance ($v$), weights ($w$) and per-row step counters ($t$) reside on NVMe/SSD storage via `np.memmap`. Only active batch rows are loaded into RAM/VRAM. By default the optimizer **does not retain the source tensors** after initialization (`retain_params=False`): they are replaced by zero-length CPU placeholders, so a full embedding table cannot stay alive on the GPU through `param_groups`.
* **True Riemannian Adam ($k > 0$):** The Euclidean gradient is converted to a Riemannian one, Adam's first moment lives in the tangent space and is parallel-transported after every step, the step is an **exponential map** (not a Euclidean subtraction followed by a clip), and the second moment is carried as a non-negative diagonal scale. The collection of $O(B)$ rows is computed in `float64`, which keeps the Möbius/gyration formulas stable near the boundary; storage remains `float32`. `k = 0` is standard Euclidean Adam.
* **Lossless Async I/O:** A background worker thread with double-buffering (`AsyncDiskWriter`) flushes state updates to disk asynchronously without stalling GPU computations. A full write queue never drops batches: `put_update` waits up to `put_timeout` seconds, then writes the batch synchronously.
* **Read Consistency:** The writer tracks in-flight rows; `step()` drains the queue before reading rows the writer has not finished writing, so no step ever computes from stale state.
* **Per-Row Adam Bias Correction:** Each row carries its own step counter (`<name>_t.dat`), so the first update of a freshly touched row is a correct Adam step regardless of the global step count.
* **Explicit State Lifecycle:** State directories are guarded by `<name>_meta.json` (shape, dtype, format version, curvature, algorithm) and a global `dsa_global_meta.json` (global step, curvature). `resume=None` (default) refuses to guess when a directory already exists: pass `resume=True` to continue or `resume=False` to reinitialize. Shape/dtype mismatches, partial or truncated state files raise explicit errors. 0.1.0/0.2.0 directories are migrated on `resume=True` (weights preserved; incompatible moments and counters reset).
* **Ball Invariants:** Fresh hyperbolic weights are globally scaled to at most `init_max_norm_fraction` (default 0.5) of the ball radius, so `torch.randn` initialization does not start on the boundary. Rows read from disk are projected, resumed tables are revalidated, and `set_curvature()` reprojects the whole table when the ball changes.
* **Safe Duplicate Aggregation & Sorted I/O:** Duplicate node/entity indices are summed; batch indices are sorted after deduplication so reads and writes hit the memmaps sequentially.
* **Explicit Weight Decay Semantics:** `weight_decay_mode="l2"` (default) adds `wd * w` to the Euclidean gradient and is therefore the Riemannian gradient of $f + \tfrac{wd}{2}\|w\|^2$; in the Euclidean case it is classic Adam L2. `"decoupled"` adds `wd * w` to the tangent-space descent direction and reproduces AdamW exactly when `k = 0`.
* **Portable Checkpoints:** `export_checkpoint(dir)` writes a self-contained directory (`checkpoint.json` + relative data files + SHA-256 checksums); `import_checkpoint(dir)` verifies and restores it, including the global step and curvature. Checkpoints can be moved between machines. `save_checkpoint`/`load_checkpoint` remain as directory-only aliases.
* **Honest Shutdown:** `shutdown()` is a retryable state machine (`RUNNING → DRAINING → STOPPED`, or `FAILED`); it never reports success while the worker is alive or the queue is unfinished, and a failed write is re-raised instead of being hidden.
* **Validated Inputs:** Negative/out-of-range indices, wrong gradient shapes, non-finite gradients, NaN/negative moment rows, `k = NaN/inf/negative`, and unknown update keys all have explicit, test-covered behavior.
* **Cross-Platform:** Runs on local Linux/Windows GPU machines, workstations, and cloud notebooks.

---

## 📊 Performance Notes

There is no fixed throughput number: DSA trades compute for memory, and its
speed is bounded by your disk — expect optimizer-state I/O to be the
bottleneck on large tables. The 0.3.0 correctness work (blocking writes,
in-flight row tracking, a fourth per-row counter file, sorted indices,
float64 batch math) adds step-time overhead compared to 0.1.0; the float64
part only touches the active batch, so on large tables it is usually small
relative to disk I/O. Measure on your own hardware before drawing conclusions.

What stays true at any scale:

| Property | Since |
| :--- | :--- |
| Optimizer states reside on disk; VRAM footprint scales with batch size $\mathcal{O}(B)$ | 0.1.0 |
| Source tensors are not retained by the optimizer | 0.3.0 |
| Duplicate node aggregation within a batch | 0.1.0 |
| No silently dropped update batches | 0.2.0 |
| Step reads never race the background writer | 0.2.0 |
| Correct bias correction for sparsely updated rows | 0.2.0 |
| Full Riemannian Adam (exp map + parallel transport) | 0.3.0 |
| Portable checkpoints with checksums | 0.3.0 |
| Retryable, honest shutdown | 0.3.0 |

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

    # Public read API: waits for in-flight writes and projects into the ball.
    vec = optimizer.read_rows(indices).to(device).requires_grad_(True)

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

    # Public read API (in-flight writes are drained, rows are projected).
    vec = optimizer.read_rows(indices).to(device).requires_grad_(True)

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

For hyperbolic knowledge-graph embeddings, set `k > 0` (the ball has curvature
$-k$ and radius $1/\sqrt{k}$):

```python
from dsa.optimizer import DiskSparseRiemannianAdam

optimizer = DiskSparseRiemannianAdam(
    params={"hyperbolic_nodes": initial_weights},
    lr=1e-3,
    k=1.0,                    # hyperbolic mode
    disk_dir="./hyperbolic_cache",
    resume=False,             # explicit: new run, reinitialize this directory
    weight_decay=0.0,
    weight_decay_mode="l2",   # classic Adam L2 in the Riemannian gradient
)

for indices, gradients in batches:
    optimizer.step(updates={"hyperbolic_nodes": (indices, gradients)})

# Read embeddings for inference: projected into the ball, safe public API.
emb = optimizer.read_rows(indices, name="hyperbolic_nodes")

# Curvature annealing: the ball changes, so every row is reprojected.
optimizer.set_curvature(0.5)

# Portable, checksummed checkpoint (a directory, not a JSON file).
optimizer.export_checkpoint("./checkpoints/run-001")
optimizer.shutdown()
```

What `step()` does when `k > 0`:

1. Reads only the batch rows `(w, m, v, t)` from the memmaps.
2. Converts the Euclidean gradient to the Riemannian one,
   $\operatorname{grad}_R f = \lambda(w)^{-2} g_E$, with
   $\lambda(w) = 2 / (1 - k\|w\|^2)$.
3. Updates Adam's moments: $m$ is a tangent vector, $v$ is a non-negative
   diagonal scale. The active rows are computed in `float64` for numerical
   stability near the boundary.
4. Takes a geodesic step with the exponential map
   $w \leftarrow \exp_w(-\eta\, \hat m / (\sqrt{\hat v}+\epsilon))$.
5. Transports $m$ (and the square root of $v$) from $T_w$ to $T_{w_{\text{new}}}$
   by parallel transport, so the moments follow the new tangent space.
6. Projects the result into the ball as a numerical guard and queues the
   consistent row update.

Notes:

* `init_max_norm_fraction=0.5` (default) globally scales fresh hyperbolic
  weights so that no row starts closer than half the radius to the boundary.
  Pass `init_max_norm_fraction=1.0` if you want only strict projection.
* `resume=True` restores the saved curvature; a differing constructor `k` is
  ignored with a warning. Use `set_curvature()` to change it deliberately.
* `current_k=...` on `step()` is a shortcut for `set_curvature()`; it
  reprojects the whole table because the ball itself changes.
* `weight_decay_mode="decoupled"` applies decay as a tangent-space term
  (AdamW semantics when `k = 0`). `"l2"` is part of the gradient and is
  therefore scaled by the Poincaré metric.

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
    weight_decay_mode="l2",      # "l2" | "decoupled"
    k=1.0,
    disk_dir="./disk_cache",
    max_queue_size=150,
    resume=None,                 # None | True | False
    put_timeout=5.0,
    retain_params=False,
    init_max_norm_fraction=0.5,
)
```

#### Parameters:
* **`params`** (*dict* or *list*): `{name: tensor}` or a list of 2-D tensors. Values are read once to initialize the on-disk state.
* **`lr`** (*float*): Learning rate (default `1e-3`).
* **`beta1`, `beta2`** (*float*): Adam moment coefficients, each in `[0, 1)` (defaults `0.9`, `0.999`).
* **`eps`** (*float*): Denominator floor, must be `> 0` (default `1e-8`).
* **`weight_decay`** (*float*): Non-negative decay coefficient (default `0.0`).
* **`weight_decay_mode`** (*str*): `"l2"` (default) puts `wd * w` into the Euclidean gradient before the Riemannian conversion; `"decoupled"` adds `wd * w` to the tangent descent direction (AdamW semantics when `k = 0`).
* **`k`** (*float*): Curvature magnitude. `k > 0` is the Poincaré ball of curvature `-k`; `k = 0` is Euclidean Adam. Must be finite and non-negative; NaN/inf/negative raise immediately.
* **`disk_dir`** (*str*): Directory for `<name>_w/_m/_v/_t.dat`, `<name>_meta.json`, and `dsa_global_meta.json`.
* **`max_queue_size`** (*int*): Background write-queue capacity (default `150`).
* **`resume`** (*None | bool*): `None` (default) raises if `disk_dir` already contains state, forcing an explicit choice; `True` continues it; `False` reinitializes from `params` with a warning. Legacy 0.1.0/0.2.0 state is migrated on `resume=True`.
* **`put_timeout`** (*float*): Seconds to wait for queue space before a synchronous write (default `5.0`; writes are never dropped).
* **`retain_params`** (*bool*): If `False` (default), the source tensors are replaced by zero-length CPU placeholders after initialization, so the optimizer cannot retain an $O(N)$ table in VRAM. Set `True` only if other code inspects `param_groups[...]["params"]`.
* **`init_max_norm_fraction`** (*float*, `(0, 1]`): For `k > 0`, fresh weights whose largest row norm exceeds this fraction of the ball radius are globally scaled down (default `0.5`). `1.0` means strict projection only.

#### Methods:
* **`step(updates, current_k=None)`**: One optimizer step.
  * `updates` (*dict*): `{param_name: (indices, grads)}`. Indices must be 1-D integer tensors/arrays in `[0, N)`; gradients must have shape `(len(indices), dim)` and be finite. Duplicate indices are summed.
  * `current_k` (*float* or *Tensor*, optional): Change curvature for this and subsequent steps; the whole table is reprojected.
* **`read_rows(indices, name=None)`**: Public read API for inference; waits for in-flight writes, validates indices, rejects corrupted rows, and projects into the current ball.
* **`set_curvature(k, reset_moments=False)`**: Change curvature and reproject every row. Moments are kept by default (documented annealing approximation); `reset_moments=True` gives a mathematically clean restart of the adaptive state.
* **`reproject_all(name=None)`**: Scan the whole table, project finite rows into the current ball, and fully reset non-finite/negative-variance rows. Returns the number of projected rows.
* **`export_checkpoint(directory)`**: Write a portable checkpoint directory (`checkpoint.json`, relative data files, SHA-256 checksums) and return the manifest path.
* **`import_checkpoint(directory, verify_hashes=True)`**: Verify and restore a portable checkpoint, including global step and curvature.
* **`save_checkpoint(directory)` / `load_checkpoint(directory)`**: Directory-only aliases for the two methods above. Passing a `.json` path raises a clear migration error.
* **`shutdown()`**: Drain the queue, flush state, and persist the global step. Retryable after a transient failure; raises while data is unfinished or a write failed. Call it at the end of training.
* **`state_dict()` / `load_state_dict()`**: Raise `NotImplementedError` — use the checkpoint API or reopen the directory with `resume=True`.

State files are an implementation detail; use `read_rows()` and checkpoints instead of reading `state_files` directly.

---

## ⬆️ Upgrading from 0.2.0 / 0.1.0

0.3.0 is a correctness release with several intentional breaking changes:

* **`resume` is now `None` by default.** An existing state directory is never reused or silently reinitialized: pass `resume=True` to continue or `resume=False` to start over.
* **Checkpoints are directories.** `save_checkpoint`/`load_checkpoint` accept a directory; `export_checkpoint`/`import_checkpoint` are the primary names. 0.2.0 manifests with absolute paths are rejected — create a new checkpoint.
* **State format v2.** 0.2.0 (`format_version=1`) and 0.1.0 (no meta, no `t`) directories are migrated on `resume=True`: weights are preserved, but moments and per-row counters are reset because 0.3.0 stores tangent-space moments and transports them. Re-running with `resume=False` reinitializes from `params` (this also repairs a partial 0.1.0 directory).
* **The Riemannian step changed.** The old code was Adam in ambient coordinates with gradient scaling and a clip. 0.3.0 uses a Riemannian gradient, tangent-space moments, an exponential map and parallel transport. Hyperbolic training dynamics will differ from 0.2.0 even with the same hyperparameters.
* **`weight_decay` semantics changed for `k > 0`.** The default `"l2"` is now part of the Riemannian gradient instead of a coordinate shrink; use `"decoupled"` if you need AdamW-style behavior.
* **Source tensors are released.** `param_groups[...]["params"]` contains zero-length placeholders by default. Use `read_rows()` for weights. Pass `retain_params=True` for the old behavior.
* **Global step is persisted.** `shutdown()` and checkpoints store it; reopening restores `_global_step` (bias correction remains per-row).
* **Validation is strict.** Negative/out-of-range indices, wrong gradient shapes, non-finite gradients, invalid curvature, and unknown update keys now fail or warn explicitly instead of being ignored.

---

## 🛠️ Best Practices & Troubleshooting

1. **Always call `optimizer.shutdown()`**: it drains the write queue, flushes all memmaps, persists the global step and stops the worker. It raises `RuntimeError` instead of pretending that unfinished data was saved; a second call retries after a transient failure.
2. **Disk space.** Four arrays per parameter (`_w/_m/_v.dat` float32, `_t.dat` int32) plus two small JSON descriptors. For 1M entities with 128 dimensions this is ~1.5 GB + 4 MB.
3. **Hyperbolic initialization.** Keep initial norms small. The default `init_max_norm_fraction=0.5` rescales a too-large table globally; a warning tells you when it happens. Starting at the boundary makes $\lambda^2$ huge and learning extremely slow.
4. **Queue overflow.** A full queue waits `put_timeout` seconds and then writes synchronously (warning + `writer._sync_fallbacks`). If this happens regularly, your disk is the bottleneck: increase `max_queue_size` or reduce the step rate.
5. **Reopening a state directory.** Use `resume=True` to continue, or `resume=False` to reinitialize. Shape mismatch, partial/truncated files, a missing meta file, a leftover reinitialization marker, or a curvature mismatch produce explicit errors.
6. **Checkpoints.** Use a dedicated directory per checkpoint. Import verifies sizes and SHA-256 checksums; corrupted or foreign checkpoints are rejected before the live state is touched.
7. **Boundary behavior.** Points are kept strictly inside the ball (`BALL_EPS = 1e-5`). Very long training with a large `lr` can push embeddings to that margin; the optimizer stays finite and projects, but the chart is ill-conditioned there. Prefer a smaller `lr` and early stopping rather than training on the boundary.

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
