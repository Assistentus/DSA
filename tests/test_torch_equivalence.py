"""External oracle: the Euclidean path (k=0) must equal torch.optim exactly.

DSA at ``k = 0`` treats ``R^d`` as the product of ``d`` one-dimensional
manifold components, so the moments stay per-coordinate and the update is
classic Adam.  This file pins the whole multi-step trajectory (dense batches,
random gradients, weight decay) against ``torch.optim.Adam`` ("l2") and
``torch.optim.AdamW`` ("decoupled") — an oracle independent of this
repository.
"""

import numpy as np
import torch

from dsa.optimizer import DiskSparseRiemannianAdam


def _run_pair(tmp_path, mode, torch_cls, steps=12):
    rng = np.random.default_rng(11)
    n, d, lr, wd = 20, 6, 0.05, 0.01
    w0 = rng.normal(0, 0.4, (n, d))

    opt = DiskSparseRiemannianAdam(
        {"e": torch.from_numpy(w0.astype(np.float32))}, lr=lr, k=0.0,
        disk_dir=str(tmp_path), resume=False, init_max_norm_fraction=1.0,
        weight_decay=wd, weight_decay_mode=mode)
    ref_param = torch.from_numpy(w0.astype(np.float32)).requires_grad_(True)
    ref = torch_cls([ref_param], lr=lr, weight_decay=wd)

    idx = torch.arange(n)
    for _ in range(steps):
        g = torch.from_numpy(rng.normal(0, 0.8, (n, d)).astype(np.float32))
        opt.step(updates={"e": (idx, g)})
        ref_param.grad = g.clone()
        ref.step()
    opt.shutdown()
    return np.array(opt.state_files["e"]["w"]), ref_param.detach().numpy()


def test_euclidean_matches_torch_adam_l2(tmp_path):
    w_dsa, w_torch = _run_pair(tmp_path, "l2", torch.optim.Adam)
    assert np.allclose(w_dsa, w_torch, atol=2e-6), \
        np.abs(w_dsa - w_torch).max()


def test_euclidean_matches_torch_adamw_decoupled(tmp_path):
    w_dsa, w_torch = _run_pair(tmp_path, "decoupled", torch.optim.AdamW)
    assert np.allclose(w_dsa, w_torch, atol=2e-6), \
        np.abs(w_dsa - w_torch).max()
