"""End-to-end optimization sanity: DSA must actually minimize on the ball.

The objective is the geodesic distance to fixed target points; Euclidean
gradients come from torch.autograd through the (verified) distance formula,
which is exactly the input contract of ``step()``.  These tests catch a
class of errors the equality oracles cannot: an update that is internally
consistent but does not descend.
"""

import math

import numpy as np
import torch

from dsa import manifold
from dsa.optimizer import DiskSparseRiemannianAdam


def _make(w0, lr, k, tmp_path):
    return DiskSparseRiemannianAdam(
        {"e": torch.from_numpy(w0.astype(np.float32))}, lr=lr, k=k,
        disk_dir=str(tmp_path), resume=False, init_max_norm_fraction=1.0)


def test_converges_to_targets_on_the_ball(tmp_path):
    rng = np.random.default_rng(3)
    n, d, k, lr = 5, 8, 1.0, 0.05
    w0 = rng.normal(0, 0.2, (n, d))
    targets = rng.normal(0, 0.3, (n, d))
    targets = (torch.from_numpy(targets)
               * (0.5 * math.sqrt(1.0 / k)
                  / torch.from_numpy(np.linalg.norm(targets, axis=1,
                                                    keepdims=True))))

    opt = _make(w0, lr, k, tmp_path)
    idx = torch.arange(n)
    tgt = targets.clone().requires_grad_(False)

    def cur_distances():
        w = opt.read_rows(idx).double()
        return manifold.distance(w, tgt.double(), k).squeeze(-1)

    d0 = cur_distances()
    for _ in range(300):
        w = opt.read_rows(idx).double().requires_grad_(True)
        loss = manifold.distance(w, tgt.double(), k).pow(2).sum()
        loss.backward()
        g = w.grad.detach().float()
        if not torch.isfinite(g).all():
            raise AssertionError("градиент расстояления стал не-конечным")
        opt.step(updates={"e": (idx, g)})
    opt.shutdown()
    d1 = cur_distances()

    assert d0.mean().item() > 1.0, "старт должен быть далеко от целей"
    assert d1.mean().item() < 0.05, (
        f"оптимизатор не сошёлся: средняя дистанция {d0.mean():.3f} -> "
        f"{d1.mean():.3f} за 300 шагов")
    assert bool((d1 < d0).all()), "какая-то строка отдалилась от цели"


def test_boundary_stress_stays_finite_and_inside(tmp_path):
    """Строки вплотную к границе + большие градиенты: конечно, внутри шара,
    без «перелёта» на противоположную сторону (метрическая нормировка
    ограничивает риманову длину первого шага lr*sqrt(d))."""
    d, k, lr = 6, 1.0, 0.05
    radius = math.sqrt((1 - manifold.BALL_EPS) / k)
    rng = np.random.default_rng(5)
    w0 = rng.normal(0, 1, (4, d))
    w0 = (w0 / np.linalg.norm(w0, axis=1, keepdims=True)) * (0.9999 * radius)

    opt = _make(w0.astype(np.float32), lr, k, tmp_path)
    idx = torch.arange(4)
    for s in range(60):
        g = torch.from_numpy(
            (rng.normal(0, 50.0, (4, d))).astype(np.float32))
        opt.step(updates={"e": (idx, g)})
    opt.shutdown()

    w = np.array(opt.state_files["e"]["w"])
    assert np.isfinite(w).all(), "веса стали не-конечными у границы"
    norms = np.linalg.norm(w, axis=1)
    assert np.all(norms <= radius + 1e-5), norms
    # не улетели к противоположной границе: средняя норма не выродилась
    # к радиусу со всех сторон — строки должны оставаться в разумном диапазоне
    assert np.all(norms > 0.05), (
        "строки схлопнулись в начало координата под стрессом")
