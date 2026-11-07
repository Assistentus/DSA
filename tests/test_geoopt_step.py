"""External oracle: the hyperbolic optimizer vs geoopt's RiemannianAdam.

Requires the real ``geoopt`` package (skipped otherwise; in CI only the
in-repo oracles run).  Install geoopt into a separate directory and run:

    PYTHONPATH=/path/to/geoopt_pkg python -m pytest tests/test_geoopt_step.py

DSA matches ``geoopt.optim.RiemannianAdam`` on ``PoincareBallExact`` (the
exact-exponential-map variant; the default ``PoincareBall`` uses a first-order
retraction, which DSA deliberately does not): scalar second moment
``lambda^2 ||g_riem||^2`` broadcast over coordinates, first moment updated in
the tangent space and parallel-transported after the exponential-map step,
second moment not transported.  The test pins the FULL multi-step trajectory
(weights and both moments), not just one step.
"""

import numpy as np
import pytest
import torch

geoopt = pytest.importorskip("geoopt")

from dsa.optimizer import DiskSparseRiemannianAdam  # noqa: E402


@pytest.mark.parametrize("k", [0.7, 1.0])
@pytest.mark.parametrize("wd", [0.0, 0.02])
def test_trajectory_matches_geoopt_exact(tmp_path, k, wd):
    # Строки на умеренных нормах (0.6 радиуса): здесь совпадение обязано быть
    # на уровне округления float32. Приочковые строки НЕ сравниваются с geoopt
    # напрямую: geoopt считает целиком в float32 и у границы теряет точность
    # катастрофически (лямбда ~1e4); поведение DSA там закреплено отдельными
    # инвариантами (ограниченная длина шага) и стресс-тестом.
    rng = np.random.default_rng(7)
    n, d, lr, steps = 4, 5, 0.03, 10
    w0 = rng.normal(0, 0.3, (n, d)).astype(np.float32)
    w0 /= (np.linalg.norm(w0, axis=1, keepdims=True) / 0.6)

    opt = DiskSparseRiemannianAdam(
        {"e": torch.from_numpy(w0.copy())}, lr=lr, k=k, disk_dir=str(tmp_path),
        resume=False, init_max_norm_fraction=1.0, weight_decay=wd,
        weight_decay_mode="l2")
    ball = geoopt.PoincareBallExact(c=k)
    p = geoopt.ManifoldParameter(torch.from_numpy(w0.copy()), manifold=ball)
    gopt = geoopt.optim.RiemannianAdam([p], lr=lr, weight_decay=wd)

    for _ in range(steps):
        g = rng.normal(0, 0.7, (n, d)).astype(np.float32)
        opt.step(updates={"e": (torch.arange(n), torch.from_numpy(g.copy()))})
        p.grad = torch.from_numpy(g.copy())
        gopt.step()
    opt.shutdown()

    st = gopt.state[p]
    assert np.allclose(np.array(opt.state_files["e"]["w"]), p.data.numpy(),
                       atol=1e-5), "траектория w разошлась с geoopt"
    assert np.allclose(np.array(opt.state_files["e"]["m"]),
                       st["exp_avg"].numpy(), atol=1e-5)
    assert np.allclose(np.array(opt.state_files["e"]["v"]),
                       st["exp_avg_sq"].numpy(), atol=1e-5)
