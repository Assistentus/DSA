"""Geometric properties of the Poincare-ball implementation.

These tests do not compare two copies of the same formula: they check the
defining properties of the exponential map, parallel transport and the
Riemannian gradient.  The same properties are checked for the independent
float64 reference used in ``test_fixes.py``.
"""

import math

import numpy as np
import pytest
import torch

from dsa import manifold
from tests import reference


K = 0.7
D = 6


def _inside(rows, radius_fraction=0.8):
    radius = math.sqrt(1.0 / K)
    x = np.asarray(rows, dtype=np.float64)
    scale = min(1.0, radius_fraction * radius / np.linalg.norm(x, axis=-1).max())
    return x * scale


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(0)
    x = _inside(rng.normal(0, 0.3, (5, D)))
    y = _inside(rng.normal(0, 0.3, (5, D)))
    u = rng.normal(0, 0.2, (5, D))
    v = rng.normal(0, 0.5, (5, D))
    return x, y, u, v


def _torch(a):
    return torch.tensor(a, dtype=torch.float64)


def test_expmap_zero_is_identity(points):
    x, _, _, _ = points
    out = manifold.expmap(_torch(x), torch.zeros_like(_torch(x)), K).numpy()
    assert np.allclose(out, x, atol=1e-14)


def test_expmap_stays_inside_ball(points):
    x, _, u, _ = points
    out = manifold.expmap(_torch(x), _torch(u), K).numpy()
    radius = math.sqrt((1.0 - manifold.BALL_EPS) / K)
    assert np.all(np.linalg.norm(out, axis=-1) <= radius + 1e-12)


def test_expmap_distance_equals_tangent_norm(points):
    """dist(x, exp_x(u)) == lambda(x) * ||u|| for small u."""
    x, _, u, _ = points
    out = manifold.expmap(_torch(x), _torch(u), K)
    lhs = manifold.distance(_torch(x), out, K).numpy().ravel()
    rhs = (manifold.lambda_x(_torch(x), K)
           * torch.linalg.norm(_torch(u), dim=-1, keepdim=True)).numpy().ravel()
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_log_exp_roundtrip(points):
    x, _, u, _ = points
    y = manifold.expmap(_torch(x), _torch(u), K)
    back = reference.logmap(x, y.numpy(), K)
    assert np.allclose(back, u, atol=1e-12)


def test_parallel_transport_identity(points):
    x, _, _, v = points
    out = manifold.parallel_transport(_torch(x), _torch(x), _torch(v), K).numpy()
    assert np.allclose(out, v, atol=1e-14)


def test_parallel_transport_is_isometry(points):
    x, y, _, v = points
    out = manifold.parallel_transport(_torch(x), _torch(y), _torch(v), K)
    lhs = (manifold.lambda_x(_torch(y), K)
           * torch.linalg.norm(out, dim=-1, keepdim=True)).numpy().ravel()
    rhs = (manifold.lambda_x(_torch(x), K)
           * torch.linalg.norm(_torch(v), dim=-1, keepdim=True)).numpy().ravel()
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_parallel_transport_roundtrip(points):
    x, y, _, v = points
    forward = manifold.parallel_transport(_torch(x), _torch(y), _torch(v), K)
    back = manifold.parallel_transport(_torch(y), _torch(x), forward, K).numpy()
    assert np.allclose(back, v, atol=1e-12)


def test_parallel_transport_linear(points):
    x, y, _, v = points
    a = 0.37
    lhs = manifold.parallel_transport(_torch(x), _torch(y), _torch(a * v), K).numpy()
    rhs = a * manifold.parallel_transport(_torch(x), _torch(y), _torch(v), K).numpy()
    assert np.allclose(lhs, rhs, atol=1e-13)


def test_riemannian_gradient_is_inverse_metric(points):
    x, _, _, _ = points
    g = _torch(np.ones_like(x))
    rg = manifold.riemannian_gradient(g, _torch(x), K)
    lam = manifold.lambda_x(_torch(x), K)
    assert np.allclose(rg.numpy(), (g / lam ** 2).numpy(), atol=1e-14)


def test_projection_and_distance_validity(points):
    x, y, _, _ = points
    radius = math.sqrt((1.0 - manifold.BALL_EPS) / K)
    outside = torch.full((2, D), 10.0)
    projected = manifold.project_to_ball(outside, K)
    assert torch.all(torch.linalg.norm(projected, dim=-1) <= radius + 1e-12)
    assert np.isfinite(manifold.distance(
        projected[:1], projected[1:], K).numpy()).all()
    # an out-of-ball point yields NaN, not a silently wrong distance
    assert np.isnan(manifold.distance(
        outside[:1], _torch(y[:1]), K).numpy()).all()


def test_reference_satisfies_same_properties(points):
    x, y, u, v = points
    e = reference.expmap(x, u, K)
    assert np.allclose(reference.expmap(x, np.zeros_like(x), K), x, atol=1e-14)
    lhs = reference.distance(x, e, K).ravel()
    rhs = reference._lambda(x, K).ravel() * np.linalg.norm(u, axis=-1)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
    assert np.allclose(reference.logmap(x, e, K), u, atol=1e-12)

    pt = reference.parallel_transport(x, y, v, K)
    assert np.allclose(
        reference._lambda(y, K).ravel() * np.linalg.norm(pt, axis=-1),
        reference._lambda(x, K).ravel() * np.linalg.norm(v, axis=-1),
        rtol=1e-12, atol=1e-12)
    assert np.allclose(reference.parallel_transport(y, x, pt, K), v, atol=1e-12)


def test_euclidean_branches_are_identity(points):
    """k=0 must reduce every geometry helper to its Euclidean counterpart."""
    x, y, u, v = points
    tx, ty, tu, tv = _torch(x), _torch(y), _torch(u), _torch(v)
    assert torch.allclose(manifold.project_to_ball(tx, 0.0), tx)
    assert np.allclose(manifold.project_to_ball_np(x, 0.0), x)
    assert torch.allclose(manifold.mobius_add(tx, ty, 0.0), tx + ty)
    assert torch.allclose(manifold.gyration(tx, ty, tv, 0.0), tv)
    assert torch.allclose(manifold.expmap(tx, tu, 0.0), tx + tu)
    assert torch.allclose(manifold.parallel_transport(tx, ty, tv, 0.0), tv)
    assert torch.allclose(manifold.riemannian_gradient(tv, tx, 0.0), tv)
    assert np.allclose(manifold.distance(tx, ty, 0.0).numpy(),
                       np.linalg.norm(x - y, axis=-1))


def test_is_in_ball_mask():
    inside = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    outside = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    nan = torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)
    assert manifold.is_in_ball(inside, 1.0).tolist() == [True, True]
    assert manifold.is_in_ball(outside, 1.0).tolist() == [False]
    assert manifold.is_in_ball(nan, 1.0).tolist() == [False]
    assert manifold.is_in_ball(outside, 0.0).tolist() == [True]
    assert manifold.is_in_ball(nan, 0.0).tolist() == [False]

