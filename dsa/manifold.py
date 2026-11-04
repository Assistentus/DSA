"""Poincare ball geometry for curvature -c (c > 0).

Conventions
-----------
The ball is ``B_c = {x : ||x|| < 1/sqrt(c)}`` with the conformal metric

    g_x = lambda(x)^2 I,    lambda(x) = 2 / (1 - c ||x||^2).

For a Euclidean gradient ``g_E`` the Riemannian gradient is

    grad_R f(x) = lambda(x)^-2 g_E = (1 - c ||x||^2)^2 / 4 * g_E.

The exponential map and the parallel transport follow the standard
Poincare-ball formulas (the same ones used by geoopt with ``k = -c``).
All functions operate on the last dimension and support batched rows.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Relative distance from the boundary kept by the projection.  A point at
# radius sqrt((1 - EPS) / c) has conformal factor ~2/EPS, which is already
# numerically extreme; keeping a small margin avoids infinities.
BALL_EPS = 1e-5

# tanh saturates to exactly 1.0 in float32 for arguments above ~9, which puts
# the second Moebius operand on the boundary and makes the addition singular.
# Keeping it strictly inside is the numerical equivalent of "finite u".
EXP_TANH_MAX = 1.0 - 1e-7


def sq_norm(x: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean norm over the last dimension, shape (..., 1)."""
    return (x * x).sum(dim=-1, keepdim=True)


def lambda_x(x: torch.Tensor, c: float) -> torch.Tensor:
    """Conformal factor ``2 / (1 - c ||x||^2)``."""
    return 2.0 / (1.0 - c * sq_norm(x)).clamp_min(1e-15)


def project_to_ball(x: torch.Tensor, c: float, eps: float = BALL_EPS) -> torch.Tensor:
    """Project rows into the open ball of radius ``sqrt((1-eps)/c)``.

    Uses a multiplicative scale instead of ``x / ||x||`` so that the zero
    row is handled without NaNs.
    """
    if c <= 0.0:
        return x
    max_radius = math.sqrt((1.0 - eps) / c)
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(
        norm > max_radius,
        max_radius / norm.clamp_min(1e-30),
        torch.ones_like(norm),
    )
    return x * scale


def project_to_ball_np(w: np.ndarray, c: float, eps: float = BALL_EPS) -> np.ndarray:
    """NumPy version of :func:`project_to_ball` (used for memmap chunks).

    Norms are computed in float64 to avoid overflow on corrupted float32
    data; the result is returned in the input dtype.
    """
    if c <= 0.0:
        return w
    max_radius = math.sqrt((1.0 - eps) / c)
    norm = np.linalg.norm(w.astype(np.float64), axis=-1)
    scale = np.where(norm > max_radius, max_radius / np.maximum(norm, 1e-30), 1.0)
    return (w.astype(np.float64) * scale[..., None]).astype(w.dtype, copy=False)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Moebius addition on the ball of curvature -c."""
    if c <= 0.0:
        return x + y
    x2 = sq_norm(x)
    y2 = sq_norm(y)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    den = (1.0 + 2.0 * c * xy + c * c * x2 * y2).clamp_min(1e-15)
    return num / den


def gyration(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, c: float) -> torch.Tensor:
    """Gyration ``gyr[u, v] w`` for curvature -c (geoopt formula)."""
    if c <= 0.0:
        return w
    u2 = sq_norm(u)
    v2 = sq_norm(v)
    uv = (u * v).sum(dim=-1, keepdim=True)
    uw = (u * w).sum(dim=-1, keepdim=True)
    vw = (v * w).sum(dim=-1, keepdim=True)
    c2 = c * c
    a = -c2 * uw * v2 + c * vw + 2.0 * c2 * uv * vw
    b = -c2 * vw * u2 - c * uw
    den = (1.0 + 2.0 * c * uv + c2 * u2 * v2).clamp_min(1e-15)
    return w + 2.0 * (a * u + b * v) / den


def expmap(x: torch.Tensor, u: torch.Tensor, c: float) -> torch.Tensor:
    """Exponential map ``exp_x(u)`` on the ball of curvature -c.

    ``u`` is a tangent vector in coordinates at ``x``; its Riemannian norm
    is ``lambda(x) ||u||``, which is why ``lambda(x)`` appears below.
    """
    if c <= 0.0:
        return x + u
    u_norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    lx = lambda_x(x, c)
    sqrt_c = math.sqrt(c)
    scale = torch.tanh(sqrt_c * lx * u_norm / 2.0).clamp(max=EXP_TANH_MAX) / (
        sqrt_c * u_norm)
    return mobius_add(x, scale * u, c)


def parallel_transport(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, c: float
) -> torch.Tensor:
    """Parallel transport of ``u`` from ``T_x`` to ``T_y`` along the geodesic.

    Uses the gyration form ``gyr[y, -x] u * lambda(x) / lambda(y)``, which is
    an isometry between the two tangent spaces (verified numerically).
    """
    if c <= 0.0:
        return u
    return gyration(y, -x, u, c) * lambda_x(x, c) / lambda_x(y, c)


def riemannian_gradient(g: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    """Convert a Euclidean gradient into a Riemannian one: ``lambda(x)^-2 g``."""
    if c <= 0.0:
        return g
    return g * (1.0 - c * sq_norm(x)).clamp_min(1e-15).pow(2) / 4.0


def distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Geodesic distance on the ball of curvature -c (for diagnostics/tests).

    Rows outside the ball yield NaN instead of a silently wrong value.
    """
    if c <= 0.0:
        return (x - y).norm(dim=-1)
    x2 = sq_norm(x)
    y2 = sq_norm(y)
    diff2 = sq_norm(x - y)
    den = (1.0 - c * x2) * (1.0 - c * y2)
    arg = 1.0 + 2.0 * c * diff2 / den.clamp_min(1e-15)
    nan = torch.full_like(arg, float("nan"))
    arg = torch.where(den > 0.0, arg.clamp_min(1.0), nan)
    return torch.acosh(arg) / math.sqrt(c)


def is_in_ball(x: torch.Tensor, c: float, eps: float = BALL_EPS) -> torch.Tensor:
    """Boolean mask: rows strictly inside the ball (with margin ``eps``)."""
    if c <= 0.0:
        return torch.isfinite(x).all(dim=-1)
    max_radius = math.sqrt((1.0 - eps) / c)
    return torch.isfinite(x).all(dim=-1) & (x.norm(dim=-1) <= max_radius)
