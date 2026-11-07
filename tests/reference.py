"""Independent float64 reference for Riemannian Adam on the Poincare ball.

This module deliberately does not import :mod:`dsa.manifold`.  It implements
the standard Poincare-ball gyrovector formulas (the same conventions as
geoopt with ``k_geoopt = -k``) and the optimizer variant used by DSA:
per-coordinate moments (geoopt style, unlike the paper's scalar second
moment), first-moment parallel transport (the paper's ``tau`` update) and a
positivity-preserving sqrt-transport of the second moment (the paper needs
no transport for its scalar ``v``; geoopt does not transport ``v`` at all).
The tests first verify that this reference satisfies the defining geometric
properties, then compare the optimizer against it, so the optimizer is not
merely checked against a copy of itself.
"""

from __future__ import annotations

import numpy as np


def _sq(x):
    return np.sum(x * x, axis=-1, keepdims=True)


def _lambda(x, k):
    return 2.0 / (1.0 - k * _sq(x))


def _mobius_add(x, y, k):
    x2, y2 = _sq(x), _sq(y)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    num = (1.0 + 2.0 * k * xy + k * y2) * x + (1.0 - k * x2) * y
    den = 1.0 + 2.0 * k * xy + k * k * x2 * y2
    return num / den


def _gyration(u, v, w, k):
    u2, v2 = _sq(u), _sq(v)
    uv = np.sum(u * v, axis=-1, keepdims=True)
    uw = np.sum(u * w, axis=-1, keepdims=True)
    vw = np.sum(v * w, axis=-1, keepdims=True)
    k2 = k * k
    a = -k2 * uw * v2 + k * vw + 2.0 * k2 * uv * vw
    b = -k2 * vw * u2 - k * uw
    den = 1.0 + 2.0 * k * uv + k2 * u2 * v2
    return w + 2.0 * (a * u + b * v) / den


def expmap(x, u, k):
    if k <= 0.0:
        return x + u
    un = np.linalg.norm(u, axis=-1, keepdims=True)
    un = np.maximum(un, 1e-300)
    scale = np.minimum(np.tanh(np.sqrt(k) * _lambda(x, k) * un / 2.0),
                       1.0 - 1e-7) / (np.sqrt(k) * un)
    return _mobius_add(x, scale * u, k)


def logmap(x, y, k):
    if k <= 0.0:
        return y - x
    v = _mobius_add(-x, y, k)
    vn = np.linalg.norm(v, axis=-1, keepdims=True)
    vn = np.maximum(vn, 1e-300)
    return (2.0 / (np.sqrt(k) * _lambda(x, k))) * np.arctanh(np.sqrt(k) * vn) * v / vn


def parallel_transport(x, y, u, k):
    if k <= 0.0:
        return u
    return _gyration(y, -x, u, k) * _lambda(x, k) / _lambda(y, k)


def distance(x, y, k):
    if k <= 0.0:
        return np.linalg.norm(x - y, axis=-1)
    x2, y2 = _sq(x), _sq(y)
    diff2 = _sq(x - y)
    den = (1.0 - k * x2) * (1.0 - k * y2)
    arg = 1.0 + 2.0 * k * diff2 / den
    return np.arccosh(np.maximum(arg, 1.0)) / np.sqrt(k)


class RiemannianAdamReference:
    """In-memory float64 tangent-space Adam with parallel transport."""

    def __init__(self, w0, lr, beta1=0.9, beta2=0.999, eps=1e-8,
                 k=0.0, weight_decay=0.0, weight_decay_mode="l2"):
        self.w = np.array(w0, dtype=np.float64)
        self.n, self.d = self.w.shape
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.t = np.zeros(self.n, dtype=np.int64)
        self.lr = float(lr)
        self.b1, self.b2, self.eps = float(beta1), float(beta2), float(eps)
        self.k = float(k)
        self.wd = float(weight_decay)
        self.wd_mode = weight_decay_mode

    def step(self, indices, grads):
        idx = np.asarray(indices, dtype=np.int64)
        g = np.asarray(grads, dtype=np.float64)
        unique, inverse = np.unique(idx, return_inverse=True)
        if len(unique) != len(idx):
            agg = np.zeros((len(unique), g.shape[1]))
            np.add.at(agg, inverse, g)
            idx, g = unique, agg
        order = np.argsort(idx)
        idx, g = idx[order], g[order]

        x = self.w[idx]
        if self.wd > 0.0 and self.wd_mode == "l2":
            g = g + self.wd * x
        if self.k > 0.0:
            g = g * (1.0 - self.k * _sq(x)) ** 2 / 4.0
            # the ball is one manifold component: scalar squared Riemannian
            # norm ||g||_x^2 = lambda^2 ||g_riem||^2, broadcast (the paper's
            # Algorithm 1; geoopt's component_inner with keepdim)
            g_sq = np.broadcast_to(
                (_lambda(x, self.k) ** 2 * _sq(g)).reshape(len(x), 1),
                g.shape,
            )
        else:
            g_sq = g * g  # R^d is a product of d components: per-coordinate

        self.t[idx] += 1
        tt = self.t[idx].astype(np.float64)[:, None]
        m = self.b1 * self.m[idx] + (1.0 - self.b1) * g
        v = self.b2 * self.v[idx] + (1.0 - self.b2) * g_sq
        direction = (m / (1.0 - self.b1 ** tt)) / (
            np.sqrt(v / (1.0 - self.b2 ** tt)) + self.eps)
        if self.wd > 0.0 and self.wd_mode == "decoupled":
            direction = direction + self.wd * x

        if self.k > 0.0:
            x_new = expmap(x, -self.lr * direction, self.k)
            m = parallel_transport(x, x_new, m, self.k)
            # the scalar second moment needs no transport
        else:
            x_new = x - self.lr * direction

        self.w[idx] = x_new
        self.m[idx] = m
        self.v[idx] = v
