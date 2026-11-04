"""Golden values produced by geoopt's stereographic Poincare math.

The reference numbers below were generated with the actual
``geoopt/manifolds/stereographic/math.py`` (master) using its convention
``k = -c`` (geoopt's signed curvature) against the DSA convention ``c > 0``.
They are hard-coded so the repository does not need geoopt installed and so a
future change in ``dsa.manifold`` is checked against an external oracle, not
only against another formula written in this repository.
"""

import torch

from dsa import manifold


C = 0.7
X = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
Y = torch.tensor([[-0.3, 0.4]], dtype=torch.float64)
U = torch.tensor([[0.5, -0.2]], dtype=torch.float64)


def test_lambda_matches_geoopt():
    assert torch.allclose(manifold.lambda_x(X, C),
                          torch.tensor([[2.072538860103627]],
                                       dtype=torch.float64),
                          atol=1e-14)


def test_mobius_add_matches_geoopt():
    expected = torch.tensor([[-0.09525184009236541, 0.3261653918314331]],
                            dtype=torch.float64)
    assert torch.allclose(manifold.mobius_add(X, Y, C), expected, atol=1e-14)


def test_expmap_matches_geoopt():
    expected = torch.tensor([[0.6304294392917015, -0.27530547641281544]],
                            dtype=torch.float64)
    assert torch.allclose(manifold.expmap(X, U, C), expected, atol=1e-12)


def test_parallel_transport_matches_geoopt():
    expected = torch.tensor([[0.41537343956347617, -0.19855381136144154]],
                            dtype=torch.float64)
    assert torch.allclose(
        manifold.parallel_transport(X, Y, U, C), expected, atol=1e-14)


def test_distance_matches_geoopt():
    assert torch.allclose(manifold.distance(X, Y, C),
                          torch.tensor([[1.487174390441505]],
                                       dtype=torch.float64),
                          atol=1e-14)
