"""
Rotation matrix strategies for SpiralOptimizer.

Decouples the choice of rotation matrix (a function of N and step index)
from the radius and the optimizer loop. Any rotation strategy pairs with
any radius strategy from :mod:`engine.strategies.spiral_radius`.

Two flavours of the textbook (Tamura & Yasuda 2011/2016) construction:

* Generic: build the N-dim rotation as the product of all N(N-1)/2 planar
  rotations (lecture §16). ``FixedAngleRotation`` and ``StochasticRotation``
  use this path.
* Special-case: when ``theta = (-1)^N * pi/2`` (lecture §17) the product
  collapses to a circulant sign-swap matrix. ``PeriodicDescentRotation``
  builds that matrix directly — cheaper and exact.
"""
from abc import ABC, abstractmethod
import numpy as np


def _build_rotation_matrix(thetas: np.ndarray, N: int) -> np.ndarray:
    """
    Product ``∏ R_{i,j}(θ_{i,j})`` (lecture §16).

    ``thetas`` is a 1D array of length ``N*(N-1)/2``, indexed by pair (i, j)
    with ``i < j`` in the order ``(0,1), (0,2), ..., (0,N-1), (1,2), ...``.
    The order follows the textbook double product; what matters is that the
    same order is used consistently across calls so trajectories stay
    reproducible.
    """
    expected = N * (N - 1) // 2
    if thetas.shape != (expected,):
        raise ValueError(
            f"Expected {expected} angles for N={N}, got shape {thetas.shape}"
        )

    R = np.eye(N)
    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            theta = thetas[k]
            c, s = np.cos(theta), np.sin(theta)
            R_ij = np.eye(N)
            R_ij[i, i] = c
            R_ij[j, j] = c
            R_ij[i, j] = -s
            R_ij[j, i] = s
            R = R_ij @ R
            k += 1
    return R


class RotationMatrixStrategy(ABC):
    is_static: bool = False

    @abstractmethod
    def get_matrix(self, N: int, step_idx: int) -> np.ndarray:
        """Return the (N, N) rotation matrix used at ``step_idx``."""


class FixedAngleRotation(RotationMatrixStrategy):
    """Same θ on every plane pair, every step. Base case (lecture §13–§16)."""
    is_static = True

    def __init__(self, theta: float):
        self.theta = float(theta)
        self._cached_matrix: np.ndarray | None = None
        self._cached_N: int | None = None

    def get_matrix(self, N: int, step_idx: int) -> np.ndarray:
        if self._cached_matrix is not None and self._cached_N == N:
            return self._cached_matrix
        thetas = np.full(N * (N - 1) // 2, self.theta)
        R = _build_rotation_matrix(thetas, N)
        self._cached_matrix = R
        self._cached_N = N
        return R


class PeriodicDescentRotation(RotationMatrixStrategy):
    """
    Lecture §17. With ``θ = (-1)^N · π/2`` the §16 rotation product collapses
    to a circulant sign-swap matrix — built directly rather than via the
    plane-product helper.

    Concretely, applied to a vector ``(d_1, ..., d_N)`` it produces
    ``(-d_N, d_1, d_2, ..., d_{N-1})``: cyclic shift right + sign flip on
    the first coordinate.
    """
    is_static = True

    def __init__(self):
        self._cached_matrix: np.ndarray | None = None
        self._cached_N: int | None = None

    def get_matrix(self, N: int, step_idx: int) -> np.ndarray:
        if self._cached_matrix is not None and self._cached_N == N:
            return self._cached_matrix
        R = np.zeros((N, N))
        for i in range(1, N):
            R[i, i - 1] = 1.0
        R[0, N - 1] = -1.0
        self._cached_matrix = R
        self._cached_N = N
        return R


class StochasticRotation(RotationMatrixStrategy):
    """
    Lecture §18. θ is drawn fresh each step from ``N(theta, sigma)`` (same
    sampled value broadcast across all plane pairs).
    """
    is_static = False

    def __init__(self, theta: float, sigma: float):
        self.theta = float(theta)
        self.sigma = float(sigma)

    def get_matrix(self, N: int, step_idx: int) -> np.ndarray:
        sampled = self.theta + np.random.normal(0.0, self.sigma)
        thetas = np.full(N * (N - 1) // 2, sampled)
        return _build_rotation_matrix(thetas, N)
