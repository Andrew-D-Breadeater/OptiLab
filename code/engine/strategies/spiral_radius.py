"""
Radius (shrinkage) strategies for SpiralOptimizer.

Independent of the rotation strategy in :mod:`engine.strategies.spiral_rotation`.

* ``is_static``     — the strategy returns the same ``r`` on every call.
* ``is_per_point``  — ``get_r`` returns a ``(N_pop,)`` array (one r per point)
  instead of a scalar. Used by the adaptive variant (lecture §19).
"""
from abc import ABC, abstractmethod
import numpy as np


class RadiusStrategy(ABC):
    is_static: bool = False
    is_per_point: bool = False

    @abstractmethod
    def get_r(self, step_idx: int, fitnesses: np.ndarray | None):
        """Return scalar ``r`` (or ``(N_pop,)`` if ``is_per_point``)."""


class FixedRadius(RadiusStrategy):
    """Plain user-supplied r. Base case (lecture §13–§16)."""
    is_static = True
    is_per_point = False

    def __init__(self, r: float):
        self.r = float(r)

    def get_r(self, step_idx, fitnesses):
        return self.r


class PrecisionBasedRadius(RadiusStrategy):
    """
    Lecture §17. ``r = delta ** (1/k_max)`` — pick ``r`` from desired
    final precision and iteration budget instead of guessing.
    """
    is_static = True
    is_per_point = False

    def __init__(self, delta: float, k_max: int):
        if not (0 < delta < 1):
            raise ValueError(f"delta must lie in (0, 1), got {delta}")
        if k_max <= 0:
            raise ValueError(f"k_max must be positive, got {k_max}")
        self.delta = float(delta)
        self.k_max = int(k_max)
        self.r = self.delta ** (1.0 / self.k_max)

    def get_r(self, step_idx, fitnesses):
        return self.r


class StochasticRadius(RadiusStrategy):
    """Lecture §18. ``r ~ U(r_l, r_u)`` resampled every step."""
    is_static = False
    is_per_point = False

    def __init__(self, r_l: float, r_u: float):
        if not (0 < r_l < r_u < 1):
            raise ValueError(f"Need 0 < r_l < r_u < 1, got r_l={r_l}, r_u={r_u}")
        self.r_l = float(r_l)
        self.r_u = float(r_u)

    def get_r(self, step_idx, fitnesses):
        return float(np.random.uniform(self.r_l, self.r_u))


class AdaptiveRadius(RadiusStrategy):
    """
    Lecture §19. Per-point radius from fitness:

        r(i) = r_u + (r_l - r_u) / (1 + c1 / (f_i - min f))

    Best point → r_u (slow, refines its area). Worst point → r_l (fast pull-in).
    """
    is_static = False
    is_per_point = True

    def __init__(self, r_l: float, r_u: float, c1: float):
        if not (0 < r_l < r_u < 1):
            raise ValueError(f"Need 0 < r_l < r_u < 1, got r_l={r_l}, r_u={r_u}")
        if c1 <= 0:
            raise ValueError(f"c1 must be positive, got {c1}")
        self.r_l = float(r_l)
        self.r_u = float(r_u)
        self.c1 = float(c1)

    def get_r(self, step_idx, fitnesses):
        f = np.asarray(fitnesses, dtype=float)
        gap = f - np.min(f)
        # gap == 0 at the best ⇒ ratio = inf ⇒ r = r_u (the intended limit).
        # gap == +inf at OOB ⇒ ratio = 0 ⇒ r = r_l (also fine — OOB pulled in hard).
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(gap > 0, self.c1 / gap, np.inf)
        return self.r_u + (self.r_l - self.r_u) / (1.0 + ratio)
