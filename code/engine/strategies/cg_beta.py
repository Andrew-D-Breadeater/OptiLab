"""
β formulas for the Conjugate Gradient method, as Strategy objects.

All three implementations coincide on a strict quadratic — Lemma 2 of the
course notes (gradients become mutually orthogonal at each CG step, so the
cross term ``⟨∇f_k, ∇f_{k-1}⟩`` vanishes and FR/PR/HS reduce to the same
ratio). On non-quadratics they diverge; Polak–Ribière is empirically the
most robust and is the recommended default for the lab.

There is intentionally no ``PureBeta`` (using the Hessian ``A`` directly):
on a quadratic the matrix-free trick ``A h_{k-1} = (∇f_k − ∇f_{k-1})/α_{k-1}``
collapses it into Hestenes–Stiefel; on non-quadratics ``A`` varies between
steps and the formula is ambiguous. See ``Reports/OMnOD/lab4`` for the
theory note.
"""
from abc import ABC, abstractmethod
import numpy as np


_EPS = 1e-12


class BetaStrategy(ABC):
    @abstractmethod
    def compute(self, grad_k: np.ndarray, grad_prev: np.ndarray,
                direction_prev: np.ndarray) -> float:
        """Return β_{k-1}. Implementations must guard against zero denominators."""


class FletcherReeves(BetaStrategy):
    """β = ‖∇f(x_k)‖² / ‖∇f(x_{k-1})‖²."""
    def compute(self, grad_k, grad_prev, direction_prev) -> float:
        denom = float(np.dot(grad_prev, grad_prev))
        if denom < _EPS:
            return 0.0
        return float(np.dot(grad_k, grad_k)) / denom


class PolakRibiere(BetaStrategy):
    """β = (∇f(x_k), ∇f(x_k) − ∇f(x_{k-1})) / ‖∇f(x_{k-1})‖²."""
    def compute(self, grad_k, grad_prev, direction_prev) -> float:
        denom = float(np.dot(grad_prev, grad_prev))
        if denom < _EPS:
            return 0.0
        return float(np.dot(grad_k, grad_k - grad_prev)) / denom


class HestenesStiefel(BetaStrategy):
    """β = (∇f(x_k), ∇f(x_k) − ∇f(x_{k-1})) / (h_{k-1}, ∇f(x_k) − ∇f(x_{k-1}))."""
    def compute(self, grad_k, grad_prev, direction_prev) -> float:
        y = grad_k - grad_prev
        denom = float(np.dot(direction_prev, y))
        if abs(denom) < _EPS:
            return 0.0
        return float(np.dot(grad_k, y)) / denom
