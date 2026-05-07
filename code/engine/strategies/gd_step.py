"""
Per-iteration step strategies for `GradientDescent`.

The optimizer keeps state (population, current_grad, used_subgradient,
projection/step-size strategies); the step strategy decides what one
outer iteration *means* — a vanilla gradient step, or a full two-level
ravine extrapolation.

History tagging
---------------
Strategies may append intermediate frames to ``optimizer.results.history``
mid-step (e.g. ``RavineStep`` records every inner descent point). Tagged
with ``"phase": "descend" | "extrapolate"`` so a future UI can distinguish
ravine sub-events from normal iterations. Existing consumers can ignore
the tag — the ``"population"`` key is unchanged.

``VanillaGradientStep`` adds no extra frames: history schema for plain GD
is identical to before this refactor.
"""
from abc import ABC, abstractmethod
import numpy as np

from engine.strategies.step_size import StepSizeStrategy


class GDStepStrategy(ABC):
    @abstractmethod
    def step(self, optimizer) -> np.ndarray:
        """Compute one outer iteration; return the new (1, D) population."""


class VanillaGradientStep(GDStepStrategy):
    """One projected gradient step. The default for plain GD."""
    def step(self, optimizer) -> np.ndarray:
        x = optimizer.population[0]
        grad, is_subgrad = optimizer.target.evaluate_gradient(x)
        optimizer.used_subgradient = is_subgrad
        optimizer.current_grad = grad
        direction = -grad
        alpha = optimizer.step_size_strategy.compute_alpha(
            x, grad, direction, optimizer.target, optimizer.projection_strategy
        )
        new_x = optimizer.projection_strategy.project(x + alpha * direction)
        return np.atleast_2d(new_x)


class OneStepRavineStep(GDStepStrategy):
    """
    Classical alternating ravine method.

    Bootstrap (same as ``RavineStep``): from start v⁰ produce
    ``x₀ = GD(v⁰)`` and ``x₁ = GD(v¹)`` where ``v¹ = v⁰ + ravine_shift``.

    Subsequent outer iterations alternate one ravine extrapolation with
    one gradient step, taken between the two **most recent iterates**
    (not converged bottoms):

    * ``x₂ = RS(x₀, x₁)``  — ravine extrapolation
    * ``x₃ = GD(x₂)``      — gradient step
    * ``x₄ = RS(x₂, x₃)``
    * ``x₅ = GD(x₄)``
    * …

    The ravine extrapolation magnitude is delegated to
    ``ravine_step_size_strategy`` so backtracking / fixed-step apply the
    same way they do for the inner gradient step.
    """
    def __init__(self, inner_strategy: GDStepStrategy,
                 ravine_step_size_strategy: StepSizeStrategy,
                 ravine_shift: float = 0.5):
        self.inner_strategy = inner_strategy
        self.ravine_step_size_strategy = ravine_step_size_strategy
        self.ravine_shift = ravine_shift
        self._prev_point = None
        self._is_ravine_next = True
        self._bootstrap_done = False

    def step(self, optimizer) -> np.ndarray:
        if not self._bootstrap_done:
            v0 = optimizer.population[0]
            x0 = self._gd_from(optimizer, v0)
            optimizer.results.history.append({
                "population": np.atleast_2d(x0).copy(),
                "phase": "descend",
                "subgrad": optimizer.used_subgradient,
            })
            v1 = optimizer.projection_strategy.project(
                v0 + np.ones_like(v0) * self.ravine_shift
            )
            optimizer.results.history.append({
                "population": np.atleast_2d(v1).copy(),
                "phase": "extrapolate",
                "subgrad": False,
            })
            x1 = self._gd_from(optimizer, v1)
            optimizer.results.history.append({
                "population": np.atleast_2d(x1).copy(),
                "phase": "descend",
                "subgrad": optimizer.used_subgradient,
            })
            self._prev_point = x0
            self._bootstrap_done = True
            self._is_ravine_next = True
            return np.atleast_2d(x1)

        x_curr = optimizer.population[0]
        x_prev = self._prev_point

        if self._is_ravine_next:
            v = _ravine_extrapolate(
                optimizer, x_curr, x_prev, self.ravine_step_size_strategy
            )
            v = optimizer.projection_strategy.project(v)
            new_pop = np.atleast_2d(v)
            optimizer.results.history.append({
                "population": new_pop.copy(),
                "phase": "extrapolate",
                "subgrad": False,
            })
            self._prev_point = x_curr
            self._is_ravine_next = False
            return new_pop

        new_pop = self.inner_strategy.step(optimizer)
        optimizer.results.history.append({
            "population": new_pop.copy(),
            "phase": "descend",
            "subgrad": optimizer.used_subgradient,
        })
        self._prev_point = x_curr
        self._is_ravine_next = True
        return new_pop

    def _gd_from(self, optimizer, start_x):
        optimizer.population = np.atleast_2d(start_x)
        return self.inner_strategy.step(optimizer)[0]


def _ravine_extrapolate(optimizer, x_curr, x_prev, step_size_strategy):
    """
    Compute the ravine extrapolation point from two recent bottoms.

    Direction: unit vector along (x_curr − x_prev), flipped by the sign of
    f(x_curr) − f(x_prev) so we always step "downhill" relative to the
    pair. Magnitude: delegated to ``step_size_strategy`` so backtracking /
    exact line search apply to the ravine step the same way they apply to
    a gradient step.
    """
    diff = x_curr - x_prev
    norm = np.linalg.norm(diff)
    if norm == 0:
        return x_curr.copy()
    f_curr = optimizer.target.evaluate(x_curr)
    f_prev = optimizer.target.evaluate(x_prev)
    sign_f = np.sign(f_curr - f_prev)
    direction = -(diff / norm) * sign_f
    grad, _ = optimizer.target.evaluate_gradient(x_curr)
    alpha = step_size_strategy.compute_alpha(
        x_curr, grad, direction, optimizer.target, optimizer.projection_strategy
    )
    return x_curr + alpha * direction


class RavineStep(GDStepStrategy):
    """
    Two-level ravine method (textbook spec — methodichka §1.1, OMnOD).

    Outer iteration = ravine extrapolation between the two most recent
    converged bottoms. Inner iteration = full gradient descent to the next
    bottom (multiple gradient steps, not a single one).

    Bootstrap (first outer call): descend from the start position to the
    first bottom x⁰, shift by ``ravine_shift`` to obtain v¹, then descend
    again to x¹. Subsequent outer calls extrapolate from
    (x_prev, x_curr) and descend from the extrapolation point.

    In practice the inner descents tend to dominate the runtime and
    rarely outperform plain GD with a good step-size strategy. Kept as
    a faithful implementation of the textbook formulation; needs parameter tweaking.
    """
    def __init__(self, inner_strategy: GDStepStrategy,
                 ravine_step_size_strategy: StepSizeStrategy,
                 ravine_shift: float = 0.5,
                 inner_tol: float | None = 1e-4,
                 inner_max_iter: int = 100):
        self.inner_strategy = inner_strategy
        self.ravine_step_size_strategy = ravine_step_size_strategy
        self.ravine_shift = ravine_shift
        self.inner_tol = inner_tol
        self.inner_max_iter = inner_max_iter
        self._prev_bottom = None
        self._initialized = False

    def step(self, optimizer) -> np.ndarray:
        if not self._initialized:
            v0 = optimizer.population[0]
            x0 = self._descend_to_bottom(optimizer, v0)
            v1 = optimizer.projection_strategy.project(
                v0 + np.ones_like(v0) * self.ravine_shift
            )
            self._record_extrapolate(optimizer, v1)
            x1 = self._descend_to_bottom(optimizer, v1)
            self._prev_bottom = x0
            self._initialized = True
            return np.atleast_2d(x1)

        x_curr = optimizer.population[0]
        x_prev = self._prev_bottom

        v = _ravine_extrapolate(
            optimizer, x_curr, x_prev, self.ravine_step_size_strategy
        )
        v = optimizer.projection_strategy.project(v)
        self._record_extrapolate(optimizer, v)

        x_new = self._descend_to_bottom(optimizer, v)
        self._prev_bottom = x_curr
        return np.atleast_2d(x_new)

    def _descend_to_bottom(self, optimizer, start_x):
        """Run inner gradient descent until convergence; record each step in history.

        ``inner_tol=None`` disables the per-step tolerance check — the inner
        descent then runs exactly ``inner_max_iter`` steps regardless of
        movement size.
        """
        x = start_x.copy()
        for _ in range(self.inner_max_iter):
            optimizer.population = np.atleast_2d(x)
            new_pop = self.inner_strategy.step(optimizer)
            new_x = new_pop[0]
            optimizer.results.history.append({
                "population": np.atleast_2d(new_x).copy(),
                "phase": "descend",
                "subgrad": optimizer.used_subgradient,
            })
            if self.inner_tol is not None and np.linalg.norm(new_x - x) < self.inner_tol:
                x = new_x
                break
            x = new_x
        return x

    def _record_extrapolate(self, optimizer, x):
        optimizer.results.history.append({
            "population": np.atleast_2d(x).copy(),
            "phase": "extrapolate",
            "subgrad": False,
        })
