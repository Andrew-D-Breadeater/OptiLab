"""
Step-size selection as Strategy objects.

Replaces the old `learning_rate` / `use_line_search` / `use_exact_line_search`
flags on `TraditionalOptimizer`. Each strategy owns its parameters (learning
rate, decay, backtracking constants) and knows how to produce a step length
α given the current state.

All strategies are projection-aware: they evaluate `f(project(x + α·d))`,
so feasibility is preserved during the step-size search.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize_scalar


class StepSizeStrategy(ABC):
    @abstractmethod
    def compute_alpha(self, x, grad, direction, target, projection) -> float:
        ...


class FixedStepSize(StepSizeStrategy):
    """Constant learning rate, optionally multiplied by `decay_rate` after each call."""
    def __init__(self, learning_rate: float, decay_rate: float = 1.0):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def compute_alpha(self, x, grad, direction, target, projection) -> float:
        alpha = self.learning_rate
        self.learning_rate *= self.decay_rate
        return alpha


class BacktrackingLineSearch(StepSizeStrategy):
    """Armijo backtracking. Halves α until the sufficient-decrease condition holds."""
    def __init__(self, c: float = 1e-4, beta: float = 0.5, alpha0: float = 1.0,
                 alpha_floor: float = 1e-12):
        self.c = c
        self.beta = beta
        self.alpha0 = alpha0
        self.alpha_floor = alpha_floor

    def compute_alpha(self, x, grad, direction, target, projection) -> float:
        alpha = self.alpha0
        f_x = target.evaluate(x)

        while True:
            x_next = projection.project(x + alpha * direction)
            f_next = target.evaluate(x_next)

            actual_step = x_next - x
            dot_product = np.dot(grad, actual_step)

            if f_next <= f_x + self.c * dot_product:
                return alpha

            alpha *= self.beta

            if alpha < self.alpha_floor:
                return alpha


class ExactLineSearch(StepSizeStrategy):
    """1D scalar minimization of φ(α) = f(project(x + α·d)) via scipy."""
    def compute_alpha(self, x, grad, direction, target, projection) -> float:
        def phi(alpha):
            projected_x = projection.project(x + alpha * direction)
            return target.evaluate(projected_x)
        result = minimize_scalar(phi)
        return getattr(result, 'x')
