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
from scipy.optimize import minimize_scalar, line_search as _scipy_line_search


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


class StrongWolfeLineSearch(StepSizeStrategy):
    """
    Strong Wolfe conditions:

        f(x + α d) ≤ f(x) + c1 α ⟨∇f(x), d⟩         (Armijo)
        |⟨∇f(x + α d), d⟩| ≤ c2 |⟨∇f(x), d⟩|        (curvature)

    Required by CG when the direction is built from prev_direction — pure
    Armijo backtracking can leave (∇f(x_{k+1}), h_{k+1}) ≥ 0, i.e. the next
    CG direction is no longer a descent direction.

    Typical CG choices: c1 = 1e-4, c2 = 0.1. We default to those.

    Delegates to ``scipy.optimize.line_search`` (Nocedal & Wright Algorithm
    3.5/3.6 — bracket then zoom). Projection is honoured by composing
    ``project(x + α·d)`` inside the wrapped φ and ∇φ, matching the pattern
    used by ``ExactLineSearch`` / ``BacktrackingLineSearch``.

    Failure mode: scipy returns ``None`` when it cannot find a Wolfe-valid α
    within ``max_iter`` evaluations — typically because the direction is not
    actually a descent direction (``∇f · d ≥ 0``) or the function is not
    bounded below along it. We **raise** rather than silently fall back,
    matching the project's "fail loudly" convention (cf.
    ``NewtonOptimizer._solve_direction``). The UI's COMPUTING-phase
    try/except surfaces the error and offers a Return-to-Setup button.
    """
    def __init__(self, c1: float = 1e-4, c2: float = 0.1, max_iter: int = 25):
        if not (0 < c1 < c2 < 1):
            raise ValueError("Strong Wolfe requires 0 < c1 < c2 < 1.")
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def compute_alpha(self, x, grad, direction, target, projection) -> float:
        def phi(z):
            return target.evaluate(projection.project(z))

        def grad_phi(z):
            g, _ = target.evaluate_gradient(projection.project(z))
            return g

        result = _scipy_line_search(
            f=phi, myfprime=grad_phi,
            xk=x, pk=direction,
            gfk=grad,
            c1=self.c1, c2=self.c2,
            maxiter=self.max_iter,
        )
        alpha = result[0]
        if alpha is None or alpha <= 0:
            raise RuntimeError(
                "Strong Wolfe line search failed to find an acceptable α "
                "(scipy.optimize.line_search returned None). The search "
                "direction is likely not a descent direction, or the "
                "function is unbounded below along it. Try a different β "
                "formula (PR is the most robust), enable restart, or check "
                "the target function and starting point."
            )
        return float(alpha)
