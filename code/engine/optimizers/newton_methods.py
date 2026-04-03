import numpy as np
from engine.optimizers.base import TraditionalOptimizer
from engine.utils import logger

class NewtonOptimizer(TraditionalOptimizer):
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.used_subgradient = False

    def _get_history_state(self):
        state = super()._get_history_state()
        state["subgrad"] = self.used_subgradient
        return state

    def _solve_direction(self, H, grad):
        """
        Solves the system H * h = -g for the search direction h.
        """
        try:
            direction = np.linalg.solve(H, -grad)
            return direction
        except np.linalg.LinAlgError:
            # Instead of falling back to gradient, we explicitly abort.
            # The try...except block in app.py will catch this and trigger the UI reset.
            logger.error("Singular Hessian encountered. Newton's method is inapplicable.")
            raise RuntimeError("The Hessian matrix is singular (flat or linear region). Newton's method cannot proceed.")

    def step(self):
        current_x = self.population[0]
        
        grad, is_subgrad = self.target.evaluate_gradient(current_x)
        H = self.target.evaluate_hessian(current_x)
        
        self.used_subgradient = is_subgrad
        self.current_grad = grad
        
        # This will raise a RuntimeError if H is singular
        direction = self._solve_direction(H, grad)
        
        alpha = self.get_alpha(current_x, grad, direction)
        new_x = current_x + alpha * direction
        
        return np.atleast_2d(new_x)