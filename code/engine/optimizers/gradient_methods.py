import numpy as np
from scipy.optimize import minimize_scalar
from engine.optimizers.base import Optimizer
from engine.utils import logger

class GradientDescent(Optimizer):
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.decay_rate = kwargs.get('decay_rate', 1.0)
        self.use_line_search = kwargs.get('use_line_search', False)
        self.use_exact_line_search = kwargs.get('use_exact_line_search', False)
        
        self.use_ravine = kwargs.get('use_ravine', False)
        self.ravine_step_size = kwargs.get('ravine_step_size', 0.5)
        self.ravine_shift = kwargs.get('ravine_shift', 0.5)
        self.prev_base_point = None 
        self.is_ravine_step = False
        self.initial_shift_done = False
        
        self.current_grad = None # Store gradient for convergence checking
        self.used_subgradient = False

    def _get_history_state(self):
        state = super()._get_history_state()
        state["subgrad"] = self.used_subgradient
        return state
    
    def _log_final_results(self):
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final best point: {self.population[0]}")
        logger.info(f"Final f(x): {self.results.final_f}")
        logger.info("-" * 40)
    
    def check_convergence(self, old_population):
        if self.stopping_criterion == 'gradient_norm' and self.current_grad is not None:
            # infinity norm of the gradient: max absolute component
            return np.max(np.abs(self.current_grad)) < self.tol
        # Fallback to the base class check (step_size)
        return super().check_convergence(old_population)

    # (Keep your _exact_line_search and _backtracking_line_search methods exactly as they are)

    def _exact_line_search(self, x, direction):
        
        # Define the 1D function phi(alpha) = f(x + alpha * direction)
        def phi(alpha):
            return self.target.evaluate(x + alpha * direction)
            
        # Run 1D optimization to find the best alpha
        result = minimize_scalar(phi)
        return result.x

    def _backtracking_line_search(self, x, grad, direction, 
                                 alpha=1.0, beta=0.5, c=1e-4):
        """
        Finds alpha that satisfies the Armijo condition.
        alpha: initial step size
        beta: contraction factor (0 < beta < 1)
        c: sufficient decrease constant (0 < c < 1)
        """
        f_x = self.target.evaluate(x)
        # Precompute the dot product: grad^T * direction
        dot_product = np.dot(grad, direction)
        
        while True:
            # Calculate f(x + alpha * direction)
            f_next = self.target.evaluate(x + alpha * direction)
            
            # Check Armijo condition
            if f_next <= f_x + c * alpha * dot_product:
                return alpha
            
            # Reduce alpha
            alpha *= beta
            
            # Safety break to prevent infinite loop on flat regions
            if alpha < 1e-12:
                return alpha
            
    def step(self):
        current_x = self.population[0] # Extract the single point for GD
        
        # 1. Initialization: Create second base point via shift
        if self.use_ravine and not self.initial_shift_done:
            self.prev_base_point = current_x.copy()
            shift_vector = np.ones_like(current_x) * self.ravine_shift
            shifted_x = current_x + shift_vector
            
            grad, is_subgrad = self.target.evaluate_gradient(shifted_x)
            self.used_subgradient = is_subgrad
            self.current_grad = grad
            direction = -grad
            alpha = self._backtracking_line_search(shifted_x, grad, direction) if self.use_line_search else self.learning_rate
            
            self.initial_shift_done = True
            self.is_ravine_step = True
            return np.atleast_2d(shifted_x + alpha * direction)

        # 2. Ravine Extrapolation Step
        if self.use_ravine and self.is_ravine_step and self.prev_base_point is not None:
            x_k = current_x
            x_prev = self.prev_base_point
            
            f_k = self.target.evaluate(x_k)
            f_prev = self.target.evaluate(x_prev)
            
            diff_vector = x_k - x_prev
            norm = np.linalg.norm(diff_vector)
            
            if norm == 0:
                self.is_ravine_step = False
                return np.atleast_2d(x_k)
            
            direction_normalized = diff_vector / norm
            sign_f = np.sign(f_k - f_prev)
            
            v_next = x_k - direction_normalized * self.ravine_step_size * sign_f
            
            self.prev_base_point = x_k.copy()
            self.is_ravine_step = False
            self.used_subgradient = False 
            
            return np.atleast_2d(v_next)

        # 3. Regular Gradient Step
        grad, is_subgrad = self.target.evaluate_gradient(current_x)
        self.used_subgradient = is_subgrad
        self.current_grad = grad
        direction = -grad
        
        alpha = self._backtracking_line_search(current_x, grad, direction) if self.use_line_search else self.learning_rate
            
        new_x = current_x + alpha * direction
        
        if self.use_ravine:
            self.is_ravine_step = True
        
        self.learning_rate *= self.decay_rate
            
        return np.atleast_2d(new_x)