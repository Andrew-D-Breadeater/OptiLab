import time
import numpy as np
from engine.utils import logger
from engine.models import OptimisationResults
from scipy.optimize import minimize_scalar
from engine.strategies.projections import NoProjection

class Optimizer:
    def __init__(self, target_function, **kwargs):
        self.target = target_function
        self.results = OptimisationResults()
        
        # Initialize state as a 2D array (N points, D dimensions). For GD, N=1.
        start_pos = kwargs.get('start_pos', np.zeros(len(target_function.variables)))
        self.population = np.atleast_2d(start_pos)
        self.stopping_criterion = kwargs.get('stopping_criterion', 'step_size')
        self.tol = 1e-6

    def _get_history_state(self):
        return {"population": self.population.copy()}

    def step(self):
        raise NotImplementedError("Subclasses must implement step()")

    def check_convergence(self, old_population):
        # Base implementation: check if the maximum movement of any point is below tolerance
        if self.stopping_criterion == 'step_size':
            max_movement = np.max(np.linalg.norm(self.population - old_population, axis=1))
            return max_movement < self.tol
        return False

    def _log_final_results(self):
        # Default behavior: log the entire population
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final f(x): {self.results.final_f}")

    def run(self, max_iter=1000, tol=1e-6, callback=None):
        start_time = time.time()
        self.tol = tol
        
        # Centralized startup logging
        logger.info(f"--- Starting {self.__class__.__name__} ---")
        # Log all instance variables as parameters
        logger.info(f"Parameters: {self.__dict__}")
        
        for i in range(max_iter):
            self.results.history.append(self._get_history_state())
            
            old_population = self.population.copy()
            self.population = self.step()
            self.results.iterations += 1
            
            if callback:
                callback(self.results.iterations)
            
            if self.check_convergence(old_population):
                self.results.converged = True
                self.results.history.append(self._get_history_state())
                break
        
        self.results.execution_time = time.time() - start_time
        self.results.final_population = self.population
        f_vals = [self.target.evaluate(p) for p in self.population]
        self.results.final_f = np.min(f_vals)
        
        self._log_final_results() # Hook called here
        return self.results
    
class TraditionalOptimizer(Optimizer):
    """Intermediate base class for single-point gradient/Hessian based methods."""
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.use_line_search = kwargs.get('use_line_search', False)
        self.use_exact_line_search = kwargs.get('use_exact_line_search', False)
        self.projection_strategy = kwargs.get('projection_strategy', NoProjection())
        self.current_grad = None

    def _log_final_results(self):
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final best point: {self.population[0]}")
        logger.info(f"Final f(x): {self.results.final_f}")
        logger.info("-" * 40)

    def check_convergence(self, old_population):
        if self.stopping_criterion == 'gradient_norm' and self.current_grad is not None:
            current_x = self.population[0]
            mapped_x = self.projection_strategy.project(current_x - self.current_grad)
            return np.max(np.abs(current_x - mapped_x)) < self.tol
        return super().check_convergence(old_population)

    def get_alpha(self, x, grad, direction):
        """Helper to determine step size based on configuration."""
        if self.use_exact_line_search:
            return self._exact_line_search(x, direction)
        elif self.use_line_search:
            return self._backtracking_line_search(x, grad, direction)
        else:
            return self.learning_rate

    def _exact_line_search(self, x, direction):
        def phi(alpha):
            projected_x = self.projection_strategy.project(x + alpha * direction)
            return self.target.evaluate(projected_x)
        result = minimize_scalar(phi)
        return getattr(result, 'x')
    
    def _backtracking_line_search(self, x, grad, direction, alpha=1.0, beta=0.5, c=1e-4):
        f_x = self.target.evaluate(x)
        
        while True:
            # Project the proposed step onto the admissible set X
            x_next = self.projection_strategy.project(x + alpha * direction)
            f_next = self.target.evaluate(x_next)
            
            actual_step = x_next - x
            
            dot_product = np.dot(grad, actual_step)
            
            if f_next <= f_x + c * dot_product:
                return alpha
            
            alpha *= beta
            
            if alpha < 1e-12:
                return alpha