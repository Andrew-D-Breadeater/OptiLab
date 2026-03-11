import time
import numpy as np
from engine.utils import logger
from engine.models import OptimisationResults

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