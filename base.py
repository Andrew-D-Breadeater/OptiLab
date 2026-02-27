import time
import numpy as np
from models import OptimisationResults, TargetFunction
from utils import logger

class Optimizer:
    def __init__(self, target_function, **kwargs):
        self.target = target_function
        self.results = OptimisationResults()
        # Set a default starting point if not provided in kwargs
        self.current_x = kwargs.get('start_pos', np.zeros(len(target_function.variables)))
        self.used_subgradient = False # Track if the last step used a subgradient for logging purposes

        logger.info(f"--- Starting {self.__class__.__name__} ---")
        logger.info(f"Parameters: {kwargs}")

    def step(self):
        """Perform one iteration. Must be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement step()")

    def run(self, max_iter=1000, tol=1e-6):
        start_time = time.time()
        
        for i in range(max_iter):
            # Log state before stepping
            self.results.history.append({
                "x": self.current_x.copy(),
                "subgrad": self.used_subgradient
            })
            
            old_x = self.current_x.copy()
            self.current_x = self.step()
            
            self.results.iterations += 1
            
            # Basic convergence check: change in x
            if np.linalg.norm(self.current_x - old_x) < tol:
                self.results.converged = True
                break
        
        self.results.execution_time = time.time() - start_time
        self.results.final_x = self.current_x
        self.results.final_f = self.target.evaluate(self.current_x)
        
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final x: {self.results.final_x}")
        logger.info(f"Final f(x): {self.results.final_f}")
        logger.info("-" * 40)
        
        return self.results