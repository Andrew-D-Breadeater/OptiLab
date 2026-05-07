import time
import numpy as np
from engine.utils import logger
from engine.models import OptimisationResults
from engine.strategies.projections import NoProjection


class Optimizer:
    def __init__(self, target_function, **kwargs):
        self.target = target_function
        self.results = OptimisationResults()

        # State as 2D array (N points, D dimensions). For single-agent methods, N=1.
        start_pos = kwargs.get('start_pos', np.zeros(len(target_function.variables)))
        self.population = np.atleast_2d(start_pos)

        self.stopping_criterion = kwargs.get('stopping_criterion')
        if self.stopping_criterion is None:
            raise ValueError(
                f"{self.__class__.__name__} requires 'stopping_criterion' "
                "(pass a StoppingCriterion instance from engine.strategies.stopping)"
            )

    def _get_history_state(self):
        return {"population": self.population.copy()}

    def step(self):
        raise NotImplementedError("Subclasses must implement step()")

    def _log_final_results(self):
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final f(x): {self.results.final_f}")

    def run(self, max_iter=1000, callback=None):
        """
        ``max_iter`` bounds the **total number of history frames** produced
        — not the number of outer ``step()`` calls. For optimizers whose
        ``step()`` emits one frame per call (plain GD, Newton, GA) the two
        notions coincide. For step strategies that emit multiple intermediate
        frames per call (e.g. ``RavineStep``'s inner descent), every emitted
        frame counts. The bound is *soft*: a single ``step()`` whose internal
        emissions overshoot ``max_iter`` finishes its work, then the loop
        exits.
        """
        start_time = time.time()

        logger.info(f"--- Starting {self.__class__.__name__} ---")
        logger.info(f"Parameters: {self.__dict__}")

        self.stopping_criterion.on_run_start(self)

        while len(self.results.history) < max_iter:
            self.results.history.append(self._get_history_state())

            old_population = self.population.copy()
            self.population = self.step()
            self.results.iterations = len(self.results.history)

            if callback:
                callback(self.results.iterations)

            if len(self.results.history) >= max_iter:
                break

            if self.stopping_criterion.should_stop(self, old_population):
                self.results.converged = True
                self.results.history.append(self._get_history_state())
                self.results.iterations = len(self.results.history)
                break

        self.results.execution_time = time.time() - start_time
        self.results.final_population = self.population
        f_vals = [self.target.evaluate(p) for p in self.population]
        self.results.final_f = np.min(f_vals)

        self._log_final_results()
        return self.results


class TraditionalOptimizer(Optimizer):
    """Single-point gradient/Hessian methods with line searches and projections."""
    def __init__(self, target_function, **kwargs):
        super().__init__(target_function, **kwargs)
        self.step_size_strategy = kwargs.get('step_size_strategy')
        if self.step_size_strategy is None:
            raise ValueError(
                f"{self.__class__.__name__} requires 'step_size_strategy' "
                "(pass a StepSizeStrategy instance from engine.strategies.step_size)"
            )
        self.projection_strategy = kwargs.get('projection_strategy', NoProjection())
        self.current_grad = None

    def _log_final_results(self):
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final best point: {self.population[0]}")
        logger.info(f"Final f(x): {self.results.final_f}")
        logger.info("-" * 40)

    def get_alpha(self, x, grad, direction):
        return self.step_size_strategy.compute_alpha(
            x, grad, direction, self.target, self.projection_strategy
        )
