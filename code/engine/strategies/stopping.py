"""
Stopping criteria as Strategy objects.

Each criterion owns its own parameters and bookkeeping state. The optimizer
calls ``on_run_start`` once before iteration begins, then ``should_stop``
after every step; the criterion decides whether to halt.

To combine criteria, wrap them in ``AnyOf`` (stop on any) or ``AllOf``
(stop only when all signal). All children's ``should_stop`` are evaluated
on every step so their internal state stays consistent — short-circuiting
would cause counters in e.g. ``StagnationCriterion`` to drift.
"""
import time
from abc import ABC, abstractmethod
import numpy as np


class StoppingCriterion(ABC):
    def on_run_start(self, optimizer) -> None:
        """Called once before the iteration loop. Default: no-op."""

    @abstractmethod
    def should_stop(self, optimizer, old_population: np.ndarray) -> bool:
        ...


class StepSizeCriterion(StoppingCriterion):
    """Stop when no point moves more than `tol` between iterations."""
    def __init__(self, tol: float):
        self.tol = tol

    def should_stop(self, optimizer, old_population):
        max_movement = np.max(np.linalg.norm(optimizer.population - old_population, axis=1))
        return max_movement < self.tol


class GradientNormCriterion(StoppingCriterion):
    """
    Projection-aware gradient stop: x is a fixed point of the
    projected-gradient map within `tol`.

    Requires the optimizer to expose `current_grad` and `projection_strategy`
    (i.e. used with `TraditionalOptimizer` and its subclasses).
    """
    def __init__(self, tol: float):
        self.tol = tol

    def should_stop(self, optimizer, old_population):
        if optimizer.current_grad is None:
            return False
        current_x = optimizer.population[0]
        mapped_x = optimizer.projection_strategy.project(current_x - optimizer.current_grad)
        return np.max(np.abs(current_x - mapped_x)) < self.tol


class StagnationCriterion(StoppingCriterion):
    """Stop when best fitness has not improved by `eps` for `patience` iterations."""
    def __init__(self, patience: int, eps: float = 1e-9):
        self.patience = patience
        self.eps = eps
        self._counter = 0
        self._best_so_far = float('inf')

    def on_run_start(self, optimizer):
        self._counter = 0
        self._best_so_far = float('inf')

    def should_stop(self, optimizer, old_population):
        current_fitnesses = [optimizer.target.evaluate(p) for p in optimizer.population]
        current_best = float(np.min(current_fitnesses))
        if current_best < self._best_so_far - self.eps:
            self._best_so_far = current_best
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


class DegenerationCriterion(StoppingCriterion):
    """Stop when the population has collapsed (max per-axis std below `tol`)."""
    def __init__(self, tol: float):
        self.tol = tol

    def should_stop(self, optimizer, old_population):
        pop_std = np.std(optimizer.population, axis=0)
        return np.max(pop_std) < self.tol


class TimeLimitCriterion(StoppingCriterion):
    """Stop after `seconds` of wall-clock time since the run started."""
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._start_time = None

    def on_run_start(self, optimizer):
        self._start_time = time.time()

    def should_stop(self, optimizer, old_population):
        return (time.time() - self._start_time) >= self.seconds


class MaxEvaluationsCriterion(StoppingCriterion):
    """
    Stop when total objective evaluations exceed `max_evals`.

    Reads `optimizer.evaluations_count`, which population-based optimizers
    increment inside their `step()` method.
    """
    def __init__(self, max_evals: int):
        self.max_evals = max_evals

    def should_stop(self, optimizer, old_population):
        return getattr(optimizer, 'evaluations_count', 0) >= self.max_evals


class MaxGenerationsCriterion(StoppingCriterion):
    """Defer to the outer ``run(max_iter=...)`` loop. Never signals on its own."""
    def should_stop(self, optimizer, old_population):
        return False


class AnyOf(StoppingCriterion):
    """Stop when any child criterion signals. All children are still polled each step."""
    def __init__(self, criteria):
        self.criteria = list(criteria)

    def on_run_start(self, optimizer):
        for c in self.criteria:
            c.on_run_start(optimizer)

    def should_stop(self, optimizer, old_population):
        results = [c.should_stop(optimizer, old_population) for c in self.criteria]
        return any(results)


class AllOf(StoppingCriterion):
    """Stop only when every child criterion signals on the same step."""
    def __init__(self, criteria):
        self.criteria = list(criteria)

    def on_run_start(self, optimizer):
        for c in self.criteria:
            c.on_run_start(optimizer)

    def should_stop(self, optimizer, old_population):
        results = [c.should_stop(optimizer, old_population) for c in self.criteria]
        return all(results)
