"""
Spiral search optimizer (Tamura & Yasuda 2011/2016 and follow-ups).

The four lecture variants (§17–§20, see
``Reports/Evolutionary/textbooks/Лекція_10.md``) are reached by swapping
the rotation/radius/restart strategies — the optimizer itself stays the
same.

Out-of-bounds policy (lecture §14): points that step outside ``bounds``
are not clipped back; their fitness is set to ``+inf`` so they never become
the pivot, but they continue to spiral toward it like any other point.
"""
from abc import ABC, abstractmethod
import numpy as np

from engine.utils import logger
from engine.strategies.stopping import MaxGenerationsCriterion
from ..initializers import PopulationInitializer
from ..strategies.spiral_rotation import RotationMatrixStrategy
from ..strategies.spiral_radius import RadiusStrategy
from .population_based import PopulationOptimizer


class RestartStrategy(ABC):
    """
    Restart hook called at the end of every ``step()``. Implementations
    mutate ``population_new`` in place (and optionally clear
    ``optimizer.center``); they return nothing.
    """
    @abstractmethod
    def maybe_restart(
        self,
        optimizer,
        step_idx: int,
        fitnesses: np.ndarray,
        population_new: np.ndarray,
    ) -> None:
        ...


class NoRestart(RestartStrategy):
    def maybe_restart(self, optimizer, step_idx, fitnesses, population_new):
        return


class PeriodicKeepBestRestart(RestartStrategy):
    """
    Lecture §20 ("improved" spiral search). After every ``rounds_length``
    iterations, keep the ``keep_ratio`` fraction of best (pre-rotation)
    points and resample the rest uniformly. The pivot is dropped so the
    next step re-elects from the fresh mixed population.
    """
    def __init__(self, rounds_length: int, keep_ratio: float = 0.2):
        if rounds_length <= 0:
            raise ValueError(f"rounds_length must be positive, got {rounds_length}")
        if not (0 < keep_ratio < 1):
            raise ValueError(f"keep_ratio must lie in (0, 1), got {keep_ratio}")
        self.rounds_length = int(rounds_length)
        self.keep_ratio = float(keep_ratio)

    def maybe_restart(self, optimizer, step_idx, fitnesses, population_new):
        if step_idx == 0 or step_idx % self.rounds_length != 0:
            return
        n_keep = max(1, int(optimizer.population_size * self.keep_ratio))
        keep_idx = np.argsort(fitnesses)[:n_keep]
        # `optimizer.population` is still the pre-rotation array — `run()`
        # only assigns the new one after `step()` returns.
        population_new[:n_keep] = optimizer.population[keep_idx]
        population_new[n_keep:] = optimizer.initializer.initialize(
            optimizer.population_size - n_keep, optimizer.target.bounds
        )
        optimizer.center = None


class SpiralOptimizer(PopulationOptimizer):
    """
    Spiral search base implementation.

    All four lecture variants reuse this class; only the injected strategies
    change.

    The transform ``S = r·R`` is cached at construction when both strategies
    are static (and the radius is scalar). Otherwise the rotation is
    recomputed only when its strategy is dynamic (the static rotation
    strategies memoize internally), and the radius is always cheap.
    """
    def __init__(
        self,
        target_function,
        population_size: int,
        initializer: PopulationInitializer,
        rotation_strategy: RotationMatrixStrategy,
        radius_strategy: RadiusStrategy,
        restart_strategy: RestartStrategy | None = None,
        **kwargs,
    ):
        kwargs.setdefault('stopping_criterion', MaxGenerationsCriterion())
        super().__init__(target_function, population_size, **kwargs)

        self.initializer = initializer
        self.rotation = rotation_strategy
        self.radius = radius_strategy
        self.restart_strategy = restart_strategy if restart_strategy is not None else NoRestart()

        self.population = self.initializer.initialize(
            population_size, self.target.bounds
        )
        self.center: np.ndarray | None = None

        N = len(self.target.variables)
        self._cached_S: np.ndarray | None = None
        if (
            self.rotation.is_static
            and self.radius.is_static
            and not self.radius.is_per_point
        ):
            R = self.rotation.get_matrix(N, 0)
            r = self.radius.get_r(0, None)
            self._cached_S = r * R

    def _evaluate_with_oob(self) -> np.ndarray:
        """Per-point fitness; ``+inf`` for points outside ``target.bounds`` (lecture §14)."""
        lb = np.array([b[0] for b in self.target.bounds])
        ub = np.array([b[1] for b in self.target.bounds])
        in_bounds = np.all((self.population >= lb) & (self.population <= ub), axis=1)

        f = np.full(self.population.shape[0], np.inf)
        for i, ok in enumerate(in_bounds):
            if ok:
                f[i] = self.target.evaluate(self.population[i])
        self.evaluations_count += int(np.sum(in_bounds))
        return f

    def _update_center(self, fitnesses: np.ndarray) -> None:
        """Sticky pivot: move ``center`` only when a strictly better point appears."""
        best_idx = int(np.argmin(fitnesses))
        best_f = float(fitnesses[best_idx])
        if not np.isfinite(best_f):
            return  # entire population was OOB — keep last pivot
        if self.center is None:
            self.center = self.population[best_idx].copy()
            self._center_f = best_f
            return
        if best_f < self._center_f:
            self.center = self.population[best_idx].copy()
            self._center_f = best_f

    def step(self) -> np.ndarray:
        fitnesses = self._evaluate_with_oob()
        self._update_center(fitnesses)

        shifted = self.population - self.center
        N = self.population.shape[1]

        if self._cached_S is not None:
            population_new = shifted @ self._cached_S.T + self.center
        else:
            R = self.rotation.get_matrix(N, self.results.iterations)
            r = self.radius.get_r(self.results.iterations, fitnesses)
            if self.radius.is_per_point:
                r_col = np.asarray(r, dtype=float).reshape(-1, 1)
                population_new = r_col * (shifted @ R.T) + self.center
            else:
                S = r * R
                population_new = shifted @ S.T + self.center

        self.restart_strategy.maybe_restart(
            self, self.results.iterations + 1, fitnesses, population_new
        )
        return population_new

    def _get_history_state(self):
        state = super()._get_history_state()
        state["center"] = None if self.center is None else self.center.copy()
        return state

    def _log_final_results(self):
        logger.info(
            f"Spiral optimization ended in {self.results.iterations} iterations."
        )
        if self.center is not None:
            logger.info(f"Final pivot (best known): {self.center}")
        logger.info(f"Final f(x*): {self.results.final_f}")
        logger.info("-" * 40)
