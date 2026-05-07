import warnings
import numpy as np
from engine.utils import logger
from .base import Optimizer
from ..initializers import PopulationInitializer
from ..strategies.selection import SelectionStrategy
from ..strategies.crossover import CrossoverStrategy
from ..strategies.mutation import MutationStrategy
from ..strategies.stopping import MaxGenerationsCriterion


class PopulationOptimizer(Optimizer):
    """
    Base for population-based methods (GA, PSO, etc.).

    Stopping logic lives entirely in the injected `stopping_criterion`.
    The only state this class adds for criteria to read is `evaluations_count`,
    which `step()` increments and `MaxEvaluationsCriterion` consumes.
    """
    def __init__(self, target_function, population_size: int, **kwargs):
        super().__init__(target_function, **kwargs)
        self.population_size = population_size
        self.evaluations_count = 0

    def _log_final_results(self):
        f_vals = [self.target.evaluate(p) for p in self.population]
        best_idx = np.argmin(f_vals)

        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final best point: {self.population[best_idx]}")
        logger.info(f"Final f(x): {self.results.final_f}")
        logger.info("-" * 40)


class GeneticAlgorithm(PopulationOptimizer):
    """The main Genetic Algorithm engine using injected strategies."""
    def __init__(self,
                 target_function,
                 population_size: int,
                 initializer: PopulationInitializer,
                 selection_strategy: SelectionStrategy,
                 crossover_strategy: CrossoverStrategy,
                 mutation_strategy: MutationStrategy,
                 phi_sel: float,
                 phi_cross: float,
                 phi_mut: float,
                 **kwargs):

        super().__init__(target_function, population_size, **kwargs)

        total_phi = phi_sel + phi_cross + phi_mut
        if not np.isclose(total_phi, 1.0):
            warnings.warn(f"Coefficients sum to {total_phi:.2f}, not 1.0. Normalizing.")
            phi_sel /= total_phi
            phi_cross /= total_phi
            phi_mut /= total_phi

        self.initializer = initializer
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy

        self.phi_sel = phi_sel
        self.phi_cross = phi_cross
        self.phi_mut = phi_mut

        self.population = self.initializer.initialize(self.population_size, self.target.bounds)

    def step(self) -> np.ndarray:
        current_fitnesses = np.array([self.target.evaluate(p) for p in self.population])
        self.evaluations_count += self.population_size

        n_sel = int(self.population_size * self.phi_sel)
        n_cross = int(self.population_size * self.phi_cross)
        n_mut = self.population_size - n_sel - n_cross

        selected_individuals = self.selection_strategy.select(self.population, current_fitnesses, n_sel)

        crossover_parents_indices = np.random.choice(self.population_size, size=n_cross * 2, replace=True)
        crossover_parents = self.population[crossover_parents_indices]
        crossover_parent_fitnesses = current_fitnesses[crossover_parents_indices]
        crossed_individuals = self.crossover_strategy.crossover(crossover_parents, crossover_parent_fitnesses, n_cross, self.target.bounds)

        mutation_parents_indices = np.random.choice(self.population_size, size=n_mut, replace=True)
        mutation_parents = self.population[mutation_parents_indices]
        mutated_individuals = self.mutation_strategy.mutate(mutation_parents, n_mut, self.target.bounds)

        new_population = np.vstack([selected_individuals, crossed_individuals, mutated_individuals])
        np.random.shuffle(new_population)
        return new_population


class MCOGeneticAlgorithm(GeneticAlgorithm):
    """
    Multi-Criteria Optimization GA.

    Defaults `stopping_criterion` to `MaxGenerationsCriterion()` because
    stagnation-on-scalar-best is meaningless for a Pareto front. Caller can
    override by passing any other StoppingCriterion explicitly.
    """
    def __init__(self, target_function, population_size: int, **kwargs):
        kwargs.setdefault('stopping_criterion', MaxGenerationsCriterion())
        super().__init__(target_function, population_size, **kwargs)

    def _calculate_pareto_fitness(self, objs: np.ndarray) -> np.ndarray:
        """
        Fitness = number of population members that strictly dominate this point.
        Score 0 ⇒ on the current Pareto front.
        """
        N = objs.shape[0]
        fitnesses = np.zeros(N)
        for i in range(N):
            dominators = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
            fitnesses[i] = np.sum(dominators)
        return fitnesses

    def _get_history_state(self):
        state = super()._get_history_state()
        objs = np.array([self.target.evaluate(p) for p in self.population])
        fitnesses = self._calculate_pareto_fitness(objs)
        state["objectives"] = objs
        state["pareto_mask"] = (fitnesses == 0)
        return state

    def step(self) -> np.ndarray:
        objs = np.array([self.target.evaluate(p) for p in self.population])
        current_fitnesses = self._calculate_pareto_fitness(objs)
        self.evaluations_count += self.population_size * self.target.num_objectives

        n_sel = int(self.population_size * self.phi_sel)
        n_cross = int(self.population_size * self.phi_cross)
        n_mut = self.population_size - n_sel - n_cross

        selected_individuals = self.selection_strategy.select(self.population, current_fitnesses, n_sel)

        crossover_parents_indices = np.random.choice(self.population_size, size=n_cross * 2, replace=True)
        crossover_parents = self.population[crossover_parents_indices]
        crossover_parent_fitnesses = current_fitnesses[crossover_parents_indices]
        crossed_individuals = self.crossover_strategy.crossover(crossover_parents, crossover_parent_fitnesses, n_cross, self.target.bounds)

        mutation_parents_indices = np.random.choice(self.population_size, size=n_mut, replace=True)
        mutation_parents = self.population[mutation_parents_indices]
        mutated_individuals = self.mutation_strategy.mutate(mutation_parents, n_mut, self.target.bounds)

        new_population = np.vstack([selected_individuals, crossed_individuals, mutated_individuals])
        np.random.shuffle(new_population)
        return new_population

    def _log_final_results(self):
        objs = np.array([self.target.evaluate(p) for p in self.population])
        fitnesses = self._calculate_pareto_fitness(objs)
        pareto_count = np.sum(fitnesses == 0)

        logger.info(f"MCO Optimization ended in {self.results.iterations} iterations.")
        logger.info(f"Found {pareto_count} Pareto-optimal points out of {self.population_size} individuals.")
        logger.info("-" * 40)
