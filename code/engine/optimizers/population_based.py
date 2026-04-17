import time
import warnings
import numpy as np
from engine.utils import logger
from .base import Optimizer
from ..initializers import PopulationInitializer
from ..strategies.selection import SelectionStrategy
from ..strategies.crossover import CrossoverStrategy
from ..strategies.mutation import MutationStrategy

class PopulationOptimizer(Optimizer):
    """
    Base class for population-based methods like GA, PSO, etc.
    Overrides convergence to handle population-specific termination criteria.
    """
    def __init__(self, target_function, population_size: int, **kwargs):
        super().__init__(target_function, **kwargs)
        self.population_size = population_size
        
        # We inherit self.stopping_criterion from base.py
        # Options will be: 'stagnation', 'degeneration', 'time_limit', 'max_evals', 'max_generations'
        
        self.patience = kwargs.get('patience', 15) 
        self._stagnation_counter = 0
        self._best_fitness_so_far = float('inf')
        
        self.degeneration_tol = kwargs.get('degeneration_tol', 1e-4)
        
        self.time_limit = kwargs.get('time_limit', float('inf')) 
        self.max_evals = kwargs.get('max_evals', float('inf'))
        
        self._evaluations_count = 0
        self._start_time = None

    def _log_final_results(self):
        f_vals = [self.target.evaluate(p) for p in self.population]
        best_idx = np.argmin(f_vals)
        
        logger.info(f"Optimization ended. Converged: {self.results.converged} in {self.results.iterations} iterations.")
        logger.info(f"Final best point: {self.population[best_idx]}")
        logger.info(f"Final f(x): {self.results.final_f}")
        logger.info("-" * 40)

    def check_convergence(self, old_population) -> bool:
        
        if self._start_time is None:
            self._start_time = time.time()
            
        if self.stopping_criterion == 'time_limit':
            return (time.time() - self._start_time) >= self.time_limit
            
        elif self.stopping_criterion == 'max_evals':
            return self._evaluations_count >= self.max_evals
            
        elif self.stopping_criterion == 'stagnation':
            current_fitnesses =[self.target.evaluate(p) for p in self.population]
            current_best = np.min(current_fitnesses)
            
            if current_best < self._best_fitness_so_far - 1e-9:
                self._best_fitness_so_far = current_best
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1
                
            return self._stagnation_counter >= self.patience
            
        elif self.stopping_criterion == 'degeneration':
            pop_std = np.std(self.population, axis=0)
            return np.max(pop_std) < self.degeneration_tol
            
        # If stopping_criterion == 'max_generations', we just return False 
        # and let the base.py max_iter loop handle the termination naturally.
        return False


class GeneticAlgorithm(PopulationOptimizer):
    """
    The main Genetic Algorithm engine using injected strategies.
    """
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
        
        # Track evaluations for Criterion 4
        self._evaluations_count += self.population_size

        n_sel = int(self.population_size * self.phi_sel)
        n_cross = int(self.population_size * self.phi_cross)
        n_mut = self.population_size - n_sel - n_cross 

        selected_individuals = self.selection_strategy.select(self.population, current_fitnesses, n_sel)
        
        crossover_parents_indices = np.random.choice(self.population_size, size=n_cross * 2, replace=True)
        crossover_parents = self.population[crossover_parents_indices]
        crossed_individuals = self.crossover_strategy.crossover(crossover_parents, n_cross, self.target.bounds)

        mutation_parents_indices = np.random.choice(self.population_size, size=n_mut, replace=True)
        mutation_parents = self.population[mutation_parents_indices]
        mutated_individuals = self.mutation_strategy.mutate(mutation_parents, n_mut, self.target.bounds)

        new_population = np.vstack([
            selected_individuals,
            crossed_individuals,
            mutated_individuals
        ])
        
        np.random.shuffle(new_population)
        
        return new_population
    
class MCOGeneticAlgorithm(GeneticAlgorithm):
    """
    Multi-Criteria Optimization (MCO) Genetic Algorithm.
    Uses Pareto dominance ranking to calculate fitness.
    """
    def __init__(self, target_function, population_size: int, **kwargs):
        # For MCO, stagnation on a single scalar fitness doesn't make sense.
        # We default to running for a fixed number of generations (max_iter).
        kwargs['stopping_criterion'] = kwargs.get('stopping_criterion', 'max_generations')
        super().__init__(target_function, population_size, **kwargs)

    def _calculate_pareto_fitness(self, objs: np.ndarray) -> np.ndarray:
        """
        Calculates fitness based on Pareto dominance.
        Fitness = number of individuals in the population that strictly dominate this point.
        A score of 0 means the point is Pareto-optimal (non-dominated) in the current generation.
        
        Args:
            objs: (N, M) matrix of objective values. N = pop_size, M = num_objectives.
        """
        N = objs.shape[0]
        fitnesses = np.zeros(N)
        
        for i in range(N):
            # Point j dominates point i if j is <= i in ALL objectives, 
            # and strictly < i in AT LEAST ONE objective.
            # We do this vectorized for speed.
            dominators = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
            fitnesses[i] = np.sum(dominators)
            
        return fitnesses

    def _get_history_state(self):
        """
        Enrich the history with objective values and the Pareto mask 
        so the Jupyter Notebook can easily plot the front.
        """
        state = super()._get_history_state()
        
        # Evaluate objectives for the current population
        objs = np.array([self.target.evaluate(p) for p in self.population])
        fitnesses = self._calculate_pareto_fitness(objs)
        
        state["objectives"] = objs
        state["pareto_mask"] = (fitnesses == 0) # Boolean mask: True if point is on the Pareto front
        
        return state

    def step(self) -> np.ndarray:
        """
        Executes one generation cycle using Pareto fitness.
        """
        # 1. Evaluate multiple objectives for each point
        objs = np.array([self.target.evaluate(p) for p in self.population])
        
        # 2. Fitness is the Pareto dominance count (0 is best)
        current_fitnesses = self._calculate_pareto_fitness(objs)
        
        # Track evaluations (N points * M objectives)
        self._evaluations_count += self.population_size * self.target.num_objectives

        # 3. Standard GA selection, crossover, and mutation using the new fitness
        n_sel = int(self.population_size * self.phi_sel)
        n_cross = int(self.population_size * self.phi_cross)
        n_mut = self.population_size - n_sel - n_cross 

        selected_individuals = self.selection_strategy.select(self.population, current_fitnesses, n_sel)
        
        crossover_parents_indices = np.random.choice(self.population_size, size=n_cross * 2, replace=True)
        crossover_parents = self.population[crossover_parents_indices]
        crossed_individuals = self.crossover_strategy.crossover(crossover_parents, n_cross, self.target.bounds)

        mutation_parents_indices = np.random.choice(self.population_size, size=n_mut, replace=True)
        mutation_parents = self.population[mutation_parents_indices]
        mutated_individuals = self.mutation_strategy.mutate(mutation_parents, n_mut, self.target.bounds)

        new_population = np.vstack([
            selected_individuals,
            crossed_individuals,
            mutated_individuals
        ])
        
        np.random.shuffle(new_population)
        
        return new_population

    def _log_final_results(self):
        """
        Overrides the base logger to provide MCO-specific output.
        """
        objs = np.array([self.target.evaluate(p) for p in self.population])
        fitnesses = self._calculate_pareto_fitness(objs)
        pareto_count = np.sum(fitnesses == 0)
        
        logger.info(f"MCO Optimization ended in {self.results.iterations} iterations.")
        logger.info(f"Found {pareto_count} Pareto-optimal points out of {self.population_size} individuals.")
        logger.info("-" * 40)