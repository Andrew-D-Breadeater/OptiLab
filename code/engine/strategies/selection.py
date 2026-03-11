import numpy as np
from abc import ABC, abstractmethod

class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    @abstractmethod
    def select(self, population: np.ndarray, fitnesses: np.ndarray, n_select: int) -> np.ndarray:
        pass

class ElitismSelection(SelectionStrategy):
    """Directly selects the n_select individuals with the lowest fitness values."""
    def select(self, population: np.ndarray, fitnesses: np.ndarray, n_select: int) -> np.ndarray:
        # Argsort returns indices that would sort the array from lowest to highest
        indices = np.argsort(fitnesses)[:n_select]
        return population[indices]

class RouletteWheelSelection(SelectionStrategy):
    """Selects individuals with probability proportional to their relative fitness."""
    def select(self, population: np.ndarray, fitnesses: np.ndarray, n_select: int) -> np.ndarray:
        # Invert fitness for minimization: lower values = higher probability
        # We use min-max scaling to ensure positive values
        f_min = np.min(fitnesses)
        f_max = np.max(fitnesses)
        
        # Avoid division by zero if all fitnesses are equal
        if np.isclose(f_max, f_min):
            probabilities = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            # Scale so best fitness has largest value
            scaled_fitness = (f_max - fitnesses) / (f_max - f_min)
            probabilities = scaled_fitness / np.sum(scaled_fitness)
        
        indices = np.random.choice(len(population), size=n_select, p=probabilities, replace=True)
        return population[indices]

class RankSelection(SelectionStrategy):
    """Probability is proportional to the rank (worst=1, best=P)."""
    def select(self, population: np.ndarray, fitnesses: np.ndarray, n_select: int) -> np.ndarray:
        p = len(fitnesses)
        # Get rank order (indices of sorted fitness)
        ranks = np.argsort(np.argsort(fitnesses))
        # Convert to score: best rank (0) gets score p, worst rank (p-1) gets score 1
        scores = (p - ranks)
        probabilities = scores / np.sum(scores)
        
        indices = np.random.choice(p, size=n_select, p=probabilities, replace=True)
        return population[indices]

class TournamentSelection(SelectionStrategy):
    """Picks a random subset and selects the best individual from that subset."""
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: np.ndarray, fitnesses: np.ndarray, n_select: int) -> np.ndarray:
        selected_indices = []
        pop_size = len(population)
        
        for _ in range(n_select):
            # Pick random competitors
            candidates = np.random.choice(pop_size, self.tournament_size, replace=False)
            # Find the best among candidates
            best_in_tournament = candidates[np.argmin(fitnesses[candidates])]
            selected_indices.append(best_in_tournament)
            
        return population[selected_indices]