import numpy as np
from abc import ABC, abstractmethod

class CrossoverStrategy(ABC):
    """Abstract base class for crossover strategies."""
    @abstractmethod
    def crossover(self, parents: np.ndarray, n_cross: int, bounds: list) -> np.ndarray:
        """
        Args:
            parents: Matrix of shape (2 * n_cross, D) containing paired parents.
            n_cross: Number of children to generate.
            bounds: List of (min, max) per dimension to ensure children stay in bounds.
        """
        pass

class UniformCrossover(CrossoverStrategy):
    """
    Real-coded uniform crossover.
    Each child coordinate is a linear combination of parents:
    z_i = beta * x_i + (1 - beta) * y_i
    where beta is uniformly distributed in [-d, 1+d].
    """
    def __init__(self, d: float = 0.25):
        self.d = d

    def crossover(self, parents: np.ndarray, n_cross: int, bounds: list) -> np.ndarray:
        # Split parents into two groups
        parent1 = parents[:n_cross]
        parent2 = parents[n_cross:]
        
        # beta in [-d, 1+d]
        beta = np.random.uniform(-self.d, 1 + self.d, size=parent1.shape)
        
        # Compute offspring
        offspring = beta * parent1 + (1 - beta) * parent2
        
        # Enforce bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        return np.clip(offspring, lb, ub)

class NonUniformCrossover(CrossoverStrategy):
    """
    Non-uniform crossover (Arithmetic Crossover).
    Biases movement toward the fitter parent.
    z_i = x_i + rho * (y_i - x_i)
    """
    def __init__(self, s: float = 1.5):
        self.s = s

    def crossover(self, parents: np.ndarray, n_cross: int, bounds: list) -> np.ndarray:
        # Split parents
        parent1 = parents[:n_cross]
        parent2 = parents[n_cross:]
        
        # Simple arithmetic crossover
        # A more advanced version would check fitness and assign x as the fitter parent
        alpha = np.random.uniform(0, 1, size=(n_cross, 1))
        offspring = alpha * parent1 + (1 - alpha) * parent2
        
        # Enforce bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        return np.clip(offspring, lb, ub)