import numpy as np
from abc import ABC, abstractmethod

class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""
    @abstractmethod
    def mutate(self, parents: np.ndarray, n_mut: int, bounds: list) -> np.ndarray:
        """
        Args:
            parents: Matrix of shape (n_mut, D) to be mutated.
            n_mut: Number of individuals to mutate.
            bounds: List of (min, max) per dimension for clamping.
        """
        pass

class RealCodedMutation(MutationStrategy):
    """
    Gaussian Mutation (Real-coded).
    Adds random Gaussian noise to the parent: Y = X + N(0, sigma)
    """
    def __init__(self, sigma: float = 0.1, mutation_rate: float = 0.1):
        """
        Args:
            sigma: Standard deviation of the Gaussian noise (step size).
            mutation_rate: Probability that a specific dimension will be mutated.
        """
        self.sigma = sigma
        self.mutation_rate = mutation_rate

    def mutate(self, parents: np.ndarray, n_mut: int, bounds: list) -> np.ndarray:
        # Create a copy of parents to mutate
        offspring = parents.copy()
        n_mut, d = offspring.shape
        
        # Determine which dimensions to mutate based on mutation_rate
        # mask is (n_mut, d) of True/False values
        mask = np.random.rand(n_mut, d) < self.mutation_rate
        
        # Generate Gaussian noise
        noise = np.random.normal(0, self.sigma, size=(n_mut, d))
        
        # Apply noise only where mask is True
        offspring[mask] += noise[mask]
        
        # Enforce bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        return np.clip(offspring, lb, ub)