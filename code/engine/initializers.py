import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import qmc

class PopulationInitializer(ABC):
    """Abstract base class for population initialization strategies."""
    
    @abstractmethod
    def initialize(self, pop_size: int, bounds: list) -> np.ndarray:
        """
        Generates the initial population matrix.
        
        Args:
            pop_size (int): Number of individuals (N).
            bounds (list): List of tuples[(lb1, ub1), (lb2, ub2), ...] for D dimensions.
            
        Returns:
            np.ndarray: Matrix of shape (N, D) containing the initial population.
        """
        pass

class RandomInitializer(PopulationInitializer):
    """Generates initial population using uniform random distribution."""
    
    def initialize(self, pop_size: int, bounds: list) -> np.ndarray:
        d = len(bounds)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        # r is a matrix of shape (pop_size, d) with values in[0, 1)
        r = np.random.rand(pop_size, d)
        
        # Scale to bounds: x = lb + r * (ub - lb)
        return lb + r * (ub - lb)

class HaltonInitializer(PopulationInitializer):
    """Generates initial population using the Halton quasi-random sequence."""
    
    def initialize(self, pop_size: int, bounds: list) -> np.ndarray:
        d = len(bounds)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        # Using SciPy's Quasi-Monte Carlo module for the Halton sequence
        # Scramble=True breaks up obvious periodic patterns in higher dimensions
        sampler = qmc.Halton(d=d, scramble=True)
        sample = sampler.random(n=pop_size)
        
        # qmc.scale applies the exact lb + r * (ub - lb) math internally
        return qmc.scale(sample, lb, ub)