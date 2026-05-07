import numpy as np
from abc import ABC, abstractmethod

class CrossoverStrategy(ABC):
    """Abstract base class for crossover strategies."""
    @abstractmethod
    def crossover(self, parents: np.ndarray, parent_fitnesses: np.ndarray, n_cross: int, bounds: list) -> np.ndarray:
        """
        Args:
            parents: Matrix of shape (2 * n_cross, D). Rows[:n_cross] are parent A,
                rows [n_cross:] are parent B (paired by row index).
            parent_fitnesses: 1D array of shape (2 * n_cross,) aligned with `parents`.
                Lower = better (the project minimizes throughout).
            n_cross: Number of children to generate.
            bounds: List of (min, max) per dimension to ensure children stay in bounds.
        """
        pass

class UniformCrossover(CrossoverStrategy):
    """
    Real-coded uniform crossover.
    z_i = beta * x_i + (1 - beta) * y_i, with beta ~ U[-d, 1+d] per coordinate.
    Fitness is unused — uniform crossover is fitness-agnostic.
    """
    def __init__(self, d: float = 0.25):
        self.d = d

    def crossover(self, parents: np.ndarray, parent_fitnesses: np.ndarray, n_cross: int, bounds: list) -> np.ndarray:
        parent1 = parents[:n_cross]
        parent2 = parents[n_cross:]

        beta = np.random.uniform(-self.d, 1 + self.d, size=parent1.shape)
        offspring = beta * parent1 + (1 - beta) * parent2

        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        return np.clip(offspring, lb, ub)

class NonUniformCrossover(CrossoverStrategy):
    """
    Real-coded non-uniform crossover (EvMO_KP1, p. 12).

    Same potential offspring region as UniformCrossover, but biased toward
    the fitter parent.

        z_i = x_i + alpha_i * (y_i - x_i)

    with beta_i ~ U[0, 1] per coordinate and:

        alpha_i = -d + (1 + 2d) * beta_i ** s        if f(X) < f(Y)   (X better)
        alpha_i = -d + (1 + 2d) * beta_i             if f(X) == f(Y)
        alpha_i = -d + (1 + 2d) * beta_i ** (1/s)    if f(X) > f(Y)   (Y better)

    s > 1 controls how strongly children resemble the better parent.
    """
    def __init__(self, s: float = 2.0, d: float = 0.25):
        if s <= 1:
            raise ValueError(f"s must be > 1 (got {s}); s controls bias strength toward the better parent.")
        self.s = s
        self.d = d

    def crossover(self, parents: np.ndarray, parent_fitnesses: np.ndarray, n_cross: int, bounds: list) -> np.ndarray:
        parent1 = parents[:n_cross]
        parent2 = parents[n_cross:]
        f1 = parent_fitnesses[:n_cross]
        f2 = parent_fitnesses[n_cross:]

        # Per-pair exponent on beta. Lower fitness = better parent.
        exponents = np.where(f1 < f2, self.s,
                             np.where(f1 > f2, 1.0 / self.s, 1.0))

        beta = np.random.uniform(0.0, 1.0, size=parent1.shape)
        alpha = -self.d + (1 + 2 * self.d) * (beta ** exponents[:, None])

        offspring = parent1 + alpha * (parent2 - parent1)

        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        return np.clip(offspring, lb, ub)