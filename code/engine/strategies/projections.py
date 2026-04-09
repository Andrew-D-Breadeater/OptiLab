import numpy as np
from abc import ABC, abstractmethod

class ProjectionStrategy(ABC):
    """Abstract base class for boundary projection strategies."""
    @abstractmethod
    def project(self, a: np.ndarray) -> np.ndarray:
        """
        Projects a point 'a' onto the admissible set X.
        
        Args:
            a (np.ndarray): The point to project (1D array).
            
        Returns:
            np.ndarray: The projected point inside or on the boundary of set X.
        """
        pass

class NoProjection(ProjectionStrategy):
    """Default strategy for unconstrained optimization."""
    def project(self, a: np.ndarray) -> np.ndarray:
        return a

class NonNegativeProjection(ProjectionStrategy):
    """
    Nonnegative orthogonal projection.
    X = {x | x_i >= 0}
    """
    def project(self, a: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, a)

class BoxProjection(ProjectionStrategy):
    """
    Coordinate parallelepiped projection.
    X = {x | alpha_i <= x_i <= beta_i}
    """
    def __init__(self, bounds: list):
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds], dtype=float)
        self.ub = np.array([b[1] for b in bounds], dtype=float)

    def project(self, a: np.ndarray) -> np.ndarray:
        # np.clip efficiently handles the mid(alpha, a, beta) logic
        return np.clip(a, self.lb, self.ub)

class HyperplaneProjection(ProjectionStrategy):
    """
    Hyperplane projection.
    X = {x | <c, x> = b}
    """
    def __init__(self, c: list | np.ndarray, b: float):
        self.c = np.array(c, dtype=float)
        self.b = float(b)
        self.c_norm_sq = np.dot(self.c, self.c)
        
        if self.c_norm_sq == 0:
            raise ValueError("Normal vector 'c' cannot be a zero vector.")

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        dot_ca = np.dot(self.c, a)
        return a - ((dot_ca - self.b) / self.c_norm_sq) * self.c

class HalfSpaceProjection(ProjectionStrategy):
    """
    Half-space projection.
    X = {x | <c, x> <= b}
    """
    def __init__(self, c: list | np.ndarray, b: float):
        self.c = np.array(c, dtype=float)
        self.b = float(b)
        self.c_norm_sq = np.dot(self.c, self.c)
        
        if self.c_norm_sq == 0:
            raise ValueError("Normal vector 'c' cannot be a zero vector.")

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        dot_ca = np.dot(self.c, a)
        # Only project if the point violates the inequality
        violation = max(0.0, dot_ca - self.b)
        return a - (violation / self.c_norm_sq) * self.c

class SphereProjection(ProjectionStrategy):
    """
    Sphere/Ball projection.
    X = {x | ||x - x0|| <= R}
    """
    def __init__(self, center: list | np.ndarray, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        
        if self.radius <= 0:
            raise ValueError("Radius must be strictly greater than 0.")

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        dist = np.linalg.norm(a - self.center)
        
        if dist <= self.radius:
            return a
        
        # Pull the point onto the surface of the sphere
        return self.center + self.radius * (a - self.center) / dist