import numpy as np
import sympy as sp
from typing import Callable, Optional, List, Dict, Any
from engine.utils import logger

class OptimisationResults:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []                 # Changed: List of dicts e.g., {"x": np.array, "subgrad": bool}
        self.final_population: Optional[np.ndarray] = None      # Final coordinates
        self.final_f: Optional[float] = None                    # Function value at final_x
        self.execution_time: float = 0.0
        self.iterations: int = 0
        self.converged: bool = False

class TargetFunction:
    def __init__(self, expression: str | Callable, bounds: list, grad_func: Callable | None = None, hessian_func: Callable | None = None):
        self.bounds = bounds
        self.expression_str = None
        self.sympy_expr = None
        self.sympy_grad = None
        self.grad_func = grad_func
        self.hessian_func = hessian_func # Optional user-provided or auto-calculated Hessian

        if isinstance(expression, str):
            self.expression_str = expression
            self._parse_string(expression)
            logger.info(f"Target initialized: {expression}. Gradient/Hessian source: Analytical (sympy).")
        elif callable(expression):
            self.callable_func = expression
            self.grad_func = grad_func
            self.hessian_func = hessian_func
            if not bounds:
                raise ValueError("Bounds must be provided to determine dimensionality of a callable function.")
            self.variables =[f"x{i+1}" for i in range(len(bounds))]
            
            grad_source = "User-provided callable" if grad_func else "Numerical approximation fallback"
            logger.info(f"Target initialized: Callable. Gradient source: {grad_source}.")
        else:
            raise TypeError("Expression must be a string or a callable function.")

    def _parse_string(self, expr_str: str):
        try:
            self.sympy_expr = sp.sympify(expr_str)
        except Exception as e:
            logger.debug(f"Failed to parse '{expr_str}': {e}")
            raise ValueError(f"Invalid expression: {expr_str}")
        
        self.variables = sorted(list(self.sympy_expr.free_symbols), key=lambda s: s.name)
        self.callable_func = sp.lambdify([self.variables], self.sympy_expr, modules="numpy")
        
        # Gradient
        try:
            sympy_grad = [sp.diff(self.sympy_expr, var) for var in self.variables]
            self.grad_func = sp.lambdify([self.variables], sympy_grad, modules="numpy")
        except Exception as e:
            logger.warning(f"Analytical gradient not available for '{expr_str}': {e}. Using numerical fallback.")
            self.grad_func = None
            
        # Hessian
        try:
            sympy_hessian = sp.hessian(self.sympy_expr, self.variables)
            self.hessian_func = sp.lambdify([self.variables], sympy_hessian, modules="numpy")
        except Exception as e:
            logger.warning(f"Analytical Hessian not available for '{expr_str}': {e}. Using numerical fallback.")
            self.hessian_func = None
        
    def evaluate(self, x):
        try:
            return self.callable_func(*x)
        except TypeError:
            return self.callable_func(x)
            
    def evaluate_gradient(self, x, h=1e-5):
        is_subgrad = False
        grad = None

        if self.grad_func:
            try:
                grad = np.array(self.grad_func(*x), dtype=float)
            except TypeError:
                grad = np.array(self.grad_func(x), dtype=float)
                
            # Check if analytical derivative failed (e.g., NaN at a kink)
            if not np.isnan(grad).any():
                return grad, is_subgrad

        # Fallback: Numerical approximation handling subgradients
        is_subgrad = True 
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += h
            x_backward[i] -= h
            
            # Try central difference first
            central_diff = (self.evaluate(x_forward) - self.evaluate(x_backward)) / (2 * h)
            
            # If exactly 0, it might be a symmetric kink. Fallback to forward difference.
            if central_diff == 0.0:
                logger.debug(f"Subgradient detected at dimension {i}. Using forward difference.")
                grad[i] = (self.evaluate(x_forward) - self.evaluate(x)) / h
            else:
                grad[i] = central_diff
                
        return grad, is_subgrad
    
    def evaluate_hessian(self, x, h=1e-4):
        x = np.array(x, dtype=float)
        n = len(x)

        if self.hessian_func:
            try:
                H = np.array(self.hessian_func(*x), dtype=float)
            except TypeError:
                H = np.array(self.hessian_func(x), dtype=float)
                
            # Handle SymPy returning a scalar 0 for flat/linear functions
            if np.isscalar(H) or H.ndim == 0:
                H = np.full((n, n), float(H)) # type: ignore
                
            if not np.isnan(H).any() and H.shape == (n, n):
                return H

        # Fallback: Numerical approximation of the Hessian
        H = np.zeros((n, n), dtype=float)
        f_x = self.evaluate(x)
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Diagonal elements: central difference for second derivative
                    x_fwd = x.copy()
                    x_bwd = x.copy()
                    x_fwd[i] += h
                    x_bwd[i] -= h
                    H[i, i] = (self.evaluate(x_fwd) - 2 * f_x + self.evaluate(x_bwd)) / (h**2)
                else:
                    # Off-diagonal elements: mixed partial derivatives
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h; x_pp[j] += h
                    x_pm[i] += h; x_pm[j] -= h
                    x_mp[i] -= h; x_mp[j] += h
                    x_mm[i] -= h; x_mm[j] -= h
                    
                    H[i, j] = (self.evaluate(x_pp) - self.evaluate(x_pm) - self.evaluate(x_mp) + self.evaluate(x_mm)) / (4 * h**2)
                    H[j, i] = H[i, j] # Hessian matrix is always symmetric
                    
        return H
    
    def check_convexity(self, samples=5):
        """
        Checks if the function is strictly convex within bounds by sampling the Hessian.
        Returns (is_convex: bool, counter_example_point: np.ndarray | None).
        """
        try:
            # Create sampling grid based on self.bounds
            grids =[np.linspace(b[0], b[1], samples) for b in self.bounds]
            
            # Create the mesh and reshape into a list of D-dimensional points
            mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(self.variables))
            
            for point in mesh:
                h_matrix = self.evaluate_hessian(point)
                
                # Check eigenvalues. Strict convexity requires all eigenvalues > 0.
                # We use 1e-9 instead of 0 to account for floating point inaccuracies.
                eigenvalues = np.linalg.eigvals(h_matrix)
                if np.any(eigenvalues < 1e-9): 
                    return False, point
                    
            return True, None
            
        except Exception as e:
            logger.error(f"Convexity check failed: {e}")
            return None, None