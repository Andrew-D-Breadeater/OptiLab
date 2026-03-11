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
    def __init__(self, expression: str | Callable, bounds: list, grad_func: Callable | None = None):
        self.bounds = bounds
        self.expression_str = None # Store original string expression for reference
        self.sympy_expr = None # Temporary storage for symbolic expression
        self.sympy_grad = None # Temporary storage for symbolic gradient
        self.grad_func = grad_func # Optional user-provided or automatically calculated gradient function

        if isinstance(expression, str):
            self.expression_str = expression
            self._parse_string(expression)
            logger.info(f"Target initialized: {expression}. Gradient source: Analytical (sympy).")
        elif callable(expression):
            self.callable_func = expression
            self.grad_func = grad_func  # User must provide grad_func if expression is a callable
            # Infer dimensionality from bounds: e.g., bounds=[(-5,5), (-5,5)] -> ['x1', 'x2']
            if not bounds:
                raise ValueError("Bounds must be provided to determine dimensionality of a callable function.")
            self.variables = [f"x{i+1}" for i in range(len(bounds))]
            
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
        
        # Extract and sort variables alphabetically so array inputs map correctly
        self.variables = sorted(list(self.sympy_expr.free_symbols), key=lambda s: s.name)
        
        # Convert symbolic expression to a fast numpy callable
        self.callable_func = sp.lambdify([self.variables], self.sympy_expr, modules="numpy")
        
        # Calculate analytical gradient vector ∇f
        self.sympy_grad = [sp.diff(self.sympy_expr, var) for var in self.variables]
        
        # Convert symbolic gradient to a fast numpy callable
        self.grad_func = sp.lambdify([self.variables], self.sympy_grad, modules="numpy")
        
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