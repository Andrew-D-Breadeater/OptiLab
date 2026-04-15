import numpy as np
import sympy as sp
from scipy.optimize import minimize
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

    @abstractmethod
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of the same shape as X and Y.
        True = Inside the allowed boundary.
        False = Outside the boundary (forbidden).
        """
        pass

class NoProjection(ProjectionStrategy):
    def project(self, a: np.ndarray) -> np.ndarray:
        return a
        
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.ones_like(X, dtype=bool)

class NonNegativeProjection(ProjectionStrategy):
    def project(self, a: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, a)
        
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X >= 0) & (Y >= 0)

class BoxProjection(ProjectionStrategy):
    def __init__(self, bounds: list):
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds], dtype=float)
        self.ub = np.array([b[1] for b in bounds], dtype=float)

    def project(self, a: np.ndarray) -> np.ndarray:
        return np.clip(a, self.lb, self.ub)
        
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X >= self.lb[0]) & (X <= self.ub[0]) & (Y >= self.lb[1]) & (Y <= self.ub[1])

class HyperplaneProjection(ProjectionStrategy):
    def __init__(self, c: list | np.ndarray, b: float):
        self.c = np.array(c, dtype=float)
        self.b = float(b)
        self.c_norm_sq = np.dot(self.c, self.c)
        if self.c_norm_sq == 0: raise ValueError("Normal vector 'c' cannot be a zero vector.")

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        dot_ca = np.dot(self.c, a)
        return a - ((dot_ca - self.b) / self.c_norm_sq) * self.c
        
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # A hyperplane is a thin line. For visibility on a grid, we give it a tiny thickness tolerance.
        tol = (X.max() - X.min()) / 300.0 * 1.5
        return np.abs(self.c[0]*X + self.c[1]*Y - self.b) <= tol

class HalfSpaceProjection(ProjectionStrategy):
    def __init__(self, c: list | np.ndarray, b: float):
        self.c = np.array(c, dtype=float)
        self.b = float(b)
        self.c_norm_sq = np.dot(self.c, self.c)
        if self.c_norm_sq == 0: raise ValueError("Normal vector 'c' cannot be a zero vector.")

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        violation = max(0.0, np.dot(self.c, a) - self.b)
        return a - (violation / self.c_norm_sq) * self.c

    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (self.c[0]*X + self.c[1]*Y) <= self.b

class SphereProjection(ProjectionStrategy):
    def __init__(self, center: list | np.ndarray, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        if self.radius <= 0: raise ValueError("Radius must be strictly greater than 0.")

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        dist = np.linalg.norm(a - self.center)
        if dist <= self.radius: return a
        return self.center + self.radius * (a - self.center) / dist
        
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X - self.center[0])**2 + (Y - self.center[1])**2 <= self.radius**2

class CustomNonlinearProjection(ProjectionStrategy):
    def __init__(self, constraint_strs: list, variables: list):
        self.constraints =[]
        self.raw_funcs =[] # Save raw lambdified funcs for the 2D meshgrid evaluator
        self.variables_sym = [sp.Symbol(v) for v in variables]
        
        for c_str in constraint_strs:
            c_str = c_str.strip()
            if not c_str: continue
                
            if '>=' in c_str:
                lhs, rhs = c_str.split('>=')
                expr = sp.sympify(lhs) - sp.sympify(rhs)
                ctype = 'ineq'
            elif '<=' in c_str:
                lhs, rhs = c_str.split('<=')
                expr = sp.sympify(rhs) - sp.sympify(lhs)
                ctype = 'ineq'
            elif '==' in c_str:
                lhs, rhs = c_str.split('==')
                expr = sp.sympify(lhs) - sp.sympify(rhs)
                ctype = 'eq'
            else:
                expr = sp.sympify(c_str)
                ctype = 'ineq'
            
            func = sp.lambdify([self.variables_sym], expr, modules="numpy")
            self.constraints.append({'type': ctype, 'fun': lambda x, f=func: f(x)})
            self.raw_funcs.append({'type': ctype, 'fun': func})

    def project(self, a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=float)
        
        # 1. Quick check: If the point is already inside the feasible region, do nothing
        is_inside = True
        for constr in self.constraints:
            val = constr['fun'](a)
            if constr['type'] == 'ineq' and val < -1e-6: is_inside = False; break
            elif constr['type'] == 'eq' and abs(val) > 1e-6: is_inside = False; break
                
        if is_inside: return a

        def objective(x): return np.sum((x - a)**2)
        def jacobian(x): return 2 * (x - a)

        res = minimize(objective, x0=a, method='SLSQP', jac=jacobian, constraints=self.constraints, options={'ftol': 1e-6, 'disp': False})
        return res.x
        
    def get_feasibility_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        mask = np.ones_like(X, dtype=bool)
        for constr in self.raw_funcs:
            try:
                # Evaluate the non-linear math over the entire 100x100 grid
                Z = constr['fun']([X, Y])
                if np.isscalar(Z):
                    Z = np.full_like(X, Z)
                
                if constr['type'] == 'ineq':
                    mask &= (Z >= -1e-6)
                else:
                    tol = (X.max() - X.min()) / 300.0 * 1.5
                    mask &= (np.abs(Z) <= tol)
            except Exception as e:
                pass # Fail silently if the user typed invalid math for the grid
        return mask