import logging
import numpy as np
import sympy as sp

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_convexity(expression_str, variables, bounds, samples=5):
    """
    Checks if a function is convex within bounds by sampling the Hessian.
    """
    try:
        vars_sym = sp.symbols(variables)
        f = sp.sympify(expression_str)
        hessian = sp.hessian(f, vars_sym)
        hessian_func = sp.lambdify(vars_sym, hessian, 'numpy')
        
        # Create sampling grid
        grids = [np.linspace(b[0], b[1], samples) for b in bounds]
        mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(variables))
        
        for point in mesh:
            h_matrix = np.array(hessian_func(*point))
            eigenvalues = np.linalg.eigvals(h_matrix)
            if np.any(eigenvalues < -1e-9): # Small epsilon for float noise
                return False, point
        return True, None
    except Exception as e:
        logger.error(f"Convexity check failed: {e}")
        return None, None