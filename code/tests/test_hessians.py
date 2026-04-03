import numpy as np
from engine.models import TargetFunction

def test_quadratic_analytical():
    print("--- Test 1: Simple Quadratic (Analytical) ---")
    # f(x, y) = x^2 + y^2
    # H = [[2, 0], [0, 2]] everywhere
    tf = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    
    x_test =[3.0, -2.0]
    H = tf.evaluate_hessian(x_test)
    
    expected_H = np.array([[2.0, 0.0],[0.0, 2.0]])
    print(f"Point: {x_test}")
    print(f"Calculated Hessian:\n{H}")
    print(f"Matches Expected: {np.allclose(H, expected_H)}\n")

def test_mixed_terms_analytical():
    print("--- Test 2: Mixed Terms (Analytical) ---")
    # f(x, y) = x^2 * y
    # H_xx = 2y
    # H_yy = 0
    # H_xy = H_yx = 2x
    tf = TargetFunction("x**2 * y", bounds=[(-5, 5), (-5, 5)])
    
    x_test = [1.0, 2.0]
    H = tf.evaluate_hessian(x_test)
    
    # At (1, 2): H_xx = 4, H_yy = 0, H_xy = 2
    expected_H = np.array([[4.0, 2.0], [2.0, 0.0]])
    print(f"Point: {x_test}")
    print(f"Calculated Hessian:\n{H}")
    print(f"Matches Expected: {np.allclose(H, expected_H)}\n")

def test_callable_numerical():
    print("--- Test 3: Callable Function (Numerical Fallback) ---")
    # Same function as Test 2, but provided as a callable to force numerical approximation
    def func(x):
        return x[0]**2 * x[1]
        
    tf = TargetFunction(func, bounds=[(-5, 5), (-5, 5)])
    
    x_test = [1.0, 2.0]
    H = tf.evaluate_hessian(x_test)
    
    expected_H = np.array([[4.0, 2.0], [2.0, 0.0]])
    print(f"Point: {x_test}")
    print(f"Calculated Hessian:\n{H}")
    # We use a slightly looser tolerance for numerical derivatives
    print(f"Matches Expected: {np.allclose(H, expected_H, atol=1e-4)}\n")

def test_linear_scalar_handling():
    print("--- Test 4: Linear Function (Scalar/Zero Handling) ---")
    # f(x, y) = 2x + 3y
    # H = [[0, 0], [0, 0]]
    # SymPy might return a single scalar 0 for this, so we test the shape expansion logic
    tf = TargetFunction("2*x + 3*y", bounds=[(-5, 5), (-5, 5)])
    
    x_test = [1.0, 1.0]
    H = tf.evaluate_hessian(x_test)
    
    expected_H = np.zeros((2, 2))
    print(f"Point: {x_test}")
    print(f"Calculated Hessian:\n{H}")
    print(f"Matches Expected: {np.allclose(H, expected_H)}\n")

if __name__ == "__main__":
    import logging
    logging.getLogger("engine.utils").setLevel(logging.WARNING) # Hide initialization spam
    
    test_quadratic_analytical()
    test_mixed_terms_analytical()
    test_callable_numerical()
    test_linear_scalar_handling()