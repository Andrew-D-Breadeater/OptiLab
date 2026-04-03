import numpy as np
from engine.models import TargetFunction

def test_strictly_convex():
    print("--- Test 1: Strictly Convex (Paraboloid) ---")
    # f(x, y) = x^2 + y^2. 
    # Hessian is [[2, 0],[0, 2]]. Eigenvalues are 2, 2. 
    tf = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    is_convex, counter_point = tf.check_convexity()
    print(f"Is convex: {is_convex} (Expected: True)")
    print(f"Counter example point: {counter_point}\n")

def test_concave():
    print("--- Test 2: Concave (Inverted Paraboloid) ---")
    # f(x, y) = -x^2 - y^2. 
    # Hessian is [[-2, 0], [0, -2]]. Eigenvalues are -2, -2.
    tf = TargetFunction("-x**2 - y**2", bounds=[(-5, 5), (-5, 5)])
    is_convex, counter_point = tf.check_convexity()
    print(f"Is convex: {is_convex} (Expected: False)")
    print(f"Counter example point: {counter_point}\n")

def test_saddle_point():
    print("--- Test 3: Saddle Point (Hyperbolic Paraboloid) ---")
    # f(x, y) = x^2 - y^2. 
    # Hessian is [[2, 0], [0, -2]]. Eigenvalues are 2, -2.
    tf = TargetFunction("x**2 - y**2", bounds=[(-5, 5), (-5, 5)])
    is_convex, counter_point = tf.check_convexity()
    print(f"Is convex: {is_convex} (Expected: False)")
    print(f"Counter example point: {counter_point}\n")

def test_linear():
    print("--- Test 4: Linear Function (Not STRICTLY convex) ---")
    # f(x, y) = 2x + 3y. 
    # Hessian is [[0, 0],[0, 0]]. Eigenvalues are 0, 0.
    # Since 0 is not > 1e-9, it fails the strict convexity test.
    tf = TargetFunction("2*x + 3*y", bounds=[(-5, 5), (-5, 5)])
    is_convex, counter_point = tf.check_convexity()
    print(f"Is convex: {is_convex} (Expected: False)")
    print(f"Counter example point: {counter_point}\n")

def test_callable_rosenbrock():
    print("--- Test 5: Rosenbrock (Non-convex callable) ---")
    # Classic optimization test function. It is non-convex overall (has a curved valley).
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
    tf = TargetFunction(rosenbrock, bounds=[(-2, 2), (-2, 2)])
    is_convex, counter_point = tf.check_convexity(samples=10)
    print(f"Is convex: {is_convex} (Expected: False)")
    print(f"Counter example point: {counter_point}\n")

if __name__ == "__main__":
    import logging
    # Silence the target initialization logs for cleaner output
    logging.getLogger("engine.utils").setLevel(logging.WARNING) 
    
    test_strictly_convex()
    test_concave()
    test_saddle_point()
    test_linear()
    test_callable_rosenbrock()