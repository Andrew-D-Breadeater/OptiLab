import numpy as np
from engine.models import TargetFunction
from engine.optimizers.newton_methods import NewtonOptimizer

def get_x(results):
    return results.final_population[0]

def test_quadratic_1_step():
    print("--- Test 1: Perfect Quadratic (1-Step Convergence) ---")
    # f(x, y) = x^2 + y^2
    # Newton's method should find the minimum of a pure quadratic in exactly 1 step with alpha=1.0
    target = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    opt = NewtonOptimizer(
        target, 
        start_pos=np.array([4.0, 4.0]), 
        learning_rate=1.0 # Standard Newton step
    )
    results = opt.run(max_iter=100)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {get_x(results)}")
    print(f"Final f(x): {results.final_f}\n")

def test_rosenbrock():
    print("--- Test 2: Rosenbrock (Non-trivial Convergence) ---")
    # f(x, y) = (1-x)^2 + 100(y-x^2)^2
    # Standard benchmark. Newton requires multiple steps here.
    target = TargetFunction("(1 - x)**2 + 100 * (y - x**2)**2", bounds=[(-5, 5), (-5, 5)])
    opt = NewtonOptimizer(
        target, 
        start_pos=np.array([-1.2, 1.0]), 
        learning_rate=1.0
    )
    results = opt.run(max_iter=100)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {get_x(results)}")
    print(f"Final f(x): {results.final_f}\n")

def test_singular_hessian_error():
    print("--- Test 3: Singular Hessian (Error Handling) ---")
    # f(x, y) = x + y
    # Hessian is [[0, 0], [0, 0]]. It is uninvertible (singular).
    target = TargetFunction("x + y", bounds=[(-5, 5), (-5, 5)])
    opt = NewtonOptimizer(
        target, 
        start_pos=np.array([0.0, 0.0])
    )
    
    try:
        opt.run(max_iter=10)
        print("FAIL: The optimizer ran, but it was supposed to crash!")
    except RuntimeError as e:
        print("SUCCESS: Caught the expected error.")
        print(f"Error message: {e}\n")

def test_line_search_integration():
    print("--- Test 4: Newton with Backtracking Line Search ---")
    target = TargetFunction("x**4 + y**4", bounds=[(-5, 5), (-5, 5)])
    opt = NewtonOptimizer(
        target, 
        start_pos=np.array([2.0, 2.0]), 
        use_line_search=True
    )
    results = opt.run(max_iter=100)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {get_x(results)}")
    print(f"Final f(x): {results.final_f}\n")


if __name__ == "__main__":
    import logging
    # Silence engine logs for cleaner test output
    logging.getLogger("engine.utils").setLevel(logging.WARNING) 
    
    test_quadratic_1_step()
    test_rosenbrock()
    test_singular_hessian_error()
    test_line_search_integration()