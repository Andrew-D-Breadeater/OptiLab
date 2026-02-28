import numpy as np
from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent

def test_string_parsing_and_standard_gd():
    print("--- Test 1: String parsing, standard GD ---")
    # Function: f(x, y) = x^2 + y^2
    target = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([4.0, 4.0]), 
        learning_rate=0.1
    )
    results = opt.run(max_iter=100)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}\n")

def test_callable_with_line_search():
    print("--- Test 2: Callable without analytical gradient, Backtracking Line Search ---")
    # Function: f(x, y) = (x - 2)^2 + (y - 3)^2
    def func(x):
        return (x[0] - 2)**2 + (x[1] - 3)**2
    
    # Passing bounds to infer 2 variables. Will trigger numerical gradient fallback.
    target = TargetFunction(func, bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([0.0, 0.0]), 
        use_line_search=True
    )
    results = opt.run(max_iter=100)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}")
    
    # Verify subgradient/numerical flag was tracked
    subgrad_uses = sum(1 for step in results.history if step["subgrad"])
    print(f"Numerical/Subgradient steps taken: {subgrad_uses}\n")

def test_ravine_method():
    print("--- Test 3: String parsing, Ravine Method ---")
    # Function: f(x, y) = 10*x^2 + y^2 (Elongated valley to test ravine jumps)
    target = TargetFunction("10*x**2 + y**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([5.0, 5.0]), 
        learning_rate=0.05, 
        use_ravine=True,
        use_line_search=True,
        ravine_step_size=0.8
    )
    results = opt.run(max_iter=500)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}\n")
    
def test_callable_ravine():
    print("--- Test 4: Callable, Ravine Method ---")
    def func(x):
        return 10 * x[0]**2 + x[1]**2
        
    target = TargetFunction(func, bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([5.0, 5.0]), 
        learning_rate=0.05, 
        use_ravine=True, 
        ravine_step_size=0.8
    )
    results = opt.run(max_iter=500)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}\n")

def test_callable_no_ravine():
    print("--- Test 5: Callable, Standard GD (No Ravine) ---")
    def func(x):
        return 10 * x[0]**2 + x[1]**2
        
    target = TargetFunction(func, bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([5.0, 5.0]), 
        learning_rate=0.05, 
        use_ravine=False
    )
    results = opt.run(max_iter=500)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}\n")
    
def test_exact_line_search():
    print("--- Test 6: Exact Line Search ---")
    target = TargetFunction("x**2 + 5*y**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([8.0, 8.0]), 
        use_exact_line_search=True
    )
    results = opt.run(max_iter=500)
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}\n")

def test_string_vs_callable():
    print("--- Test 7: String vs Callable Evaluation Match ---")
    tf_str = TargetFunction("x**2 + 3*y**2", bounds=[(-5, 5), (-5, 5)])
    
    def func(x):
        return x[0]**2 + 3*x[1]**2
        
    tf_call = TargetFunction(func, bounds=[(-5, 5), (-5, 5)])
    
    test_point = np.array([2.0, -1.5])
    
    val_str = tf_str.evaluate(test_point)
    val_call = tf_call.evaluate(test_point)
    
    grad_str, _ = tf_str.evaluate_gradient(test_point)
    grad_call, _ = tf_call.evaluate_gradient(test_point)
    
    print(f"Value match: {np.isclose(val_str, val_call)} ({val_str} vs {val_call})")
    print(f"Gradient match: {np.allclose(grad_str, grad_call)} ({grad_str} vs {grad_call})\n")

def test_subgradients():
    print("--- Test 8: Subgradient Handling (Kinks) ---")
    # f(x, y) = |x| + |y|
    def func(x):
        return np.abs(x[0]) + np.abs(x[1])
        
    target = TargetFunction(func, bounds=[(-5, 5), (-5, 5)])
    
    # Starting exactly at y=0 forces central difference to evaluate to 0.0
    # This must trigger our subgradient forward-difference resolution
    opt = GradientDescent(
        target, 
        start_pos=np.array([3.0, 0.0]), 
        learning_rate=0.5,        
        decay_rate=0.9 # Decay to help with convergence around the kink,
    )
    results = opt.run(max_iter=100)
    
    subgrad_uses = sum(1 for step in results.history if step.get("subgrad", False))
    
    print(f"Converged: {results.converged} in {results.iterations} iters")
    print(f"Final x: {results.final_x}")
    print(f"Final f(x): {results.final_f}\n")
    print(f"Subgradient steps triggered: {subgrad_uses}\n")

  

if __name__ == "__main__":
    test_string_parsing_and_standard_gd()
    test_callable_with_line_search()
    test_ravine_method()
    test_callable_ravine()
    test_callable_no_ravine()
    test_exact_line_search()
    test_string_vs_callable()
    test_subgradients()