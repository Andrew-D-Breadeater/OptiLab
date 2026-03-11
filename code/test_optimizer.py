import numpy as np
from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent

def run_test(name, **kwargs):
    print(f"\n--- {name} ---")
    opt = GradientDescent(**kwargs)
    opt.run(max_iter=500, tol=1e-4)

# 1. String defined function
target_str = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
run_test("Test 1: String Defined", 
         target_function=target_str, 
         start_pos=np.array([1.0, 1.0]), 
         learning_rate=0.1, 
         stopping_criterion='gradient_norm')

# 2. Same function but as a callable (no analytical gradient)
def func_call(x): return x[0]**2 + x[1]**2
target_call = TargetFunction(func_call, bounds=[(-5, 5), (-5, 5)])
run_test("Test 2: Callable (Numerical Gradient)", 
         target_function=target_call, 
         start_pos=np.array([1.0, 1.0]), 
         learning_rate=0.1, 
         stopping_criterion='gradient_norm')

# 3. Ravine method with Backtracking
target_ravine = TargetFunction("10*x**2 + y**2", bounds=[(-10, 10), (-10, 10)])
run_test("Test 3: Ravine + Backtracking", 
         target_function=target_ravine, 
         start_pos=np.array([5.0, 5.0]), 
         learning_rate=0.05, 
         use_ravine=True, 
         use_line_search=True)

# 4. Subgradient test (kink at 0)
def func_kink(x): return np.abs(x[0]) + np.abs(x[1])
target_kink = TargetFunction(func_kink, bounds=[(-5, 5), (-5, 5)])
run_test("Test 4: Subgradient (Kink)", 
         target_function=target_kink, 
         start_pos=np.array([3.0, 0.0]), 
         learning_rate=0.1)