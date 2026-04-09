import numpy as np
from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent
from engine.optimizers.newton_methods import NewtonOptimizer
from engine.strategies.projections import (
    NoProjection, NonNegativeProjection, BoxProjection, 
    HyperplaneProjection, HalfSpaceProjection, SphereProjection
)

def get_x(results):
    return np.round(results.final_population[0], 4)

def test_unconstrained_newton():
    print("--- Test 1: Newton's Method (No Projection) ---")
    target = TargetFunction("x**2 + y**2", bounds=[(-10, 10), (-10, 10)])
    opt = NewtonOptimizer(target, start_pos=np.array([4.0, 4.0]), learning_rate=1.0)
    res = opt.run(max_iter=50)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected: [0. 0.])\n")

def test_unconstrained_gd():
    print("--- Test 2: Gradient Descent (No Projection) ---")
    target = TargetFunction("x**2 + y**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(target, start_pos=np.array([4.0, 4.0]), learning_rate=0.1)
    res = opt.run(max_iter=100)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected: [0. 0.])\n")

def test_nonnegative_projection():
    print("--- Test 3: GD + Non-Negative Projection ---")
    # Unconstrained min is at (-3, -3). 
    # With x_i >= 0, the closest allowable point is (0, 0).
    target = TargetFunction("(x+3)**2 + (y+3)**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([2.0, 2.0]), 
        learning_rate=0.1,
        projection_strategy=NonNegativeProjection()
    )
    res = opt.run(max_iter=100)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected:[0. 0.])\n")

def test_box_projection():
    print("--- Test 4: GD + Box Projection (Coordinate Parallelepiped) ---")
    # Unconstrained min is at (5, 5).
    # Box restricts to [-2, 2]. The constrained min should hit the corner at (2, 2).
    target = TargetFunction("(x-5)**2 + (y-5)**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([0.0, 0.0]), 
        learning_rate=0.1,
        projection_strategy=BoxProjection([(-2, 2), (-2, 2)])
    )
    res = opt.run(max_iter=100)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected: [2. 2.])\n")

def test_hyperplane_projection():
    print("--- Test 5: GD + Hyperplane Projection ---")
    # Unconstrained min is at (0, 0).
    # Constraint line: x + y = 4 (Normal c=[1, 1], b=4).
    # The closest point on this line to (0,0) is (2, 2).
    target = TargetFunction("x**2 + y**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([5.0, 5.0]), 
        learning_rate=0.1,
        projection_strategy=HyperplaneProjection(c=[1, 1], b=4.0)
    )
    res = opt.run(max_iter=100)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected: [2. 2.])\n")

def test_halfspace_projection():
    print("--- Test 6: GD + Half-Space Projection ---")
    # Unconstrained min is at (4, 4).
    # Constraint: x + y <= 2 (Normal c=[1, 1], b=2).
    # Since (4,4) violates x+y<=2, it should project onto the boundary at (1, 1).
    target = TargetFunction("(x-4)**2 + (y-4)**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([0.0, 0.0]), # Start inside valid space
        learning_rate=0.1,
        projection_strategy=HalfSpaceProjection(c=[1, 1], b=2.0)
    )
    res = opt.run(max_iter=100)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected: [1. 1.])\n")

def test_sphere_projection():
    print("--- Test 7: GD + Sphere/Ball Projection ---")
    # Unconstrained min is at (5, 0).
    # Constraint: Sphere centered at (0,0) with radius 2.
    # The optimizer should travel along the X-axis and get stopped by the sphere at (2, 0).
    target = TargetFunction("(x-5)**2 + y**2", bounds=[(-10, 10), (-10, 10)])
    opt = GradientDescent(
        target, 
        start_pos=np.array([0.0, 0.0]), 
        learning_rate=0.1,
        projection_strategy=SphereProjection(center=[0, 0], radius=2.0)
    )
    res = opt.run(max_iter=100)
    print(f"Converged: {res.converged} | Final x: {get_x(res)} (Expected: [2. 0.])\n")

if __name__ == "__main__":
    import logging
    # Silence the target initialization logs for cleaner test output
    logging.getLogger("engine.utils").setLevel(logging.WARNING) 
    
    test_unconstrained_newton()
    test_unconstrained_gd()
    test_nonnegative_projection()
    test_box_projection()
    test_hyperplane_projection()
    test_halfspace_projection()
    test_sphere_projection()