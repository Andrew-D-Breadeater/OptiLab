"""
Smoke scripts for ``SpiralOptimizer``. Print-based per ``claude/tests.md`` —
visual inspection, not pytest assertions.

Run from the repo root:

    python code/tests/test_spiral.py
"""
import numpy as np
from engine.models import TargetFunction
from engine.initializers import HaltonInitializer, RandomInitializer
from engine.strategies.spiral_rotation import (
    FixedAngleRotation,
    PeriodicDescentRotation,
    StochasticRotation,
    _build_rotation_matrix,
)
from engine.strategies.spiral_radius import (
    FixedRadius,
    PrecisionBasedRadius,
    StochasticRadius,
    AdaptiveRadius,
)
from engine.optimizers.spiral import (
    SpiralOptimizer,
    NoRestart,
    PeriodicKeepBestRestart,
)
from engine.strategies.stopping import MaxGenerationsCriterion


def test_base_spiral():
    print("\n--- Test 1: Base spiral (Rosenbrock, FixedAngleRotation + FixedRadius) ---")
    target = TargetFunction("(1-x)**2 + 100*(y-x**2)**2", bounds=[(-2, 2), (-2, 2)])
    opt = SpiralOptimizer(
        target_function=target,
        population_size=20,
        initializer=HaltonInitializer(),
        rotation_strategy=FixedAngleRotation(theta=np.pi / 4),
        radius_strategy=FixedRadius(r=0.95),
    )
    results = opt.run(max_iter=200)
    print(f"  best f = {results.final_f:.6f}")
    print(f"  pivot  = {opt.center}")


def test_periodic_descent():
    print("\n--- Test 2: §17 periodic descent (Rastrigin, k_max=200) ---")
    target = TargetFunction(
        "20 + (x**2 - 10*cos(2*pi*x)) + (y**2 - 10*cos(2*pi*y))",
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    )
    opt = SpiralOptimizer(
        target_function=target,
        population_size=20,
        initializer=HaltonInitializer(),
        rotation_strategy=PeriodicDescentRotation(),
        radius_strategy=PrecisionBasedRadius(delta=1e-3, k_max=200),
    )
    results = opt.run(max_iter=200)
    print(f"  r = delta^(1/k_max) = {opt.radius.r:.6f}")
    print(f"  best f = {results.final_f:.6f}")


def test_high_dim_sphere():
    print("\n--- Test 3: High-dim (5D) sphere — §16 product builds correctly ---")
    target = TargetFunction(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        bounds=[(-5, 5)] * 5,
    )
    opt = SpiralOptimizer(
        target_function=target,
        population_size=30,
        initializer=HaltonInitializer(),
        rotation_strategy=FixedAngleRotation(theta=np.pi / 6),
        radius_strategy=FixedRadius(r=0.92),
    )
    results = opt.run(max_iter=200)
    print(f"  best f = {results.final_f:.6f}  (true min = 0)")
    print(f"  pivot  = {np.round(opt.center, 3)}")


def test_oob_handling():
    print("\n--- Test 4: OOB → f=+inf, pivot never moves to OOB point ---")
    target = TargetFunction("x**2 + y**2", bounds=[(-1, 1), (-1, 1)])
    opt = SpiralOptimizer(
        target_function=target,
        population_size=4,
        initializer=HaltonInitializer(),
        rotation_strategy=FixedAngleRotation(theta=np.pi / 4),
        radius_strategy=FixedRadius(r=0.95),
    )
    # Force the entire population OOB except one point at (0.5, 0.5).
    opt.population = np.array([
        [10.0, 10.0],
        [-10.0, -10.0],
        [10.0, -10.0],
        [0.5, 0.5],
    ])
    f = opt._evaluate_with_oob()
    print(f"  fitnesses = {f}  (three should be +inf)")
    opt._update_center(f)
    print(f"  pivot     = {opt.center}  (should be [0.5, 0.5])")


def test_cache_equivalence():
    print("\n--- Test 5: Cached vs uncached transform produce identical trajectories ---")
    target = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    np.random.seed(0)

    # Run with caching (static rotation + static radius)
    opt_cached = SpiralOptimizer(
        target_function=target,
        population_size=10,
        initializer=HaltonInitializer(),
        rotation_strategy=FixedAngleRotation(theta=np.pi / 4),
        radius_strategy=FixedRadius(r=0.9),
    )
    pop0 = opt_cached.population.copy()

    # Run without caching: same params, but disable the cache after construction.
    opt_uncached = SpiralOptimizer(
        target_function=target,
        population_size=10,
        initializer=HaltonInitializer(),
        rotation_strategy=FixedAngleRotation(theta=np.pi / 4),
        radius_strategy=FixedRadius(r=0.9),
    )
    opt_uncached.population = pop0.copy()
    opt_uncached._cached_S = None  # force the per-step recomputation path

    r1 = opt_cached.run(max_iter=20)
    r2 = opt_uncached.run(max_iter=20)
    diff = np.max(np.abs(r1.final_population - r2.final_population))
    print(f"  max final-population diff = {diff:.2e}  (should be ~0)")


def test_periodic_descent_orbit():
    """
    The §17 circulant matrix is defined directly (not as a product of plane
    rotations), so for N >= 3 it differs from ``FixedAngleRotation`` even at
    the textbook θ — they coincide only in 2D. The property that matters is
    the **orbit**: applying R(θ) gives the sequence
    ``d, R d, R² d, ..., -d, -R d, ...`` and returns to ``d`` after 2N steps.
    """
    print("\n--- Test 6: PeriodicDescentRotation orbit closes after 2N steps ---")
    for N in (2, 3, 4, 5):
        R = PeriodicDescentRotation().get_matrix(N, 0)
        d = np.arange(1, N + 1, dtype=float)
        v = d.copy()
        v_half = None
        for k in range(1, 2 * N + 1):
            v = R @ v
            if k == N:
                v_half = v.copy()
        cycle_err = np.max(np.abs(v - d))
        antipode_err = np.max(np.abs(v_half + d)) if v_half is not None else float('nan')
        print(f"  N={N}: ‖R^(2N)d − d‖_∞ = {cycle_err:.2e}, ‖R^N d + d‖_∞ = {antipode_err:.2e}")

    print("\n--- Test 6b: 2D — FixedAngleRotation(θ) matches PeriodicDescentRotation ---")
    theta_2d = ((-1) ** 2) * np.pi / 2
    R_fixed = _build_rotation_matrix(np.array([theta_2d]), 2)
    R_periodic = PeriodicDescentRotation().get_matrix(2, 0)
    print(f"  max |R_fixed − R_periodic| = {np.max(np.abs(R_fixed - R_periodic)):.2e}")


def test_stochastic_distribution():
    print("\n--- Test 7: Stochastic θ/r — sample means close to configured centers ---")
    np.random.seed(0)
    rot = StochasticRotation(theta=np.pi / 4, sigma=0.05)
    rad = StochasticRadius(r_l=0.90, r_u=0.99)

    # Pull θ out of repeated matrix builds via R[0,1] = -sin(θ) in a 2D plane:
    thetas = []
    rs = []
    for k in range(2000):
        R = rot.get_matrix(2, k)
        thetas.append(np.arctan2(R[1, 0], R[0, 0]))
        rs.append(rad.get_r(k, None))
    print(f"  θ mean = {np.mean(thetas):.4f}  (target {np.pi/4:.4f})")
    print(f"  r mean = {np.mean(rs):.4f}  (target {(0.90 + 0.99) / 2:.4f})")


def test_adaptive_radius_extremes():
    print("\n--- Test 8: AdaptiveRadius — best→r_u, worst→r_l ---")
    rad = AdaptiveRadius(r_l=0.85, r_u=0.99, c1=1.0)
    fitnesses = np.array([0.0, 1.0, 10.0, 100.0])
    rs = rad.get_r(0, fitnesses)
    print(f"  fitnesses = {fitnesses}")
    print(f"  r-values  = {np.round(rs, 4)}  (best {rs[0]:.4f} ≈ r_u={rad.r_u}, worst {rs[-1]:.4f} ↓ r_l={rad.r_l})")


def test_periodic_restart():
    print("\n--- Test 9: PeriodicKeepBestRestart preserves μn best at pre-rotation positions ---")
    target = TargetFunction("x**2 + y**2", bounds=[(-5, 5), (-5, 5)])
    opt = SpiralOptimizer(
        target_function=target,
        population_size=10,
        initializer=HaltonInitializer(),
        rotation_strategy=FixedAngleRotation(theta=np.pi / 4),
        radius_strategy=FixedRadius(r=0.9),
        restart_strategy=PeriodicKeepBestRestart(rounds_length=5, keep_ratio=0.2),
    )
    results = opt.run(max_iter=12)
    # Check that the history shows a population jump at iteration 5 and 10.
    pops = [h["population"] for h in results.history]
    for i in (5, 10):
        if i < len(pops):
            jump = np.max(np.linalg.norm(pops[i] - pops[i - 1], axis=1))
            print(f"  iter {i}: max ‖x_new - x_old‖ = {jump:.3f}  (should be large)")
    print(f"  best f after restarts = {results.final_f:.6f}")


if __name__ == "__main__":
    test_base_spiral()
    test_periodic_descent()
    test_high_dim_sphere()
    test_oob_handling()
    test_cache_equivalence()
    test_periodic_descent_orbit()
    test_stochastic_distribution()
    test_adaptive_radius_extremes()
    test_periodic_restart()
