import numpy as np
from engine.models import TargetFunction
from engine.optimizers.conjugate_gradient import ConjugateGradient
from engine.strategies.cg_beta import FletcherReeves, PolakRibiere, HestenesStiefel
from engine.strategies.step_size import ExactLineSearch, StrongWolfeLineSearch
from engine.strategies.stopping import StepSizeCriterion


def test_quadratic_convergence():
    """Each β formula should solve a 2D quadratic in ≤ 2 steps with exact line search."""
    target = TargetFunction("x**2 + 2*y**2", bounds=[(-5, 5), (-5, 5)])
    for name, beta in [("FR", FletcherReeves()),
                       ("PR", PolakRibiere()),
                       ("HS", HestenesStiefel())]:
        opt = ConjugateGradient(
            target, start_pos=np.array([3.0, 4.0]),
            beta_strategy=beta,
            step_size_strategy=ExactLineSearch(),
            stopping_criterion=StepSizeCriterion(tol=1e-8),
        )
        r = opt.run(max_iter=10)
        print(f"  {name}: iters={r.iterations}, f={r.final_f:.2e}, x={r.final_population[0]}")


def test_quadratic_beta_equivalence():
    """All three β should produce numerically identical iterates on a strict quadratic."""
    target = TargetFunction("x**2 + 2*y**2", bounds=[(-5, 5), (-5, 5)])
    histories = {}
    for name, beta in [("FR", FletcherReeves()),
                       ("PR", PolakRibiere()),
                       ("HS", HestenesStiefel())]:
        opt = ConjugateGradient(
            target, start_pos=np.array([3.0, 4.0]),
            beta_strategy=beta,
            step_size_strategy=ExactLineSearch(),
            stopping_criterion=StepSizeCriterion(tol=1e-12),
        )
        opt.run(max_iter=10)
        histories[name] = np.array([h["population"][0] for h in opt.results.history])
    diff_fr_pr = np.max(np.abs(histories["FR"] - histories["PR"]))
    diff_fr_hs = np.max(np.abs(histories["FR"] - histories["HS"]))
    print(f"  max|FR - PR| = {diff_fr_pr:.2e}")
    print(f"  max|FR - HS| = {diff_fr_hs:.2e}")
    print(f"  expected: both ≤ 1e-8 on a strict quadratic")


def test_rosenbrock_pr_wolfe():
    """PR + Wolfe + restart_every=n on the classical CG benchmark."""
    target = TargetFunction("(1 - x)**2 + 100 * (y - x**2)**2",
                            bounds=[(-5, 5), (-5, 5)])
    opt = ConjugateGradient(
        target, start_pos=np.array([-1.2, 1.0]),
        beta_strategy=PolakRibiere(),
        step_size_strategy=StrongWolfeLineSearch(),
        stopping_criterion=StepSizeCriterion(tol=1e-6),
        restart_every=None,  # default → n = 2
    )
    r = opt.run(max_iter=200)
    print(f"  iters={r.iterations}, converged={r.converged}, f={r.final_f:.2e}")


def test_restart_demo():
    """
    Contrast restart-on vs restart-off on a non-quadratic. Restart-on should
    converge cleanly; restart-off typically either stalls or, more often,
    builds a non-descent direction that Strong Wolfe rejects (we raise rather
    than silently degrade — see ``step_size.py``). Either outcome is a
    concrete artifact for the lab report.
    """
    target = TargetFunction("(1 - x)**2 + 100 * (y - x**2)**2",
                            bounds=[(-5, 5), (-5, 5)])
    for label, restart_every in [("restart=ON  (every n)", None),
                                 ("restart=OFF (no restart)", 0)]:
        opt = ConjugateGradient(
            target, start_pos=np.array([-1.2, 1.0]),
            beta_strategy=PolakRibiere(),
            step_size_strategy=StrongWolfeLineSearch(),
            stopping_criterion=StepSizeCriterion(tol=1e-6),
            restart_every=restart_every,
        )
        try:
            r = opt.run(max_iter=200)
            print(f"  {label}: iters={r.iterations}, converged={r.converged}, f={r.final_f:.2e}")
        except RuntimeError as e:
            iters_done = len(opt.results.history)
            print(f"  {label}: ABORTED after {iters_done} iters — Wolfe rejected the CG direction")
            print(f"     ({type(e).__name__}: direction was no longer a descent direction)")


if __name__ == "__main__":
    import logging
    logging.getLogger("engine.utils").setLevel(logging.WARNING)
    print("--- Quadratic convergence (≤ n steps expected) ---")
    test_quadratic_convergence()
    print("\n--- β equivalence on quadratic ---")
    test_quadratic_beta_equivalence()
    print("\n--- Rosenbrock with PR + Wolfe ---")
    test_rosenbrock_pr_wolfe()
    print("\n--- Restart demo (on vs off) ---")
    test_restart_demo()
