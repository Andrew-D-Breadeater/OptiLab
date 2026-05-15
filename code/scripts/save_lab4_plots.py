"""
Run CG on the Lab 4 demo functions and dump contour/convergence PNGs + history CSVs.

Reuses the same plot builders as the Streamlit UI so the figures match what
the live demo shows. Saves into Reports/OMnOD/lab4/figures/<run-name>/.

Run from the repo root:
    .venv/bin/python code/scripts/save_lab4_plots.py
"""
from pathlib import Path
import sys

import numpy as np

from engine.models import TargetFunction
from engine.optimizers.conjugate_gradient import ConjugateGradient
from engine.strategies.cg_beta import HestenesStiefel
from engine.strategies.step_size import ExactLineSearch
from engine.strategies.stopping import GradientNormCriterion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui_interactive"))
from plots import (  # noqa: E402
    build_contour_figure,
    build_convergence_figure,
    build_history_dataframe,
)
from session import precompute_f_history  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parents[2] / "Reports" / "OMnOD" / "lab4" / "figures"


RUNS = [
    {
        "name": "tilted_ellipse",
        "expr": "10*x**2 + y**2 + 5*x*y",
        "bounds": [(-5, 5), (-5, 5)],
        "start": [4.0, 4.0],
        "restart_every": None,
        "max_iter": 50,
    },
    {
        "name": "labfunc15",
        "expr": "12*x**2 + 18*y**2 + 3*z**2 - 0.01*x*z + x - y",
        "bounds": [(-5, 5), (-5, 5), (-5, 5)],
        "start": [4.0, 4.0, 4.0],
        "restart_every": None,
        "max_iter": 50,
    },
    {
        "name": "labfunc15_no_restart",
        "expr": "12*x**2 + 18*y**2 + 3*z**2 - 0.01*x*z + x - y",
        "bounds": [(-5, 5), (-5, 5), (-5, 5)],
        "start": [4.0, 4.0, 4.0],
        "restart_every": 0,
        "max_iter": 200,
    },
    {
        "name": "himmelblau_with_restart",
        "expr": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "bounds": [(-5, 5), (-5, 5)],
        "start": [0.0, 0.0],
        "restart_every": None,
        "max_iter": 200,
    },
    {
        "name": "himmelblau_no_restart",
        "expr": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
        "bounds": [(-5, 5), (-5, 5)],
        "start": [0.0, 0.0],
        "restart_every": 0,
        "max_iter": 500,
    },
]


def run_one(cfg: dict) -> None:
    out_dir = OUT_ROOT / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target = TargetFunction(cfg["expr"], bounds=cfg["bounds"])
    opt = ConjugateGradient(
        target,
        start_pos=np.array(cfg["start"]),
        beta_strategy=HestenesStiefel(),
        step_size_strategy=ExactLineSearch(),
        stopping_criterion=GradientNormCriterion(tol=1e-6),
        restart_every=cfg["restart_every"],
    )
    results = opt.run(max_iter=cfg["max_iter"])

    f_history = precompute_f_history(target, results.history)
    last = len(results.history) - 1

    if len(target.bounds) == 2:
        fig_contour = build_contour_figure(
            target, opt.projection_strategy, results.history,
            frame=last, mode="trajectory", show_restarts=True,
        )
        fig_contour.write_image(str(out_dir / "contour.png"),
                                width=900, height=700, scale=2)

    fig_conv = build_convergence_figure(f_history, frame=last)
    fig_conv.write_image(str(out_dir / "convergence.png"),
                         width=900, height=700, scale=2)

    df = build_history_dataframe(target, results.history, f_history,
                                 mode="single", frame=last)
    df.index.name = "iter"
    df.to_csv(out_dir / "history.csv")

    print(f"[{cfg['name']}] iters={results.iterations} "
          f"converged={results.converged} f*={results.final_f:.6g} "
          f"x*={np.round(opt.population[0], 4).tolist()}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cfg in RUNS:
        run_one(cfg)


if __name__ == "__main__":
    main()
