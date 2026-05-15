"""Phase bodies: INPUT, COMPUTING, RESULTS."""
import time
import numpy as np
import streamlit as st

from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent
from engine.optimizers.newton_methods import NewtonOptimizer
from engine.optimizers.conjugate_gradient import ConjugateGradient
from engine.optimizers.population_based import GeneticAlgorithm

from ui_interactive.session import (ui_logger, parse_tuple_string,
                                    precompute_f_history, reset_optimization)
from ui_interactive.plots import (build_contour_figure, build_convergence_figure,
                                  build_history_dataframe)


def render_input_phase(method, max_iter, kwargs):
    lib = st.session_state.func_lib

    def on_preset_change():
        sel = st.session_state.preset_selector
        if sel != "-- Custom / Manual --":
            data = lib.functions[sel]
            st.session_state.form_expr = data["expr"]
            st.session_state.form_bounds = data["bounds"]
            st.session_state.form_start = data["start_pos"]

            st.session_state.persistent_expr = data["expr"]
            st.session_state.persistent_bounds = data["bounds"]
            st.session_state.persistent_start = data["start_pos"]

    col_pre1, col_pre2, _ = st.columns([2, 1, 1])
    preset_names = ["-- Custom / Manual --"] + list(lib.functions.keys())

    with col_pre1:
        st.selectbox("Load Function Preset", preset_names,
                     key="preset_selector", on_change=on_preset_change)

    with col_pre2:
        st.write("")
        st.write("")
        sel = st.session_state.get("preset_selector", "-- Custom / Manual --")
        if sel != "-- Custom / Manual --" and not lib.functions[sel].get("is_default", False):
            if st.button("🗑️ Delete Preset", use_container_width=True):
                lib.delete(sel)
                st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.text_input("Target Function f(x)", value=st.session_state.persistent_expr, key="form_expr")
    c2.text_input("Bounds", value=st.session_state.persistent_bounds, key="form_bounds")
    c3.text_input("Starting Point (Single-Agent only)", value=st.session_state.persistent_start,
                  key="form_start", disabled=(method == "Genetic Algorithm"))

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        st.write("")
        start_clicked = st.button("Start Optimization", type="primary", use_container_width=True)

    with col_btn2:
        new_preset_name = st.text_input("New Preset Name",
                                        placeholder="e.g. My Trap Func",
                                        label_visibility="collapsed")

    with col_btn3:
        if st.button("💾 Save as Preset", use_container_width=True):
            if new_preset_name and new_preset_name != "-- Custom / Manual --":
                lib.save(new_preset_name, st.session_state.form_expr,
                         st.session_state.form_bounds, st.session_state.form_start)
                st.success(f"Saved '{new_preset_name}'!")
                time.sleep(1)
                st.rerun()

    if start_clicked:
        _handle_start_click(method, max_iter, kwargs)


def _handle_start_click(method, max_iter, kwargs):
    st.session_state.persistent_expr = st.session_state.form_expr
    st.session_state.persistent_bounds = st.session_state.form_bounds
    st.session_state.persistent_start = st.session_state.form_start

    ui_logger.info(f"User initiated 'Start Optimization'. Target: '{st.session_state.form_expr}'")

    bounds = parse_tuple_string(st.session_state.form_bounds)
    if bounds is None:
        ui_logger.warning(f"User input invalid bounds format: {st.session_state.form_bounds}")
        st.error("Invalid bounds format. Please use format: (-5, 5), (-5, 5)")
        return

    try:
        ui_logger.info("Initializing TargetFunction and Optimizer engine.")
        target = TargetFunction(st.session_state.form_expr, bounds=bounds)
        st.session_state.target = target

        is_convex, bad_point = target.check_convexity()
        st.session_state.is_convex = is_convex
        st.session_state.bad_point = bad_point

        if method in ["Gradient Descent", "Newton's Method", "Conjugate Gradient"]:
            start_pos = np.array(parse_tuple_string(st.session_state.form_start))
            if method == "Gradient Descent":
                opt = GradientDescent(target, start_pos=start_pos, **kwargs)
            elif method == "Newton's Method":
                opt = NewtonOptimizer(target, start_pos=start_pos, **kwargs)
            else:
                opt = ConjugateGradient(target, start_pos=start_pos, **kwargs)
        else:
            opt = GeneticAlgorithm(target, **kwargs)

        st.session_state.optimizer = opt
        st.session_state.max_iter = max_iter
        st.session_state.phase = 'COMPUTING'
        ui_logger.info("Transitioning to COMPUTING phase.")
        st.rerun()

    except Exception as e:
        ui_logger.error(f"Initialization failed due to exception: {e}", exc_info=True)
        st.error(f"Initialization failed: {e}")


def render_computing_phase():
    st.markdown("### Optimizing...")
    ui_logger.info("Rendering Progress Bar and starting optimization run loop.")
    progress_bar = st.progress(0.0)

    opt = st.session_state.optimizer
    max_iter = st.session_state.max_iter

    def update_progress(iteration):
        progress_bar.progress(min(iteration / max_iter, 1.0))

    try:
        t0 = time.time()
        results = opt.run(max_iter=max_iter, callback=update_progress)
        t_elapsed = time.time() - t0
        ui_logger.info(f"Optimization loop finished in {t_elapsed:.4f} seconds. "
                       f"Converged: {results.converged}.")

        target = st.session_state.target
        if target is not None and results is not None and results.history is not None:
            ui_logger.info("Precomputing f(x) convergence history for UI animation.")
            st.session_state.results = results
            st.session_state.f_history = precompute_f_history(target, results.history)
            st.session_state.phase = 'RESULTS'
            ui_logger.info("Transitioning to RESULTS phase.")
            st.rerun()
        else:
            ui_logger.error("Optimization results are invalid or missing. Aborting results render.")
            st.error("Optimization failed to produce results. Please try again.")
            st.button("Return to Setup", on_click=reset_optimization)

    except Exception as e:
        ui_logger.error(f"Optimization crashed during execution: {e}", exc_info=True)
        st.error(f"Execution Error: {e}")
        st.warning("Hint: If using Newton's method, the Hessian might be singular. "
                   "If writing math, ensure valid syntax (e.g., use `x*y` not `xy`).")
        st.button("Return to Setup", on_click=reset_optimization)


def render_results_phase():
    res = st.session_state.results
    target = st.session_state.target

    if res is None or target is None:
        ui_logger.warning("Attempted to render RESULTS phase, but session state data was missing.")
        st.warning("Results data missing. Please run the optimization again.")
        return

    _render_results_header(res, target)
    _render_results_visualizations(res, target)


def _render_results_header(res, target):
    if st.session_state.is_convex is True:
        conv_str = "✅ **Convexity:** PSD Confirmed"
    elif st.session_state.is_convex is False:
        bp = st.session_state.bad_point
        bp_str = np.round(bp, 3) if bp is not None else "Unknown"
        conv_str = f"❌ **Convexity:** Failed at {bp_str}"
    else:
        conv_str = "⚠️ **Convexity:** Unknown/Error"

    c1, c2 = st.columns([4, 1])
    c1.markdown(f"**Optimization Complete** ({st.session_state.method}) "
                f"&nbsp; | &nbsp; {conv_str}")
    c1.write(f"Converged: `{res.converged}` in `{res.iterations}` steps. "
             f"Execution time: `{res.execution_time:.4f}s`")

    best_idx = np.argmin([target.evaluate(p) for p in res.final_population])
    x_vals = res.final_population[best_idx]
    x_str = ", ".join([f"{val:.4f}" for val in x_vals])
    c1.write(f"**Final x:** `[{x_str}]` | **Final f(x):** `{res.final_f:.6f}`")
    c2.button("New Optimization", on_click=reset_optimization, use_container_width=True)
    ui_logger.info(f"Rendered RESULTS box successfully. Final best f(x): {res.final_f:.6f}")


def _render_results_visualizations(res, target):
    frame = st.slider("Generation / Iteration", 0, len(res.history) - 1, 0)
    ui_logger.info(f"Rendering visualization for frame: {frame}/{len(res.history) - 1}")

    col_graph1, col_graph2, col_hist = st.columns([3, 3, 2])

    method = st.session_state.method
    is_single_agent = method in ["Gradient Descent", "Newton's Method", "Conjugate Gradient"]
    contour_mode = 'trajectory' if is_single_agent else 'swarm'
    history_mode = 'single' if is_single_agent else 'population'

    with col_graph1:
        bounds = target.bounds
        if not bounds or len(bounds) != 2:
            st.warning("Contour plotting requires exactly 2 dimensions.")
        else:
            opt = st.session_state.optimizer
            proj_strat = getattr(opt, 'projection_strategy', None)
            has_restarts = any(step.get("restart") for step in res.history)
            show_restarts = (st.checkbox("Show restart markers", value=True,
                                         key="show_restart_markers")
                             if has_restarts else True)
            fig = build_contour_figure(target, proj_strat, res.history, frame,
                                       contour_mode, show_restarts=show_restarts)
            st.plotly_chart(fig, use_container_width=True)

    with col_graph2:
        fig = build_convergence_figure(st.session_state.f_history, frame)
        st.plotly_chart(fig, use_container_width=True)

    with col_hist:
        st.markdown("**Optimization Log**")
        df = build_history_dataframe(target, res.history, st.session_state.f_history,
                                     history_mode, frame)
        st.dataframe(df, height=450, use_container_width=True)
