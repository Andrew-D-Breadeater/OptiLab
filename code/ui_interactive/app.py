# ui_interactive/app.py
import ast
import time
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sympy as sp

from engine.models import TargetFunction
from engine.optimizers.gradient_methods import GradientDescent
from engine.optimizers.newton_methods import NewtonOptimizer
from engine.optimizers.population_based import GeneticAlgorithm
from engine.initializers import RandomInitializer, HaltonInitializer
from engine.strategies.selection import (ElitismSelection, TournamentSelection, 
                                         RouletteWheelSelection, RankSelection)
from engine.strategies.crossover import UniformCrossover, NonUniformCrossover
from engine.strategies.mutation import RealCodedMutation
from engine.function_library import FunctionLibrary
from engine.strategies.projections import (
    NoProjection, NonNegativeProjection, BoxProjection, 
    HyperplaneProjection, HalfSpaceProjection, SphereProjection, CustomNonlinearProjection
)

# --- Logging Setup ---
ui_logger = logging.getLogger("ui_app")
ui_logger.setLevel(logging.INFO)
# Prevent duplicate handlers on Streamlit reruns
if not ui_logger.handlers:
    handler = logging.FileHandler("ui.log")
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] UI: %(message)s'))
    ui_logger.addHandler(handler)

# --- Initialization ---
st.set_page_config(layout="wide", page_title="Optimization Engine")

if 'func_lib' not in st.session_state:
    st.session_state.func_lib = FunctionLibrary()
    
# The "Vault": These keys persist because they aren't linked to widgets
if 'persistent_expr' not in st.session_state:
    st.session_state.persistent_expr = "x**2 + y**2"
if 'persistent_bounds' not in st.session_state:
    st.session_state.persistent_bounds = "(-5, 5), (-5, 5)"
if 'persistent_start' not in st.session_state:
    st.session_state.persistent_start = "4.0, 4.0"

if 'phase' not in st.session_state:
    ui_logger.info("Initializing new user session.")
    st.session_state.phase = 'INPUT'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'f_history' not in st.session_state:
    st.session_state.f_history =[]
if 'target' not in st.session_state:
    st.session_state.target = None
if 'is_convex' not in st.session_state:
    st.session_state.is_convex = None
if 'bad_point' not in st.session_state:
    st.session_state.bad_point = None

# --- UI Helpers ---
def reset_optimization():
    ui_logger.info("User clicked 'New Optimization'. Resetting application state.")
    st.session_state.phase = 'INPUT'
    
    # Reset the dropdown to manual so it doesn't conflict with persisted text
    st.session_state.preset_selector = "-- Custom / Manual --"
    
    # Clear the results but KEEP the form_expr, form_bounds, and form_start
    st.session_state.results = None
    st.session_state.f_history = []
    st.session_state.is_convex = None
    st.session_state.bad_point = None

def parse_tuple_string(s):
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (tuple, list)):
            return list(val)
        return None
    except:
        return None

# --- Sidebar: Method Specific Options ---
with st.sidebar:
    st.title("Method Specific Options")
    
    method = st.selectbox("Optimisation Method", ["Gradient Descent", "Newton's Method", "Genetic Algorithm"], key="method")
    
    max_iter = st.number_input("Max Iterations", value=100)
    tol = st.number_input("Tolerance (\u03B5)", value=1e-4, format="%.1e")
    
    kwargs = {}
    
    if method in ["Gradient Descent", "Newton's Method"]:
        default_lr = 0.1 if method == "Gradient Descent" else 1.0
        kwargs['learning_rate'] = st.number_input("Learning Rate", value=default_lr)
        kwargs['stopping_criterion'] = st.selectbox("Stopping Criterion", ['gradient_norm', 'step_size'])
        kwargs['use_line_search'] = st.checkbox("Backtracking Line Search")
        kwargs['use_exact_line_search'] = st.checkbox("Exact Line Search")
        
        if method == "Gradient Descent":
            kwargs['use_ravine'] = st.checkbox("Use Ravine Method")
            if kwargs['use_ravine']:
                kwargs['ravine_step_size'] = st.number_input("Ravine Step Size", value=0.5)
            
            # --- Projection Strategy Instantiation ---
            st.markdown("---")
            st.markdown("**Constraint / Projection**")
            proj_type = st.selectbox("Boundary Projection",["None", "Non-Negative", "Box (Bounds)", "Hyperplane", "Half-Space", "Sphere", "Custom Non-linear"])
            
            if proj_type == "None":
                kwargs['projection_strategy'] = NoProjection()
                
            elif proj_type == "Non-Negative":
                kwargs['projection_strategy'] = NonNegativeProjection()
                
            elif proj_type == "Box (Bounds)":
                bounds_str = st.session_state.get('form_bounds', "(-5, 5), (-5, 5)")
                parsed_bounds = parse_tuple_string(bounds_str)
                if parsed_bounds:
                    kwargs['projection_strategy'] = BoxProjection(parsed_bounds)
                else:
                    st.sidebar.error("Invalid Bounds format in main area.")
                    kwargs['projection_strategy'] = NoProjection()
                    
            elif proj_type in ["Hyperplane", "Half-Space"]:
                c_str = st.text_input("Normal Vector (c)", value="1.0, 1.0")
                b_val = st.number_input("Scalar (b)", value=4.0)
                c_vec = parse_tuple_string(c_str)
                if c_vec:
                    if proj_type == "Hyperplane":
                        kwargs['projection_strategy'] = HyperplaneProjection(c=c_vec, b=b_val)
                    else:
                        kwargs['projection_strategy'] = HalfSpaceProjection(c=c_vec, b=b_val)
                else:
                    st.sidebar.error("Invalid normal vector format.")
                    kwargs['projection_strategy'] = NoProjection()
                    
            elif proj_type == "Sphere":
                center_str = st.text_input("Center", value="0.0, 0.0")
                r_val = st.number_input("Radius", value=2.0, min_value=0.01)
                center_vec = parse_tuple_string(center_str)
                if center_vec:
                    kwargs['projection_strategy'] = SphereProjection(center=center_vec, radius=r_val)
                else:
                    st.sidebar.error("Invalid center format.")
                    kwargs['projection_strategy'] = NoProjection()

            elif proj_type == "Custom Non-linear":
                st.info("Write one constraint per line (e.g., `y - x**3 >= 0`).")
                c_text = st.text_area("Constraints", value="y - x**2 >= 0\nx >= 0\ny >= 0")
                c_lines =[line.strip() for line in c_text.split('\n') if line.strip()]
                
                # Safely extract variables from the current string
                try:
                    expr_str = st.session_state.get('form_expr', 'x')
                    sympy_expr = sp.sympify(expr_str)
                    variables = sorted([s.name for s in sympy_expr.free_symbols])
                    
                    # Assign directly to projection_strategy!
                    kwargs['projection_strategy'] = CustomNonlinearProjection(c_lines, variables)
                except Exception as e:
                    st.sidebar.error(f"Waiting for valid target function... ({e})")
                    kwargs['projection_strategy'] = NoProjection()
            
    elif method == "Genetic Algorithm":
        kwargs['population_size'] = st.number_input("Population Size", value=50)
        kwargs['stopping_criterion'] = st.selectbox("Stopping Criterion", ['stagnation', 'degeneration', 'max_generations'])
        
        init_choice = st.selectbox("Initializer", ["Random", "Halton"])
        kwargs['initializer'] = RandomInitializer() if init_choice == "Random" else HaltonInitializer()
        
        sel_choice = st.selectbox("Selection", ["Tournament", "Elitism", "Roulette", "Rank"])
        if sel_choice == "Tournament":
            kwargs['selection_strategy'] = TournamentSelection(tournament_size=3)
        elif sel_choice == "Elitism":
            kwargs['selection_strategy'] = ElitismSelection()
        elif sel_choice == "Roulette":
            kwargs['selection_strategy'] = RouletteWheelSelection()
        else:
            kwargs['selection_strategy'] = RankSelection()
            
        cross_choice = st.selectbox("Crossover", ["Uniform", "Non-Uniform"])
        kwargs['crossover_strategy'] = UniformCrossover() if cross_choice == "Uniform" else NonUniformCrossover()
        
        kwargs['mutation_strategy'] = RealCodedMutation(sigma=0.2)
        
        st.markdown("**Coefficients**")
        col1, col2, col3 = st.columns(3)
        kwargs['phi_sel'] = col1.number_input("\u03D5 sel", value=0.2, step=0.1)
        kwargs['phi_cross'] = col2.number_input("\u03D5 cross", value=0.6, step=0.1)
        kwargs['phi_mut'] = col3.number_input("\u03D5 mut", value=0.2, step=0.1)

# --- Top Main Area (Input / Progress / Results) ---
control_area = st.container(border=True)

with control_area:
    if st.session_state.phase == 'INPUT':
        lib = st.session_state.func_lib
        
        # Callback to update the "Vault" and widgets when a preset is selected
        def on_preset_change():
            sel = st.session_state.preset_selector
            if sel != "-- Custom / Manual --":
                data = lib.functions[sel]
                # Update both the widget keys and the persistent vault
                st.session_state.form_expr = data["expr"]
                st.session_state.form_bounds = data["bounds"]
                st.session_state.form_start = data["start_pos"]
                
                st.session_state.persistent_expr = data["expr"]
                st.session_state.persistent_bounds = data["bounds"]
                st.session_state.persistent_start = data["start_pos"]
                
        # --- UI LAYOUT ---

        # Preset Selection Row
        col_pre1, col_pre2, col_pre3 = st.columns([2, 1, 1])
        preset_names = ["-- Custom / Manual --"] + list(lib.functions.keys())
        
        with col_pre1:
            st.selectbox("Load Function Preset", preset_names, key="preset_selector", on_change=on_preset_change)
            
        with col_pre2:
            st.write("")
            st.write("")
            sel = st.session_state.get("preset_selector", "-- Custom / Manual --")
            if sel != "-- Custom / Manual --" and not lib.functions[sel].get("is_default", False):
                if st.button("🗑️ Delete Preset", use_container_width=True):
                    lib.delete(sel)
                    st.rerun()

        # Math Input Row: We set 'value' from the vault
        c1, c2, c3 = st.columns(3)
        func_str = c1.text_input("Target Function f(x)", value=st.session_state.persistent_expr, key="form_expr")
        bounds_str = c2.text_input("Bounds", value=st.session_state.persistent_bounds, key="form_bounds")
        start_str = c3.text_input("Starting Point (Single-Agent only)", value=st.session_state.persistent_start, 
                                  key="form_start", disabled=(method == "Genetic Algorithm"))
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            st.write("")
            start_clicked = st.button("Start Optimization", type="primary", use_container_width=True)
            
        with col_btn2:
            new_preset_name = st.text_input("New Preset Name", placeholder="e.g. My Trap Func", label_visibility="collapsed")
            
        with col_btn3:
            if st.button("💾 Save as Preset", use_container_width=True):
                if new_preset_name and new_preset_name != "-- Custom / Manual --":
                    lib.save(new_preset_name, st.session_state.form_expr, st.session_state.form_bounds, st.session_state.form_start)
                    st.success(f"Saved '{new_preset_name}'!")
                    time.sleep(1)
                    st.rerun()

        if start_clicked:
            # Save current inputs before changing phase
            st.session_state.persistent_expr = st.session_state.form_expr
            st.session_state.persistent_bounds = st.session_state.form_bounds
            st.session_state.persistent_start = st.session_state.form_start
            
            ui_logger.info(f"User initiated 'Start Optimization'. Target: '{st.session_state.form_expr}'")
            
            bounds = parse_tuple_string(st.session_state.form_bounds)
            if bounds is None:
                ui_logger.warning(f"User input invalid bounds format: {bounds_str}")
                st.error("Invalid bounds format. Please use format: (-5, 5), (-5, 5)")
            else:
                try:
                    ui_logger.info("Initializing TargetFunction and Optimizer engine.")
                    target = TargetFunction(st.session_state.form_expr, bounds=bounds)
                    st.session_state.target = target
                    
                    # Run Convexity Check
                    is_convex, bad_point = target.check_convexity()
                    st.session_state.is_convex = is_convex
                    st.session_state.bad_point = bad_point
                    
                    if method in ["Gradient Descent", "Newton's Method"]:
                        start_pos = np.array(parse_tuple_string(st.session_state.form_start))
                        if method == "Gradient Descent":
                            opt = GradientDescent(target, start_pos=start_pos, **kwargs)
                        else:
                            opt = NewtonOptimizer(target, start_pos=start_pos, **kwargs)
                    else:
                        opt = GeneticAlgorithm(target, **kwargs)
                    
                    st.session_state.optimizer = opt
                    st.session_state.max_iter = max_iter
                    st.session_state.tol = tol
                    st.session_state.phase = 'COMPUTING'
                    ui_logger.info("Transitioning to COMPUTING phase.")
                    st.rerun()
                    
                except Exception as e:
                    ui_logger.error(f"Initialization failed due to exception: {e}", exc_info=True)
                    st.error(f"Initialization failed: {e}")

    elif st.session_state.phase == 'COMPUTING':
        st.markdown("### Optimizing...")
        ui_logger.info("Rendering Progress Bar and starting optimization run loop.")
        progress_bar = st.progress(0.0)
        
        opt = st.session_state.optimizer
        max_iter = st.session_state.max_iter
        
        def update_progress(iteration):
            progress_bar.progress(min(iteration / max_iter, 1.0))
            
        try:
            # Run optimization (blocking, but updates progress bar via callback)
            t0 = time.time()
            results = opt.run(max_iter=max_iter, tol=st.session_state.tol, callback=update_progress)
            t_elapsed = time.time() - t0
            ui_logger.info(f"Optimization loop finished in {t_elapsed:.4f} seconds. Converged: {results.converged}.")
            
            # Precompute convergence history to prevent lag during animation
            if st.session_state.target is not None and results is not None and results.history is not None:
                ui_logger.info("Precomputing f(x) convergence history for UI animation.")
                f_hist = []
                for step in results.history:
                    f_vals =[st.session_state.target.evaluate(p) for p in step["population"]]
                    f_hist.append(np.min(f_vals))
                
                st.session_state.results = results
                st.session_state.f_history = f_hist
                st.session_state.phase = 'RESULTS'
                ui_logger.info("Transitioning to RESULTS phase.")
                st.rerun()
            else:
                ui_logger.error("Optimization results are invalid or missing. Aborting results render.")
                st.error("Optimization failed to produce results. Please try again.")
                st.button("Return to Setup", on_click=reset_optimization)

        except Exception as e:
            # Catch mathematical/evaluation errors gracefully
            ui_logger.error(f"Optimization crashed during execution: {e}", exc_info=True)
            st.error(f"Execution Error: {e}")
            st.warning("Hint: If using Newton's method, the Hessian might be singular. If writing math, ensure valid syntax (e.g., use `x*y` not `xy`).")
            st.button("Return to Setup", on_click=reset_optimization)

    elif st.session_state.phase == 'RESULTS':
        res = st.session_state.results
        
        if res is not None and st.session_state.target is not None:
            # Build Convexity string
            if st.session_state.is_convex is True:
                conv_str = "✅ **Convexity:** PSD Confirmed"
            elif st.session_state.is_convex is False:
                bp_str = np.round(st.session_state.bad_point, 3) if st.session_state.bad_point is not None else "Unknown"
                conv_str = f"❌ **Convexity:** Failed at {bp_str}"
            else:
                conv_str = "⚠️ **Convexity:** Unknown/Error"

            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**Optimization Complete** ({st.session_state.method}) &nbsp; | &nbsp; {conv_str}")
            c1.write(f"Converged: `{res.converged}` in `{res.iterations}` steps. Execution time: `{res.execution_time:.4f}s`")
            
            target = st.session_state.target
            best_idx = np.argmin([target.evaluate(p) for p in res.final_population])
            x_vals = res.final_population[best_idx]
            x_str = ", ".join([f"{val:.4f}" for val in x_vals])
            
            c1.write(f"**Final x:** `[{x_str}]` | **Final f(x):** `{res.final_f:.6f}`")
            c2.button("New Optimization", on_click=reset_optimization, use_container_width=True)
            ui_logger.info(f"Rendered RESULTS box successfully. Final best f(x): {res.final_f:.6f}")
        else:
            ui_logger.warning("Attempted to render RESULTS phase, but session state data was missing.")
            st.warning("Results data missing. Please run the optimization again.")

# --- Graphs & History Area ---
if st.session_state.phase == 'RESULTS' and st.session_state.results and st.session_state.target:
    res = st.session_state.results
    target = st.session_state.target
    
    # Animation Slider
    frame = st.slider("Generation / Iteration", 0, len(res.history) - 1, 0)
    ui_logger.info(f"Rendering visualization for frame: {frame}/{len(res.history) - 1}")
    
    # Use 3 columns: Contour, Convergence, and History Dataframe
    col_graph1, col_graph2, col_hist = st.columns([3, 3, 2])
    
    # 1. Function Contour Plot
    with col_graph1:
        bounds = target.bounds
        if not bounds or len(bounds) != 2:
            st.warning("Contour plotting requires exactly 2 dimensions.")
        else:
            x_range = np.linspace(bounds[0][0], bounds[0][1], 300)
            y_range = np.linspace(bounds[1][0], bounds[1][1], 300)
            X, Y = np.meshgrid(x_range, y_range)
            
            try:
                Z = target.evaluate([X, Y])
                if np.isscalar(Z):
                    Z = np.full_like(X, Z, dtype=float)
            except Exception as e:
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i,j] = target.evaluate([X[i,j], Y[i,j]])
            
            fig_contour = go.Figure(data=[go.Contour(x=x_range, y=y_range, z=Z, colorscale='Viridis')])
                        
            # --- Shading for Boundaries ---
            opt = st.session_state.optimizer
            proj_strat = getattr(opt, 'projection_strategy', None)
            
            if proj_strat is not None:
                mask = proj_strat.get_feasibility_mask(X, Y)
                
                # If the mask is not entirely True, there are forbidden zones
                if not np.all(mask):
                    # Invert the mask: Allowed = 0.0, Blocked = 1.0
                    shadow_z = (~mask).astype(float)
                    
                    fig_contour.add_trace(go.Contour(
                        x=x_range, y=y_range, z=shadow_z,
                        showscale=False,
                        # 0.0 (Allowed) -> Completely transparent
                        # 1.0 (Blocked) -> Shaded dark grey with 20% opacity
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0.2)']],
                        hoverinfo='skip'
                    ))
            
            # --- Trajectory / Scatter logic ---
            # Extract current population points ...
            
            # --- Trajectory / Scatter logic ---
            if st.session_state.method in ["Gradient Descent", "Newton's Method"]:
                # Trajectory plot for single-agent methods
                path = [step["population"][0] for step in res.history[:frame+1]]
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                fig_contour.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', 
                                                 marker=dict(color='red', size=8), 
                                                 line=dict(color='red', width=2)))
            else:
                # Scatter swarm for Population methods
                current_pop = res.history[frame]["population"]
                px = current_pop[:, 0]
                py = current_pop[:, 1]
                fig_contour.add_trace(go.Scatter(x=px, y=py, mode='markers', marker=dict(color='red', size=8)))
            
            fig_contour.update_layout(title="Search Space", margin=dict(l=0, r=0, t=30, b=0), height=450)
            st.plotly_chart(fig_contour, use_container_width=True)
            
    # 2. Convergence Rate Plot
    with col_graph2:
        f_hist = st.session_state.f_history
        iters = list(range(len(f_hist)))
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=iters, y=f_hist, mode='lines', line=dict(color='blue', width=2), name='f(x)'))
        fig_conv.add_trace(go.Scatter(x=[frame], y=[f_hist[frame]], mode='markers', marker=dict(color='red', size=10), showlegend=False))
        
        fig_conv.update_layout(title="Convergence Rate", xaxis_title="Iteration", yaxis_title="Best f(x)", margin=dict(l=0, r=0, t=30, b=0), height=450)
        st.plotly_chart(fig_conv, use_container_width=True)

        # 3. History View Panel
    with col_hist:
        st.markdown("**Optimization Log**")
        
        hist_data =[]
        # Slice up to the current frame for clean animation sync
        for i, step in enumerate(res.history[:frame + 1]):
            if st.session_state.method in["Gradient Descent", "Newton's Method"]:
                pt = step["population"][0]
                row = {"Iter": i} 
                
                # Dynamically map each coordinate to its variable name (x, y, z, x1, etc.)
                for j, var_name in enumerate(target.variables):
                    row[var_name] = round(pt[j], 4)
                    
                row["f(x)"] = round(st.session_state.f_history[i], 6)
                hist_data.append(row)
            else:
                hist_data.append({"Iter": i, "Best f(x)": round(st.session_state.f_history[i], 6)})
                
        df_hist = pd.DataFrame(hist_data)
        st.dataframe(df_hist, height=450, use_container_width=True)
        
#with st.sidebar.expander("🔍 Live Session State Debugger"):
#    st.write(st.session_state)